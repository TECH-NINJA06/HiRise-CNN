# hirise_cnn

A from-scratch convolutional neural network in Rust, built on
[candle](https://github.com/huggingface/candle), that classifies landmarks in Mars orbital
images from the HiRISE camera (Mars Reconnaissance Orbiter) into 8 classes: craters, dunes,
slope streaks, impact ejecta, polar "swiss cheese" terrain, "spiders", and everything else.

- [Quick start](#quick-start)
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Results](#results)
- [The candle CUDA bug (and the patch)](#the-candle-cuda-bug-and-the-patch)
- [Performance notes](#performance-notes)
- [Windows + CUDA setup](#windows--cuda-setup)
- [Project layout and tests](#project-layout-and-tests)
- [Limitations and next steps](#limitations-and-next-steps)

## Quick start

```sh
# 1. Download hirise-map-proj-v3_2.zip (911 MB) from https://doi.org/10.5281/zenodo.4002935
#    and extract it into data/ (skip the __MACOSX/ folder), giving
#    data/hirise-map-proj-v3_2/{map-proj-v3_2/, labels-map-proj_v3_2_train_val_test.txt}

# 2. Decode + resize every image once into data/cache/{train,val,test}.safetensors
build-cuda.cmd run -- prepare

# 3. Train (writes checkpoints/best.safetensors and checkpoints/last.safetensors)
build-cuda.cmd run -- train

# 4. Evaluate and predict
build-cuda.cmd run -- eval --split test
build-cuda.cmd run -- predict data/hirise-map-proj-v3_2/map-proj-v3_2/ESP_013788_1910_RED-0295.jpg

# Tests (on the GPU, including GPU-vs-CPU gradient checks)
build-cuda.cmd test
```

`build-cuda.cmd <cargo subcommand> [args]` runs `cargo <subcommand> --release --features cuda`
inside the Visual Studio environment that CUDA needs (see [setup](#windows--cuda-setup)).
Without a GPU, use plain cargo instead, e.g. `cargo run --release -- train` (much slower).

| command   | what it does | useful flags |
|-----------|--------------|--------------|
| `prepare` | reads the official split file, drops duplicate rows, decodes and resizes all images | `--data-dir`, `--cache-dir` |
| `train`   | trains, evaluates on val after every epoch, keeps the best macro-F1 checkpoint | `--epochs 12`, `--batch-size 64`, `--lr 1e-3`, `--weight-decay 1e-4`, `--seed 42`, `--out-dir`, `--cpu` |
| `eval`    | per-class precision/recall/F1 and a confusion matrix | `--split test\|val\|train`, `--checkpoint` |
| `predict` | top-3 classes with probabilities for image files | `--checkpoint` |

The input size is the constant `IMG_SIZE` in `src/data.rs` (currently 128). Checkpoints are
tied to it: after changing it, rerun `prepare` and retrain.

## Dataset

**Mars orbital image (HiRISE) labeled data set, version 3.2** (Doran, Wagstaff et al.,
NASA/JPL, [Zenodo 4002935](https://doi.org/10.5281/zenodo.4002935)). 10,815 landmarks were
cropped from 232 HiRISE browse images and resized to 227x227 grayscale; 9,022 of them were
augmented six ways (rotations by 90/180/270 degrees, horizontal and vertical flips, random
brightness), giving 64,947 images.

| id | class          | what it is |
|----|----------------|------------|
| 0  | other          | anything that is not one of the landmarks below |
| 1  | crater         | crater at least 1/5 of the image wide with at least half its rim visible |
| 2  | dark dune      | fully defrosted sand dunes |
| 3  | slope streak   | dark flow-like streaks where dust slid down a slope |
| 4  | bright dune    | frost-covered dunes, often with dark defrosting spots |
| 5  | impact ejecta  | material blasted out around a (possibly invisible) impact |
| 6  | swiss cheese   | south-polar CO2-ice terrain full of pits |
| 7  | spider         | south-polar radial troughs from CO2 sublimation |

### What the split file actually contains

`prepare` uses the official `labels-map-proj_v3_2_train_val_test.txt` (note the underscore
in `proj_v3_2`, unlike the name given on the Zenodo page). Checked before training:

- **No leakage.** Train, val and test come from disjoint HiRISE source images (152 / 50 / 30),
  so augmented copies of one landmark never straddle splits.
- **Test is un-augmented.** It holds exactly the 1,793 landmarks that were never augmented;
  train and val contain the rotated/flipped/brightened copies.
- **Duplicate rows.** The file repeats 2,079 train rows and 784 val rows (the authors'
  oversampling). `prepare` keeps each file once; the loss is class-weighted instead.

After de-duplication:

| class         | train          | val            | test          |
|---------------|---------------:|---------------:|--------------:|
| other         | 40,019 (81.7%) | 11,221 (79.2%) | 1,482 (82.7%) |
| crater        |  3,598  (7.3%) |  1,337  (9.4%) |    89  (5.0%) |
| dark dune     |    616  (1.3%) |     84  (0.6%) |    66  (3.7%) |
| slope streak  |    924  (1.9%) |    602  (4.2%) |    49  (2.7%) |
| bright dune   |  1,561  (3.2%) |     77  (0.5%) |    16  (0.9%) |
| impact ejecta |    343  (0.7%) |    126  (0.9%) |     7  (0.4%) |
| swiss cheese  |  1,281  (2.6%) |    511  (3.6%) |    42  (2.3%) |
| spider        |    637  (1.3%) |    217  (1.5%) |    42  (2.3%) |
| **total**     | **48,979**     | **14,175**     | **1,793**     |

Because about 82% of everything is `other`, a model that always answers `other` scores 82.7%
test accuracy while being useless. **Macro-F1** (the unweighted mean of per-class F1) is the
metric that matters and the one used for model selection.

## Model

A small VGG-style CNN (`src/model.rs`), ~1.17M trainable parameters:

```
input 1 x 128 x 128 (grayscale, scaled to [-1, 1])
stem    conv3x3   1 -> 32,  BN, ReLU                  128x128 -> maxpool -> 64x64
block0  2x [conv3x3 -> 64,  BN, ReLU]                  64x64  -> maxpool -> 32x32
block1  2x [conv3x3 -> 128, BN, ReLU]                  32x32  -> maxpool -> 16x16
block2  2x [conv3x3 -> 256, BN, ReLU]                  16x16  -> maxpool -> 8x8
global average pool -> 256 -> dropout 0.3 -> linear -> 8 logits
```

Convolutions have no bias (BatchNorm follows). `BatchNorm2d` is a small custom module rather
than `candle_nn::BatchNorm` for speed (see [performance](#performance-notes)); its running
mean/variance live in the `VarMap`, so they are saved with checkpoints, and a unit test
checks exactly that.

## Training

- **Loss:** cross-entropy weighted per class by `sqrt(N / (8 * n_class))`, a dampened inverse
  frequency (full inverse frequency over-weights classes with a few hundred examples):

  | other | crater | dark dune | slope streak | bright dune | impact ejecta | swiss cheese | spider |
  |------:|-------:|----------:|-------------:|------------:|--------------:|-------------:|-------:|
  | 0.39  | 1.30   | 3.15      | 2.57         | 1.98        | 4.22          | 2.19         | 3.10   |

- **Augmentation** (on the fly, CPU): one of the 8 rotations/flips of the square (orbital
  images have no canonical "up"), plus contrast x[0.9, 1.1] and brightness +-0.1.
- **Optimizer:** AdamW, lr 1e-3, weight decay 1e-4, cosine decay to 0 over all steps;
  batch 64, last partial batch dropped (a tiny batch gives BatchNorm garbage statistics).
- **Epochs:** 12. Each epoch already contains 7 augmented copies of most landmarks, so this is
  many passes over each landmark.
- **Model selection:** after every epoch, evaluate on val; keep the checkpoint with the best
  val macro-F1 in `checkpoints/best.safetensors`. Test is only used for the final report.
- **Precision:** fp32 with TF32 tensor-core matmuls on the GPU.

## Results

Two runs with identical settings (12 epochs, seed 42), differing only in input size. Both
keep the checkpoint with the best val macro-F1, which was the final epoch in both runs.

| input | val acc | val macro-F1 | test acc | test macro-F1 | time/epoch | checkpoint |
|-------|--------:|-------------:|---------:|--------------:|-----------:|------------|
| 96px  | 0.876   | **0.602**    | 0.883    | **0.586**     | ~3.5 min   | `checkpoints/96px/best.safetensors` |
| 128px | 0.848   | 0.570        | 0.868    | 0.583         | ~6.2 min   | `checkpoints/best.safetensors` |
| always `other` | 0.792 | ~0.11 | 0.827  | ~0.11         |            | |

Times are on the RTX 4060 Laptop GPU (the 96px figure is from its first epoch). To evaluate
the 96px checkpoint, set `IMG_SIZE = 96`, rerun `prepare`, and pass `--checkpoint`.

### Per-class F1

| class         | val 96px | val 128px | test 96px | test 128px | test support |
|---------------|---------:|----------:|----------:|-----------:|-------------:|
| other         | 0.931    | 0.914     | 0.938     | 0.927      | 1,482 |
| crater        | **0.771**| 0.645     | **0.725** | 0.606      | 89 |
| dark dune     | 0.362    | 0.369     | 0.779     | 0.782      | 66 |
| slope streak  | 0.494    | 0.492     | 0.467     | 0.467      | 49 |
| bright dune   | 0.617    | 0.538     | 0.711     | 0.696      | 16 |
| impact ejecta | 0.456    | 0.382     | 0.348     | 0.400      | 7 |
| swiss cheese  | 0.810    | 0.829     | 0.720     | 0.789      | 42 |
| spider        | 0.378    | 0.392     | **0.000** | **0.000**  | 42 |

### What the comparison shows

- **96px is the better model**: higher val macro-F1 (the selection metric), equal on test,
  and about twice as fast to train.
- **Higher resolution did not help slope streaks** (val F1 0.494 vs. 0.492). The idea behind
  the 128px run was that thin streaks were being lost at 96px; that was wrong. About half of
  all slope streaks are still called `other` at both sizes (321 of 602 on val at 128px).
- **Craters got clearly worse at 128px** (val 0.771 to 0.645), mostly crater/`other` confusion.
  A likely reason: the last conv layer's receptive field is about 70 pixels whatever the input
  size, so at 96px it covers most of a crop and at 128px only about half. A crater is defined
  by its whole rim (at least 1/5 of the image wide), so seeing less of it hurts.
- **At 128px, val macro-F1 was still rising** at the last epoch (0.558 to 0.570), so a longer
  run might close some of the gap. Each resolution was trained once, so differences of a few
  hundredths are partly run-to-run noise.
- **Dark dune** is poor on val (0.37) but good on test (0.78): val has only 84 dark dunes,
  and many val impact ejecta and spiders get predicted as dark dune (both models), which
  hurts its precision.

### 128px training history

| epoch | train loss | val acc | val macro-F1 |
|------:|-----------:|--------:|-------------:|
| 1     | 1.279      | 0.653   | 0.213 |
| 2     | 0.985      | 0.694   | 0.274 |
| 3     | 0.818      | 0.724   | 0.346 |
| 4     | 0.687      | 0.751   | 0.348 |
| 5     | 0.584      | 0.749   | 0.381 |
| 6     | 0.476      | 0.776   | 0.461 |
| 7     | 0.406      | 0.760   | 0.442 |
| 8     | 0.351      | 0.833   | 0.516 |
| 9     | 0.306      | 0.824   | 0.536 |
| 10    | 0.267      | 0.838   | 0.553 |
| 11    | 0.247      | 0.843   | 0.558 |
| 12    | 0.234      | 0.848   | 0.570 |

Train loss is the class-weighted loss averaged over the epoch, with augmentation and dropout
on. Val accuracy starts low because the class weights push early predictions toward the rare
classes. The 96px run's per-epoch log was not saved (it printed to a terminal); its epoch 1
val macro-F1 was 0.257.

<details>
<summary>Test confusion matrices (rows = truth, columns = prediction)</summary>

96px:

```
                other  crate  darkd  slope  brigh  impac  swiss  spide
other            1398     17     11     30     10      7      6      3
crater             19     62      0      3      1      4      0      0
dark dune          10      0     51      0      2      0      0      3
slope streak       20      0      3     25      0      1      0      0
bright dune         0      0      0      0     16      0      0      0
impact ejecta       1      0      0      0      0      4      0      2
swiss cheese       11      0      0      0      0      0     27      4
spider             39      3      0      0      0      0      0      0
```

128px:

```
                other  crate  darkd  slope  brigh  impac  swiss  spide
other            1373     32     13     40     10      9      4      1
crater             28     53      0      3      1      4      0      0
dark dune           9      0     52      0      3      0      0      2
slope streak       18      1      2     28      0      0      0      0
bright dune         0      0      0      0     16      0      0      0
impact ejecta       1      0      0      0      0      5      0      1
swiss cheese        8      0      0      0      0      0     30      4
spider             42      0      0      0      0      0      0      0
```

</details>

### Why spider scores 0 on test

All 42 test spiders come from a **single** HiRISE image, `PSP_003470_0930_RED`, where they
appear as **bright branching troughs on dark ground**. Training spiders (91 unique landmarks
from 11 source images, 57 of them from `ESP_013049_0950_RED`) are **dark spots and fans on bright, cracked terrain**:
the opposite contrast. The model has never seen a spider like the test ones and calls them
`other`. This is a property of the split, not a bug. Global intensity inversion would be
the obvious augmentation, but it is not safe here: dark vs. bright dune is defined by
brightness. Note that this was found by inspecting the test set; any fix for it should be
judged on val only, or the test score stops being an honest estimate.

## The candle CUDA bug (and the patch)

**candle-core 0.11.0 computes wrong conv2d weight gradients on CUDA.** Everything else was
correct, so training "worked" but the conv filters never learned.

**Symptom.** After two epochs on the GPU, val macro-F1 was 0.11 (always `other`) and the
weighted training loss (~1.63-1.78) barely beat the ~1.70 that a model outputting only the
class priors would get.

**Diagnosis.**

1. A test that makes the model memorize 16 random images passed on the CPU, so the model,
   loss and optimizer were wired correctly.
2. A grid of cached training images per class looked right, so labels matched images.
3. Comparing gradients of a single conv layer on CPU vs. GPU at the model's real shapes:
   the input gradient matched (relative error ~1e-7), but the **weight gradient had relative
   error ~1.41 = sqrt(2)**, i.e. uncorrelated with the true gradient, at every shape and with
   or without TF32.
4. Isolating further: CUDA `conv2d` is only wrong when the **kernel** argument is
   non-contiguous. Backprop computes the weight gradient as
   `arg.transpose(0, 1).conv2d(grad.transpose(0, 1))`, so it always hits that case.

**Root cause.** In `candle-core/src/cuda_backend/mod.rs`, the non-contiguous-kernel branch of
`conv2d` copies the kernel into a contiguous buffer `kernel_c`, then passes the original
strided `kernel` (with its original offset) to the matmul anyway. Upstream fixed it in
[huggingface/candle#3893](https://github.com/huggingface/candle/issues/3893), not yet in a
release. (The same pattern remains in the conv1d path upstream; this project doesn't use conv1d.)

**Patch.** `vendor/candle-core` is candle-core 0.11.0 with exactly the upstream conv2d fix
(marked `[hirise_cnn]` in the source), wired in with `[patch.crates-io]` in `Cargo.toml`. The
test `cuda_conv_weight_grad_matches_cpu` guards it, and the memorization test now runs on the
GPU in CUDA builds. Remove the patch once a candle release contains the fix.

**Effect.** Epoch-1 val macro-F1 went from 0.110 to 0.257, and the model trained normally.

## Performance notes

Measured on an RTX 4060 Laptop GPU (8 GB), candle 0.11, CUDA 12.9, no cuDNN, batch 64
(forward + backward + AdamW), with a game running in the background:

| variant                                 | ms/step |
|-----------------------------------------|--------:|
| 128px, `candle_nn::BatchNorm`           |     789 |
| 128px, custom `BatchNorm2d`             |     631 |
| 128px, custom `BatchNorm2d`, avg-pool   |     573 |
| 96px,  custom `BatchNorm2d`             |     334 |
| 64px,  custom `BatchNorm2d`             |     150 |

- Most of a step is memory-bound elementwise and reduction work (BatchNorm, ReLU, pooling
  backward), not convolution math, so time scales almost linearly with pixel count.
- `candle_nn::BatchNorm` transposes to (C, N*H*W) and back, costing two full copies per layer;
  the custom `BatchNorm2d` reduces in place: ~20% faster steps.
- TF32 matmuls (`set_gemm_reduced_precision_f32`) gave ~10%.
- Candle's CUDA conv input-gradient (`conv_transpose2d`) is a direct kernel, 2-4x slower than
  the forward im2col convolution for the wider layers.
- Reductions to a single value are slow on CUDA: `sum_all` over a 64x64x64x64 tensor took
  23 ms, versus 2.4 ms for per-channel means over the same tensor.
- Close games and other GPU-heavy apps while training: they compete for the GPU.

## Windows + CUDA setup

What this machine needed, and why:

- **CUDA 12.9 from NVIDIA's redist archives, on `D:\CUDA\v12.9`.** The regular installer
  (via `winget install Nvidia.CUDA --version 12.9`, and again with only the needed components)
  failed with `0xE0E00064`, its disk-space check for `C:`. Instead, the component archives
  `cuda_nvcc`, `cuda_cudart`, `cuda_cccl`, `cuda_nvrtc`, `libcublas` and `libcurand` from
  <https://developer.download.nvidia.com/compute/cuda/redist/> (manifest
  `redistrib_12.9.1.json`) were downloaded, SHA-256 verified and merged into one folder. No
  admin rights needed. CUDA 12.x was chosen because it supports the Visual Studio 2019
  compiler installed here.
- **User environment variables:** `CUDA_PATH=D:\CUDA\v12.9`, and `D:\CUDA\v12.9\bin` on the
  user `PATH` so the cuBLAS/cuRAND/NVRTC DLLs are found at runtime.
- **Visual Studio 2019 C++ tools (`D:\VS`).** candle compiles its CUDA kernels with `nvcc`,
  which on Windows needs `cl.exe`; without it the build fails with
  `nvcc fatal : Cannot find compiler 'cl.exe' in PATH`. `build-cuda.cmd` calls
  `D:\VS\VC\Auxiliary\Build\vcvars64.bat` first.
- **MSVC Rust toolchain.** The default toolchain here is `windows-gnullvm`; this directory
  uses `rustup override set stable-x86_64-pc-windows-msvc`, since the CUDA libraries are MSVC
  import libraries.
- The first CUDA build takes ~6.5 minutes (candle compiles its kernels). A linker warning
  `LNK4098: defaultlib 'LIBCMT' conflicts` comes from candle's statically built MoE kernels,
  which this model does not use; it is harmless.

## Project layout and tests

```
src/main.rs        CLI (prepare / train / eval / predict)
src/data.rs        split parsing, image decoding, safetensors cache, batching + augmentation
src/model.rs       the CNN and the custom BatchNorm2d
src/train.rs       loss, training loop, evaluation, prediction
src/metrics.rs     confusion matrix, per-class precision/recall/F1, macro-F1
vendor/candle-core candle-core 0.11.0 + CUDA conv2d fix
build-cuda.cmd     cargo wrapper that sets up VS 2019 + CUDA
data/              dataset and cache (git-ignored)
checkpoints/       trained weights and logs (git-ignored)
```

| test | guards against |
|------|----------------|
| `dihedral_yields_eight_distinct_permutations` | broken rotation/flip augmentation |
| `weighted_ce_matches_candle_ce_with_uniform_weights` | a wrong loss |
| `batch_norm_normalizes_and_tracks_stats_in_varmap` | BatchNorm stats not saved, making eval-mode output garbage |
| `model_outputs_one_logit_per_class` | shape errors |
| `model_overfits_tiny_batch` | mis-wired gradients (runs on the GPU in CUDA builds) |
| `cuda_conv_weight_grad_matches_cpu` | the candle bug above (CUDA builds only) |

## Limitations and next steps

- **Default resolution:** the code is set to 128px (the latest run), but the 96px model is
  better on val and twice as fast to train. Switching back means `IMG_SIZE = 96` and `prepare`.
- **Receptive field:** at 128px the network sees only about half of each crop at once. An
  extra conv block with pooling (or a stride-2 stem) would let a 128px model see whole craters.
- **Slope streaks** are the weakest class with plenty of data (about half called `other` at
  either resolution); more resolution was not the fix.
- **Spider generalization:** see above. The class needs more varied training images, or a
  spider-only contrast augmentation judged on val.
- **Tiny test classes:** impact ejecta has 7 test images and bright dune 16, so their test F1
  moves a lot with a single prediction; val (14k images) is the steadier estimate.
- **No resume:** `train` always starts from scratch and overwrites `checkpoints/`.
- **Speed:** cuDNN (candle's `cudnn` feature) should speed up convolutions, but needs a
  separate cuDNN install; it was not tried.
