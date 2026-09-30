use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{AdamW, ModuleT, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::data::{self, CLASS_NAMES, NUM_CLASSES, Split};
use crate::metrics::Confusion;
use crate::model::Cnn;

#[derive(clap::Args)]
pub struct TrainArgs {
    #[arg(long, default_value_t = 12)]
    epochs: usize,
    #[arg(long, default_value_t = 64)]
    batch_size: usize,
    /// Peak learning rate; decays to zero on a cosine schedule.
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,
    #[arg(long, default_value_t = 1e-4)]
    weight_decay: f64,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value = "data/cache")]
    cache_dir: PathBuf,
    #[arg(long, default_value = "checkpoints")]
    out_dir: PathBuf,
    ///Hopefully this will work, if not then I will have to figure out a way to make it work ;-;
    #[arg(long)]
    cpu: bool,
}

pub fn device(force_cpu: bool) -> Result<Device> {
    let device = if force_cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
    if device.is_cuda() {
        // TF32 tensor-core matmuls for the im2col convolutions.
        candle_core::cuda::set_gemm_reduced_precision_f32(true);
    }
    Ok(device)
}

pub fn load_model(checkpoint: Option<&Path>, device: &Device) -> Result<(Cnn, VarMap)> {
    let mut varmap = VarMap::new();
    let model = Cnn::new(VarBuilder::from_varmap(&varmap, DType::F32, device))?;
    if let Some(path) = checkpoint {
        varmap.load(path)?;
    }
    Ok((model, varmap))
}

fn class_weights(counts: &[usize; NUM_CLASSES], device: &Device) -> Result<Tensor> {
    let total: usize = counts.iter().sum();
    let w: Vec<f32> = counts
        .iter()
        .map(|&n| (total as f32 / (NUM_CLASSES as f32 * n.max(1) as f32)).sqrt())
        .collect();
    Ok(Tensor::from_vec(w, NUM_CLASSES, device)?)
}

fn weighted_cross_entropy(logits: &Tensor, labels: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let log_probs = candle_nn::ops::log_softmax(logits, D::Minus1)?;
    let nll = log_probs.gather(&labels.unsqueeze(1)?, 1)?.squeeze(1)?.neg()?;
    let w = weights.index_select(labels, 0)?;
    Ok((nll * &w)?.sum_all()?.div(&w.sum_all()?)?)
}

pub fn evaluate(model: &Cnn, split: &Split, batch_size: usize, device: &Device) -> Result<Confusion> {
    let mut conf = Confusion::new();
    let idx: Vec<usize> = (0..split.len()).collect();
    for chunk in idx.chunks(batch_size) {
        let (x, y) = data::batch(split, chunk, None, device)?;
        let pred = model.forward_t(&x, false)?.argmax(D::Minus1)?.to_vec1::<u32>()?;
        for (t, p) in y.to_vec1::<u32>()?.into_iter().zip(pred) {
            conf.add(t, p);
        }
    }
    Ok(conf)
}

pub fn train(args: TrainArgs) -> Result<()> {
    let device = device(args.cpu)?;
    println!("device: {device:?}");

    let train = Split::load(&args.cache_dir.join("train.safetensors"))?;
    let val = Split::load(&args.cache_dir.join("val.safetensors"))?;
    let counts = train.class_counts();
    let weights = class_weights(&counts, &device)?;
    println!("train: {} images, val: {} images", train.len(), val.len());
    for (name, w) in CLASS_NAMES.iter().zip(weights.to_vec1::<f32>()?) {
        println!("  loss weight {name:<14} {w:.2}");
    }

    let (model, varmap) = load_model(None, &device)?;
    let params = ParamsAdamW { lr: args.lr, weight_decay: args.weight_decay, ..Default::default() };
    let mut opt = AdamW::new(varmap.all_vars(), params)?;

    std::fs::create_dir_all(&args.out_dir)?;
    let best_path = args.out_dir.join("best.safetensors");
    let steps_per_epoch = train.len() / args.batch_size;
    let total_steps = steps_per_epoch * args.epochs;
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut best_f1 = -1.0;

    for epoch in 1..=args.epochs {
        let start = Instant::now();
        order.shuffle(&mut rng);
        let mut loss_sum = 0.0;
        // Drop the ragged tail: a tiny final batch gives BatchNorm garbage statistics.
        for (i, chunk) in order.chunks_exact(args.batch_size).enumerate() {
            let step = (epoch - 1) * steps_per_epoch + i;
            let progress = step as f64 / total_steps as f64;
            opt.set_learning_rate(args.lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()));

            let (x, y) = data::batch(&train, chunk, Some(&mut rng), &device)?;
            let loss = weighted_cross_entropy(&model.forward_t(&x, true)?, &y, &weights)?;
            opt.backward_step(&loss)?;
            loss_sum += loss.to_scalar::<f32>()? as f64;

            if (i + 1) % 100 == 0 {
                print!("\r  epoch {epoch} step {}/{steps_per_epoch} loss {:.4}", i + 1, loss_sum / (i + 1) as f64);
                use std::io::Write;
                std::io::stdout().flush()?;
            }
        }

        let conf = evaluate(&model, &val, args.batch_size, &device)?;
        let f1 = conf.macro_f1();
        let improved = f1 > best_f1;
        if improved {
            best_f1 = f1;
            varmap.save(&best_path)?;
        }
        println!(
            "\repoch {epoch:>3}/{}  loss {:.4}  val acc {:.4}  val macro-F1 {:.4}  {:.0}s{}",
            args.epochs,
            loss_sum / steps_per_epoch as f64,
            conf.accuracy(),
            f1,
            start.elapsed().as_secs_f64(),
            if improved { "  * saved" } else { "" },
        );
    }

    varmap.save(args.out_dir.join("last.safetensors"))?;
    println!("\nbest val macro-F1 {best_f1:.4}; weights in {}", best_path.display());
    let (best, _vm) = load_model(Some(&best_path), &device)?;
    evaluate(&best, &val, args.batch_size, &device)?.print_report();
    Ok(())
}

pub fn predict(images: &[PathBuf], checkpoint: &Path, force_cpu: bool) -> Result<()> {
    let device = device(force_cpu)?;
    let (model, _vm) = load_model(Some(checkpoint), &device)?;
    for path in images {
        let x = data::load_image(path)?
            .into_iter()
            .map(|p| p as f32 / 127.5 - 1.0)
            .collect::<Vec<_>>();
        let x = Tensor::from_vec(x, (1, 1, data::IMG_SIZE, data::IMG_SIZE), &device)?;
        let probs = candle_nn::ops::softmax(&model.forward_t(&x, false)?, D::Minus1)?
            .squeeze(0)?
            .to_vec1::<f32>()?;
        let mut ranked: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        println!("{}", path.display());
        for (c, p) in ranked.iter().take(3) {
            println!("  {:<14} {:>6.2}%", CLASS_NAMES[*c], p * 100.0);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_ce_matches_candle_ce_with_uniform_weights() -> Result<()> {
        let dev = Device::Cpu;
        let logits = Tensor::new(&[[2.0f32, 0.5, -1.0], [0.1, 0.2, 3.0]], &dev)?;
        let labels = Tensor::new(&[0u32, 1], &dev)?;
        let ours = weighted_cross_entropy(&logits, &labels, &Tensor::ones(3, DType::F32, &dev)?)?;
        let reference = candle_nn::loss::cross_entropy(&logits, &labels)?;
        assert!((ours.to_scalar::<f32>()? - reference.to_scalar::<f32>()?).abs() < 1e-6);
        Ok(())
    }

    /// Catches mis-wired gradients: a healthy model memorizes 16 random images quickly.
    /// Runs on the GPU in CUDA builds, where backend bugs have bitten before.
    #[test]
    fn model_overfits_tiny_batch() -> Result<()> {
        let dev = Device::cuda_if_available(0)?;
        let (model, varmap) = load_model(None, &dev)?;
        let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW { lr: 3e-3, ..Default::default() })?;
        let x = Tensor::randn(0f32, 1., (16, 1, 16, 16), &dev)?;
        let y = Tensor::new(&[0u32, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7], &dev)?;
        let weights = Tensor::ones(NUM_CLASSES, DType::F32, &dev)?;
        let mut loss = f32::INFINITY;
        for _ in 0..150 {
            let l = weighted_cross_entropy(&model.forward_t(&x, true)?, &y, &weights)?;
            opt.backward_step(&l)?;
            loss = l.to_scalar::<f32>()?;
        }
        assert!(loss < 0.1, "train-mode loss still {loss}");
        let acc = model.forward_t(&x, false)?.argmax(D::Minus1)?.eq(&y)?.to_dtype(DType::F32)?.mean_all()?;
        assert!(acc.to_scalar::<f32>()? > 0.9, "eval-mode accuracy {acc}");
        Ok(())
    }

    /// candle 0.11.0 computed garbage conv2d weight gradients on CUDA ;-;, We need more Rust devs to save the world.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_conv_weight_grad_matches_cpu() -> Result<()> {
        let x = Tensor::randn(0f32, 1., (4, 8, 12, 12), &Device::Cpu)?;
        let w = Tensor::randn(0f32, 0.1, (16, 8, 3, 3), &Device::Cpu)?;
        let grad_w = |dev: &Device| -> Result<Vec<f32>> {
            let w = candle_core::Var::from_tensor(&w.to_device(dev)?)?;
            let loss = x.to_device(dev)?.conv2d(w.as_tensor(), 1, 1, 1, 1)?.sqr()?.sum_all()?;
            Ok(loss.backward()?.get(w.as_tensor()).unwrap().flatten_all()?.to_vec1::<f32>()?)
        };
        let (cpu, gpu) = (grad_w(&Device::Cpu)?, grad_w(&Device::new_cuda(0)?)?);
        for (a, b) in cpu.iter().zip(&gpu) {
            assert!((a - b).abs() <= 1e-3 * a.abs().max(1.0), "cpu {a} vs cuda {b}");
        }
        Ok(())
    }

    #[test]
    fn model_outputs_one_logit_per_class() -> Result<()> {
        let (model, _vm) = load_model(None, &Device::Cpu)?;
        let x = Tensor::zeros((2, 1, data::IMG_SIZE, data::IMG_SIZE), DType::F32, &Device::Cpu)?;
        assert_eq!(model.forward_t(&x, false)?.dims(), [2, NUM_CLASSES]);
        Ok(())
    }
}
