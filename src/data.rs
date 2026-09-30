
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use image::imageops::FilterType;
use rand::Rng;
use rand::rngs::StdRng;
use rayon::prelude::*;

pub const IMG_SIZE: usize = 128;
pub const CLASS_NAMES: [&str; 8] = [
    "other",
    "crater",
    "dark dune",
    "slope streak",
    "bright dune",
    "impact ejecta",
    "swiss cheese",
    "spider",
];
pub const NUM_CLASSES: usize = CLASS_NAMES.len();

pub const SPLITS: [&str; 3] = ["train", "val", "test"];
pub struct Split {
    pub images: Vec<u8>,
    pub labels: Vec<u8>,
}

impl Split {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn image(&self, i: usize) -> &[u8] {
        let px = IMG_SIZE * IMG_SIZE;
        &self.images[i * px..(i + 1) * px]
    }

    pub fn class_counts(&self) -> [usize; NUM_CLASSES] {
        let mut counts = [0; NUM_CLASSES];
        for &l in &self.labels {
            counts[l as usize] += 1;
        }
        counts
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut tensors = candle_core::safetensors::load(path, &Device::Cpu)
            .with_context(|| format!("loading {} (did you run `prepare`?)", path.display()))?;
        let mut take = |name: &str| tensors.remove(name).with_context(|| format!("{name} missing in cache"));
        let (images, labels) = (take("images")?, take("labels")?);
        let (_, h, w) = images.dims3()?;
        if (h, w) != (IMG_SIZE, IMG_SIZE) {
            bail!("{} holds {h}x{w} images but IMG_SIZE is {IMG_SIZE}; rerun `prepare`", path.display());
        }
        Ok(Self { images: images.flatten_all()?.to_vec1::<u8>()?, labels: labels.to_vec1::<u8>()? })
    }

    fn save(&self, path: &Path) -> Result<()> {
        let n = self.len();
        let tensors = HashMap::from([
            ("images".to_string(), Tensor::from_slice(&self.images, (n, IMG_SIZE, IMG_SIZE), &Device::Cpu)?),
            ("labels".to_string(), Tensor::from_slice(&self.labels, n, &Device::Cpu)?),
        ]);
        candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }
}

pub fn load_image(path: &Path) -> Result<Vec<u8>> {
    let img = image::open(path).with_context(|| format!("opening {}", path.display()))?;
    let img = img.to_luma8();
    let img = image::imageops::resize(&img, IMG_SIZE as u32, IMG_SIZE as u32, FilterType::Triangle);
    Ok(img.into_raw())
}

/// The file repeats some rows (the authors' oversampling); those are dropped since the loss is already class-weighted.
pub fn prepare(data_dir: &Path, cache_dir: &Path) -> Result<()> {
    let labels_path = data_dir.join("labels-map-proj_v3_2_train_val_test.txt");
    let img_dir = data_dir.join("map-proj-v3_2");
    let text = std::fs::read_to_string(&labels_path)
        .with_context(|| format!("reading {}", labels_path.display()))?;

    let mut seen = std::collections::HashSet::new();
    let mut by_split: HashMap<&str, Vec<(&str, u8)>> = HashMap::new();
    for (lineno, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let cols: Vec<&str> = line.split_whitespace().collect();
        let [file, class, split] = cols[..] else {
            bail!("{}:{}: expected `filename class_id set`, got {line:?}", labels_path.display(), lineno + 1);
        };
        let class: u8 = class.parse()?;
        if class as usize >= NUM_CLASSES || !SPLITS.contains(&split) {
            bail!("{}:{}: unexpected class/split in {line:?}", labels_path.display(), lineno + 1);
        }
        if seen.insert(file) {
            by_split.entry(split).or_default().push((file, class));
        }
    }

    std::fs::create_dir_all(cache_dir)?;
    for split in SPLITS {
        let entries = by_split.get(split).map(Vec::as_slice).unwrap_or_default();
        let images: Vec<Vec<u8>> = entries
            .par_iter()
            .map(|(file, _)| load_image(&img_dir.join(file)))
            .collect::<Result<_>>()?;
        let split_data = Split {
            images: images.concat(),
            labels: entries.iter().map(|(_, c)| *c).collect(),
        };
        let path = cache_dir.join(format!("{split}.safetensors"));
        split_data.save(&path)?;
        println!("{split:>5}: {:>6} images -> {}", split_data.len(), path.display());
        print_class_counts(&split_data.class_counts());
    }
    Ok(())
}

pub fn print_class_counts(counts: &[usize; NUM_CLASSES]) {
    let total: usize = counts.iter().sum();
    for (name, &n) in CLASS_NAMES.iter().zip(counts) {
        println!("       {name:<14} {n:>6}  ({:>5.1}%)", 100.0 * n as f64 / total.max(1) as f64);
    }
}

/// Builds an `(N, 1, H, W)` f32 batch scaled to [-1, 1]. With an rng, each image gets a random
/// rotation/flip (orbital imagery has no canonical "up") plus a small brightness/contrast jitter.
pub fn batch(
    split: &Split,
    idx: &[usize],
    mut augment: Option<&mut StdRng>,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let px = IMG_SIZE * IMG_SIZE;
    let mut xs = Vec::with_capacity(idx.len() * px);
    let mut tmp = vec![0u8; px];
    for &i in idx {
        let img = split.image(i);
        match augment.as_deref_mut() {
            Some(rng) => {
                dihedral(img, &mut tmp, IMG_SIZE, rng.random_range(0..8));
                let contrast = rng.random_range(0.9f32..1.1);
                let shift = rng.random_range(-0.1f32..0.1);
                xs.extend(tmp.iter().map(|&p| (p as f32 / 127.5 - 1.0) * contrast + shift));
            }
            None => xs.extend(img.iter().map(|&p| p as f32 / 127.5 - 1.0)),
        }
    }
    let ys: Vec<u32> = idx.iter().map(|&i| split.labels[i] as u32).collect();
    Ok((
        Tensor::from_vec(xs, (idx.len(), 1, IMG_SIZE, IMG_SIZE), device)?,
        Tensor::from_vec(ys, idx.len(), device)?,
    ))
}

/// Applies one of the 8 symmetries of a square: bits 0-1 pick a 90-degree rotation, bit 2 a flip.
fn dihedral(src: &[u8], dst: &mut [u8], s: usize, t: u8) {
    for y in 0..s {
        for x in 0..s {
            let (sx, sy) = match t & 3 {
                0 => (x, y),
                1 => (y, s - 1 - x),
                2 => (s - 1 - x, s - 1 - y),
                _ => (s - 1 - y, x),
            };
            let sx = if t & 4 != 0 { s - 1 - sx } else { sx };
            dst[y * s + x] = src[sy * s + sx];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dihedral_yields_eight_distinct_permutations() {
        let s = 4;
        let src: Vec<u8> = (0..(s * s) as u8).collect();
        let mut seen = std::collections::HashSet::new();
        for t in 0..8 {
            let mut dst = vec![0; s * s];
            dihedral(&src, &mut dst, s, t);
            let mut sorted = dst.clone();
            sorted.sort();
            assert_eq!(sorted, src, "transform {t} is not a permutation");
            seen.insert(dst);
        }
        assert_eq!(seen.len(), 8);
    }
}
