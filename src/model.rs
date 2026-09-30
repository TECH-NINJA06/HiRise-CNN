use candle_core::{Result, Tensor, Var};
use candle_nn::{Conv2d, Conv2dConfig, Dropout, Init, Linear, Module, ModuleT, VarBuilder, conv2d_no_bias, linear};

use crate::data::NUM_CLASSES;


/// Running statistics are VarMap entries, so they are saved with the checkpoint.
struct BatchNorm2d {
    weight: Tensor,
    bias: Tensor,
    running_mean: Var,
    running_var: Var,
}

impl BatchNorm2d {
    const EPS: f64 = 1e-5;
    const MOMENTUM: f64 = 0.1;

    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let shape = (1, channels, 1, 1);
        Ok(Self {
            weight: vb.get_with_hints(shape, "weight", Init::Const(1.))?,
            bias: vb.get_with_hints(shape, "bias", Init::Const(0.))?,
            running_mean: Var::from_tensor(&vb.get_with_hints(shape, "running_mean", Init::Const(0.))?)?,
            running_var: Var::from_tensor(&vb.get_with_hints(shape, "running_var", Init::Const(1.))?)?,
        })
    }

    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let channel_mean = |t: &Tensor| t.mean_keepdim(0)?.mean_keepdim(2)?.mean_keepdim(3);
        let (centered, var) = if train {
            let mean = channel_mean(xs)?;
            let centered = xs.broadcast_sub(&mean)?;
            let var = channel_mean(&centered.sqr()?)?;
            let m = Self::MOMENTUM;
            let blend = |running: &Var, batch: &Tensor| -> Result<()> {
                running.set(&((running.as_tensor() * (1. - m))? + (batch.detach() * m)?)?)
            };
            blend(&self.running_mean, &mean)?;
            blend(&self.running_var, &var)?;
            (centered, var)
        } else {
            (xs.broadcast_sub(&self.running_mean.as_detached_tensor())?, self.running_var.as_detached_tensor())
        };
        let scale = ((var + Self::EPS)?.sqrt()?.recip()? * &self.weight)?;
        centered.broadcast_mul(&scale)?.broadcast_add(&self.bias)
    }
}

struct ConvBnRelu {
    conv: Conv2d,
    bn: BatchNorm2d,
}

impl ConvBnRelu {
    fn new(c_in: usize, c_out: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig { padding: 1, ..Default::default() };
        Ok(Self {
            conv: conv2d_no_bias(c_in, c_out, 3, cfg, vb.pp("conv"))?,
            bn: BatchNorm2d::new(c_out, vb.pp("bn"))?,
        })
    }

    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        self.bn.forward_t(&self.conv.forward(xs)?, train)?.relu()
    }
}

pub struct Cnn {
    stem: ConvBnRelu,
    blocks: Vec<(ConvBnRelu, ConvBnRelu)>,
    dropout: Dropout,
    head: Linear,
}

impl Cnn {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let stem = ConvBnRelu::new(1, 32, vb.pp("stem"))?;
        let widths = [(32, 64), (64, 128), (128, 256)];
        let blocks = widths
            .iter()
            .enumerate()
            .map(|(i, &(c_in, c_out))| {
                let vb = vb.pp(format!("block{i}"));
                Ok((ConvBnRelu::new(c_in, c_out, vb.pp("a"))?, ConvBnRelu::new(c_out, c_out, vb.pp("b"))?))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            stem,
            blocks,
            dropout: Dropout::new(0.3),
            head: linear(256, NUM_CLASSES, vb.pp("head"))?,
        })
    }
}

impl ModuleT for Cnn {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let mut xs = self.stem.forward_t(xs, train)?.max_pool2d(2)?;
        for (a, b) in &self.blocks {
            xs = b.forward_t(&a.forward_t(&xs, train)?, train)?.max_pool2d(2)?;
        }
        let xs = xs.mean((2, 3))?;
        self.head.forward(&self.dropout.forward(&xs, train)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn batch_norm_normalizes_and_tracks_stats_in_varmap() -> Result<()> {
        let dev = Device::Cpu;
        let varmap = VarMap::new();
        let bn = BatchNorm2d::new(3, VarBuilder::from_varmap(&varmap, DType::F32, &dev))?;
        let xs = Tensor::randn(0f32, 1., (8, 3, 5, 5), &dev)?.affine(4.0, 2.0)?;

        let ys = bn.forward_t(&xs, true)?;
        let mean = ys.mean_keepdim(0)?.mean_keepdim(2)?.mean_keepdim(3)?.flatten_all()?.to_vec1::<f32>()?;
        let var = ys.sqr()?.mean_keepdim(0)?.mean_keepdim(2)?.mean_keepdim(3)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(mean.iter().all(|m| m.abs() < 1e-4), "{mean:?}");
        assert!(var.iter().all(|v| (v - 1.).abs() < 1e-3), "{var:?}");

        // The running mean must move toward ~2.0 in the VarMap itself, or checkpoints
        // would save the initial zeros and eval-mode predictions would be garbage.
        let saved = varmap.data().lock().unwrap()["running_mean"].as_tensor().flatten_all()?.to_vec1::<f32>()?;
        assert!(saved.iter().all(|m| (m - 0.2).abs() < 0.1), "{saved:?}");
        Ok(())
    }
}
