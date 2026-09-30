mod data;
mod metrics;
mod model;
mod train;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Decode and resize the raw dataset into per-split tensor caches.
    Prepare {
        #[arg(long, default_value = "data/hirise-map-proj-v3_2")]
        data_dir: PathBuf,
        #[arg(long, default_value = "data/cache")]
        cache_dir: PathBuf,
    },
    /// Train the CNN, keeping the checkpoint with the best validation macro-F1.
    Train(train::TrainArgs),
    /// Report metrics and a confusion matrix for a checkpoint on one split.
    Eval {
        #[arg(long, default_value = "test")]
        split: String,
        #[arg(long, default_value = "checkpoints/best.safetensors")]
        checkpoint: PathBuf,
        #[arg(long, default_value = "data/cache")]
        cache_dir: PathBuf,
        #[arg(long)]
        cpu: bool,
    },
    /// Classify individual image files.
    Predict {
        #[arg(required = true)]
        images: Vec<PathBuf>,
        #[arg(long, default_value = "checkpoints/best.safetensors")]
        checkpoint: PathBuf,
        #[arg(long)]
        cpu: bool,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Prepare { data_dir, cache_dir } => data::prepare(&data_dir, &cache_dir),
        Cmd::Train(args) => train::train(args),
        Cmd::Eval { split, checkpoint, cache_dir, cpu } => {
            let device = train::device(cpu)?;
            let data = data::Split::load(&cache_dir.join(format!("{split}.safetensors")))?;
            let (model, _vm) = train::load_model(Some(&checkpoint), &device)?;
            println!("{split} split, checkpoint {}", checkpoint.display());
            train::evaluate(&model, &data, 128, &device)?.print_report();
            Ok(())
        }
        Cmd::Predict { images, checkpoint, cpu } => train::predict(&images, &checkpoint, cpu),
    }
}
