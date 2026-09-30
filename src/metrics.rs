//! Confusion matrix and per-class metrics. Accuracy alone is misleading here because "other" dominates the dataset, so model selection uses macro-averaged F1.
use crate::data::{CLASS_NAMES, NUM_CLASSES};

pub struct Confusion {
    /// `m[truth][pred]`
    m: [[usize; NUM_CLASSES]; NUM_CLASSES],
}

pub struct ClassStats {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

impl Confusion {
    pub fn new() -> Self {
        Self { m: [[0; NUM_CLASSES]; NUM_CLASSES] }
    }

    pub fn add(&mut self, truth: u32, pred: u32) {
        self.m[truth as usize][pred as usize] += 1;
    }

    pub fn total(&self) -> usize {
        self.m.iter().flatten().sum()
    }

    pub fn accuracy(&self) -> f64 {
        let correct: usize = (0..NUM_CLASSES).map(|c| self.m[c][c]).sum();
        correct as f64 / self.total().max(1) as f64
    }

    pub fn class_stats(&self, c: usize) -> ClassStats {
        let tp = self.m[c][c] as f64;
        let predicted: usize = (0..NUM_CLASSES).map(|t| self.m[t][c]).sum();
        let support: usize = self.m[c].iter().sum();
        let precision = if predicted > 0 { tp / predicted as f64 } else { 0.0 };
        let recall = if support > 0 { tp / support as f64 } else { 0.0 };
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        ClassStats { precision, recall, f1, support }
    }

    /// Mean F1 over classes that actually appear in the split.
    pub fn macro_f1(&self) -> f64 {
        let present: Vec<f64> = (0..NUM_CLASSES)
            .map(|c| self.class_stats(c))
            .filter(|s| s.support > 0)
            .map(|s| s.f1)
            .collect();
        present.iter().sum::<f64>() / present.len().max(1) as f64
    }

    pub fn print_report(&self) {
        println!("\n{:<14} {:>9} {:>9} {:>9} {:>8}", "class", "precision", "recall", "f1", "support");
        for (c, name) in CLASS_NAMES.iter().enumerate() {
            let s = self.class_stats(c);
            println!("{name:<14} {:>9.3} {:>9.3} {:>9.3} {:>8}", s.precision, s.recall, s.f1, s.support);
        }
        println!("\naccuracy {:.4}   macro-F1 {:.4}   ({} images)", self.accuracy(), self.macro_f1(), self.total());

        println!("\nconfusion matrix (rows = truth, cols = prediction)");
        print!("{:<14}", "");
        for c in 0..NUM_CLASSES {
            print!("{:>7}", abbrev(CLASS_NAMES[c]));
        }
        println!();
        for (t, name) in CLASS_NAMES.iter().enumerate() {
            print!("{name:<14}");
            for p in 0..NUM_CLASSES {
                print!("{:>7}", self.m[t][p]);
            }
            println!();
        }
    }
}
fn abbrev(name: &str) -> String {
    name.chars().filter(|c| !c.is_whitespace()).take(5).collect()
}
