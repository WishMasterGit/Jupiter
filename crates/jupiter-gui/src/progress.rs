use std::sync::mpsc;
use std::sync::atomic::{AtomicUsize, Ordering};

use jupiter_core::pipeline::{PipelineStage, ProgressReporter};

use crate::messages::WorkerResult;

/// Progress reporter that sends updates over an mpsc channel to the UI thread.
pub struct ChannelProgressReporter {
    tx: mpsc::Sender<WorkerResult>,
    ctx: egui::Context,
    current_total: AtomicUsize,
}

impl ChannelProgressReporter {
    pub fn new(tx: mpsc::Sender<WorkerResult>, ctx: egui::Context) -> Self {
        Self {
            tx,
            ctx,
            current_total: AtomicUsize::new(0),
        }
    }
}

impl ProgressReporter for ChannelProgressReporter {
    fn begin_stage(&self, stage: PipelineStage, total_items: Option<usize>) {
        self.current_total.store(total_items.unwrap_or(0), Ordering::Relaxed);
        let _ = self.tx.send(WorkerResult::Progress {
            stage,
            items_done: Some(0),
            items_total: total_items,
        });
        self.ctx.request_repaint();
    }

    fn advance(&self, items_done: usize) {
        let total = self.current_total.load(Ordering::Relaxed);
        let _ = self.tx.send(WorkerResult::Progress {
            stage: PipelineStage::Stacking, // Generic — the UI knows current stage
            items_done: Some(items_done),
            items_total: if total > 0 { Some(total) } else { None },
        });
        self.ctx.request_repaint();
    }

    fn finish_stage(&self) {
        // UI handles stage transitions via specific Complete messages
    }
}
