mod drizzle;
mod multi_point;
mod standard;
mod surface_warp;

use std::sync::mpsc;

use jupiter_core::pipeline::config::StackMethod;

use crate::messages::WorkerResult;

use super::PipelineCache;

pub(crate) fn handle_stack(
    method: &StackMethod,
    cache: &mut PipelineCache,
    tx: &mpsc::Sender<WorkerResult>,
    ctx: &egui::Context,
) {
    match method {
        StackMethod::MultiPoint(ref mp_config) => {
            multi_point::handle_multi_point(mp_config, cache, tx, ctx);
        }
        StackMethod::SurfaceWarp(ref sw_config) => {
            surface_warp::handle_surface_warp(sw_config, cache, tx, ctx);
        }
        StackMethod::Drizzle(ref drizzle_config) => {
            drizzle::handle_drizzle(drizzle_config, cache, tx, ctx);
        }
        method @ (StackMethod::Mean | StackMethod::Median | StackMethod::SigmaClip(_)) => {
            standard::handle_standard(method, cache, tx, ctx);
        }
    }
}
