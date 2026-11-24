//! `jero-arch-machine`: Non-tokenized machine-model architecture recipes (`TTM`,
//! `FT-Transformer`, `TabPFN-v2`), sequential offline-RL policy execution
//! (`DecisionTransformer`), and the multimodal soft-prompt bridge (`MmprojLatentBridge`) linking
//! machine latents to target LLM prefill Layer 0.
//!
//! All execution recipes support both [`jero_core::GgufFile`] and [`jero_core::SafeTensorsFile`]
//! (`ModelWeights`) with strict fail-closed validation on tensor existence, exact geometry,
//! supported quantization formats, bounded context budgets, and upfront element finiteness
//! (`NaN` / `±Inf` rejection) per ADR-045.

#![deny(unsafe_op_in_unsafe_fn)]

mod bridge;
mod common;
mod decision_transformer;
mod ft_transformer;
mod kernels_compat;
mod tabpfn;
mod ttm;

#[cfg(test)]
mod tests;

pub use bridge::{
    MmprojLatentBridge, forward_with_soft_prompt, project_machine_latent_to_soft_prompt,
};
pub use decision_transformer::DecisionTransformerRecipe;
pub use ft_transformer::FtTransformerRecipe;
pub use tabpfn::TabPfnV2Recipe;
pub use ttm::TtmRecipe;

/// Canonical crate identifier for this architecture recipe package.
pub const CRATE_NAME: &str = env!("CARGO_PKG_NAME");

/// Canonical SPDX license identifier for this architecture recipe package.
pub const CRATE_LICENSE: &str = "LGPL-3.0-only WITH LGPL-3.0-linking-exception";

/// Returns the crate description configured in `Cargo.toml`.
#[must_use]
pub const fn crate_description() -> &'static str {
    env!("CARGO_PKG_DESCRIPTION")
}
