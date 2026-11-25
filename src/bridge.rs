//! Multimodal soft-prompt projector (`MmprojLatentBridge`) linking non-tokenized machine-model
//! latent representations to target LLM Layer 0 continuous soft-prompt embeddings.

use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelMetadata, ModelWeights};

use crate::common::{
    bounds_err, checked_mul, contains_any_tensor, lookup_optional_bias_aliases,
    lookup_tensor_aliases, mismatch_err, project_mat_view, require_matvec_view_aliases,
    require_tensor_1d_aliases, validate_finite_slice, validate_raw_unquantized_finite,
};
use crate::kernels_compat::{gelu_inplace_f32, layernorm_f32};

const FC1_WEIGHT_ALIASES: &[&str] = &[
    "mmproj.fc1.weight",
    "mmproj.0.weight",
    "multi_modal_projector.linear_1.weight",
    "mm_projector.0.weight",
];
const FC1_BIAS_ALIASES: &[&str] = &[
    "mmproj.fc1.bias",
    "mmproj.0.bias",
    "multi_modal_projector.linear_1.bias",
    "mm_projector.0.bias",
];
const FC2_WEIGHT_ALIASES: &[&str] = &[
    "mmproj.fc2.weight",
    "mmproj.2.weight",
    "multi_modal_projector.linear_2.weight",
    "mm_projector.2.weight",
];
const FC2_BIAS_ALIASES: &[&str] = &[
    "mmproj.fc2.bias",
    "mmproj.2.bias",
    "multi_modal_projector.linear_2.bias",
    "mm_projector.2.bias",
];
const LINEAR_WEIGHT_ALIASES: &[&str] = &[
    "mmproj.weight",
    "multi_modal_projector.weight",
    "mm_projector.weight",
];
const LINEAR_BIAS_ALIASES: &[&str] = &[
    "mmproj.bias",
    "multi_modal_projector.bias",
    "mm_projector.bias",
];
const NORM_WEIGHT_ALIASES: &[&str] = &[
    "mmproj.norm.weight",
    "multi_modal_projector.norm.weight",
    "mm_projector.norm.weight",
];
const NORM_BIAS_ALIASES: &[&str] = &[
    "mmproj.norm.bias",
    "multi_modal_projector.norm.bias",
    "mm_projector.norm.bias",
];

/// Target-compatible multimodal soft-prompt projector (`MmprojLatentBridge`) connecting a
/// machine-model latent vector to target LLM prefill embeddings in Layer 0.
#[derive(Debug, Clone, PartialEq)]
pub struct MmprojLatentBridge {
    /// Expected target LLM GGUF `general.architecture` identifier (e.g., `"qwen2"`, `"llama"`).
    pub target_architecture_id: String,
    /// Expected target LLM recipe revision string (e.g., `"1.0.0"`).
    pub target_recipe_version: String,
    /// Machine-model latent feature dimension expected on input.
    pub machine_latent_dim: usize,
    /// Target LLM embedding length (`n_embd`).
    pub target_embedding_length: usize,
    /// Number of Layer-0 soft-prompt tokens emitted per machine latent vector.
    pub num_soft_tokens: usize,
    /// Optional `LayerNorm` epsilon applied to projected soft-prompt vectors.
    pub eps: f32,
}

impl MmprojLatentBridge {
    /// Creates a new [`MmprojLatentBridge`] targeting `(target_architecture_id, target_recipe_version, target_embedding_length)`.
    #[must_use]
    pub fn new(
        target_architecture_id: impl Into<String>,
        target_recipe_version: impl Into<String>,
        machine_latent_dim: usize,
        target_embedding_length: usize,
        num_soft_tokens: usize,
    ) -> Self {
        Self {
            target_architecture_id: target_architecture_id.into(),
            target_recipe_version: target_recipe_version.into(),
            machine_latent_dim,
            target_embedding_length,
            num_soft_tokens,
            eps: 1e-5,
        }
    }

    /// Validates static bridge geometry (`machine_latent_dim > 0`, `target_embedding_length > 0`,
    /// `num_soft_tokens > 0`, finite positive `eps`).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if any bridge dimension or parameter is invalid.
    pub fn validate_geometry(&self) -> Result<(), K3Error> {
        if self.machine_latent_dim == 0
            || self.target_embedding_length == 0
            || self.num_soft_tokens == 0
        {
            return Err(mismatch_err(
                &self.target_architecture_id,
                format!(
                    "mmproj dimensions must all be non-zero (latent={}, embed={}, tokens={})",
                    self.machine_latent_dim, self.target_embedding_length, self.num_soft_tokens
                ),
            ));
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err(mismatch_err(
                &self.target_architecture_id,
                format!("mmproj eps ({}) must be finite and > 0", self.eps),
            ));
        }
        let _ = checked_mul(
            "mmproj_output",
            self.num_soft_tokens,
            self.target_embedding_length,
        )?;
        Ok(())
    }

    /// Verifies that the target LLM `(architecture_id, recipe_version, embedding_length)` tuple
    /// matches this projector bridge before execution.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] on any mismatch across `architecture_id`,
    /// `recipe_version`, or `embedding_length`.
    pub fn verify_target_compatibility(
        &self,
        target_arch: &str,
        target_version: &str,
        target_embedding_len: usize,
    ) -> Result<(), K3Error> {
        self.validate_geometry()?;
        if self.target_architecture_id != target_arch {
            return Err(mismatch_err(
                target_arch,
                format!(
                    "mmproj target architecture mismatch: bridge expects '{}', target LLM is '{}'",
                    self.target_architecture_id, target_arch
                ),
            ));
        }
        if self.target_recipe_version != target_version {
            return Err(mismatch_err(
                target_arch,
                format!(
                    "mmproj target recipe version mismatch for '{target_arch}': bridge expects '{}', target LLM is '{}'",
                    self.target_recipe_version, target_version
                ),
            ));
        }
        if self.target_embedding_length != target_embedding_len {
            return Err(mismatch_err(
                target_arch,
                format!(
                    "mmproj target embedding_length mismatch for '{target_arch}': bridge expects {}, target LLM has {target_embedding_len}",
                    self.target_embedding_length
                ),
            ));
        }
        Ok(())
    }

    /// Verifies compatibility against a concrete [`ArchRecipe`] and its [`ModelMetadata`].
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if any tuple field diverges.
    pub fn verify_target_recipe(
        &self,
        target_recipe: &dyn ArchRecipe,
        metadata: &ModelMetadata,
    ) -> Result<(), K3Error> {
        self.verify_target_compatibility(
            target_recipe.architecture_id(),
            target_recipe.recipe_version(),
            metadata.embedding_dim(),
        )
    }
}

/// Projects a single machine-model latent vector (`[bridge.machine_latent_dim]`) into `K` Layer-0
/// LLM soft-prompt tokens (`[bridge.num_soft_tokens * bridge.target_embedding_length]`) using
/// resident `mmproj.*` GGUF or `SafeTensors` projection weights (`mmproj.fc1.weight` +
/// `mmproj.fc2.weight`, or single linear `mmproj.weight`).
///
/// Fails closed with [`K3Error::TensorBoundsError`] when required projector weights are absent,
/// mismatched in geometry, or produce non-finite outputs.
///
/// # Errors
/// Returns [`K3Error::CapabilityMismatch`] if `bridge` parameters are invalid, or
/// [`K3Error::TensorBoundsError`] on slice/weight shape mismatch or non-finite values.
pub fn project_machine_latent_to_soft_prompt(
    bridge: &MmprojLatentBridge,
    latent: &[f32],
    weights: &dyn ModelWeights,
    out_embeds: &mut [f32],
) -> Result<(), K3Error> {
    bridge.validate_geometry()?;
    if latent.len() != bridge.machine_latent_dim {
        return Err(bounds_err(
            "machine_latent",
            bridge.machine_latent_dim,
            latent.len(),
            format!(
                "expected machine latent of length {}, got {}",
                bridge.machine_latent_dim,
                latent.len()
            ),
        ));
    }
    validate_finite_slice("machine_latent", latent)?;

    let total_out = checked_mul(
        "mmproj_output",
        bridge.num_soft_tokens,
        bridge.target_embedding_length,
    )?;
    if out_embeds.len() != total_out {
        return Err(bounds_err(
            "soft_prompt_embeds",
            total_out,
            out_embeds.len(),
            format!(
                "expected out_embeds length {total_out} ({}x{}), got {}",
                bridge.num_soft_tokens,
                bridge.target_embedding_length,
                out_embeds.len()
            ),
        ));
    }

    let projected = project_mmproj_weights(bridge, latent, weights, total_out)?;
    out_embeds.copy_from_slice(&projected);
    apply_mmproj_norm(bridge, weights, out_embeds)?;
    validate_finite_slice("soft_prompt_embeds", out_embeds)?;
    Ok(())
}

fn resolve_fc1_hidden_dim(shape: &[usize], latent_dim: usize) -> Option<usize> {
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return None;
    }
    if shape[0] == latent_dim {
        Some(shape[1])
    } else if shape[1] == latent_dim {
        Some(shape[0])
    } else {
        None
    }
}

fn project_mmproj_weights(
    bridge: &MmprojLatentBridge,
    latent: &[f32],
    weights: &dyn ModelWeights,
    total_out: usize,
) -> Result<Vec<f32>, K3Error> {
    if contains_any_tensor(weights, FC1_WEIGHT_ALIASES) {
        let fc1_view = lookup_tensor_aliases(weights, FC1_WEIGHT_ALIASES)?;
        let fc1_name = fc1_view.name();
        let hidden_dim = resolve_fc1_hidden_dim(fc1_view.shape(), bridge.machine_latent_dim)
            .ok_or_else(|| {
                bounds_err(
                    fc1_name,
                    bridge.machine_latent_dim,
                    fc1_view.num_elements(),
                    format!(
                        "{fc1_name} expected 2D shape with latent dim {}, got {:?}",
                        bridge.machine_latent_dim,
                        fc1_view.shape()
                    ),
                )
            })?;
        validate_raw_unquantized_finite(&fc1_view, fc1_name)?;
        let fc1_bias = lookup_optional_bias_aliases(weights, FC1_BIAS_ALIASES, hidden_dim)?;
        let mut mid = project_mat_view(&fc1_view, latent, hidden_dim, fc1_bias.as_deref())?;
        gelu_inplace_f32(&mut mid)?;
        let fc2_view =
            require_matvec_view_aliases(weights, FC2_WEIGHT_ALIASES, hidden_dim, total_out)?;
        let fc2_bias = lookup_optional_bias_aliases(weights, FC2_BIAS_ALIASES, total_out)?;
        project_mat_view(&fc2_view, &mid, total_out, fc2_bias.as_deref())
    } else if contains_any_tensor(weights, LINEAR_WEIGHT_ALIASES) {
        let w_view = require_matvec_view_aliases(
            weights,
            LINEAR_WEIGHT_ALIASES,
            bridge.machine_latent_dim,
            total_out,
        )?;
        let bias = lookup_optional_bias_aliases(weights, LINEAR_BIAS_ALIASES, total_out)?;
        project_mat_view(&w_view, latent, total_out, bias.as_deref())
    } else {
        match weights.tensor("mmproj.fc1.weight") {
            Err(err) => Err(err),
            Ok(_) => Err(bounds_err(
                "mmproj.fc1.weight",
                1,
                0,
                "missing required mmproj projector weights".to_owned(),
            )),
        }
    }
}

fn apply_mmproj_norm(
    bridge: &MmprojLatentBridge,
    weights: &dyn ModelWeights,
    out_embeds: &mut [f32],
) -> Result<(), K3Error> {
    if !contains_any_tensor(weights, NORM_WEIGHT_ALIASES) {
        return Ok(());
    }
    let norm_w =
        require_tensor_1d_aliases(weights, NORM_WEIGHT_ALIASES, bridge.target_embedding_length)?;
    let norm_b =
        lookup_optional_bias_aliases(weights, NORM_BIAS_ALIASES, bridge.target_embedding_length)?;
    let mut normed_tok = vec![0.0_f32; bridge.target_embedding_length];
    for chunk in out_embeds.chunks_exact_mut(bridge.target_embedding_length) {
        layernorm_f32(
            &mut normed_tok,
            chunk,
            Some(&norm_w),
            norm_b.as_deref(),
            bridge.eps,
        )?;
        chunk.copy_from_slice(&normed_tok);
    }
    Ok(())
}

/// Executes an LLM prefill forward pass (`recipe.prefill_step`) with Layer-0 continuous
/// `soft_prompt` embeddings prepended ahead of discrete `prompt_tokens`, returning the advanced
/// KV sequence position (`ctx.position()`).
///
/// # Errors
/// Returns [`K3Error`] if context bounds are exceeded, soft-prompt geometry fails validation, or
/// recipe execution fails.
#[expect(
    clippy::too_many_arguments,
    reason = "bridges raw caller prompt/soft-prompt slices into ExecutionContext without mutating jero-core"
)]
pub fn forward_with_soft_prompt(
    recipe: &dyn ArchRecipe,
    prompt_tokens: &[u32],
    n_ctx: usize,
    soft_prompt: &[f32],
    num_soft_tokens: usize,
    embed_dim: usize,
    logits: &mut [f32],
    kv_cache: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<usize, K3Error> {
    let total_tokens =
        num_soft_tokens
            .checked_add(prompt_tokens.len())
            .ok_or(K3Error::ContextLengthExceeded {
                requested_tokens: usize::MAX,
                max_context_tokens: n_ctx,
            })?;
    if total_tokens > n_ctx {
        return Err(K3Error::ContextLengthExceeded {
            requested_tokens: total_tokens,
            max_context_tokens: n_ctx,
        });
    }
    let mut ctx = ExecutionContext::new(prompt_tokens, n_ctx, logits, kv_cache)?
        .with_soft_prompt_embeds(soft_prompt, num_soft_tokens, embed_dim)?;
    recipe.prefill_step(&mut ctx, weights)?;
    Ok(ctx.position())
}
