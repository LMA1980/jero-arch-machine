//! Causal offline-RL `DecisionTransformer` recipe operating over `(return_to_go, state, action)`
//! trajectory triples with bounded KV caching and self-contained GQA + `SwiGLU` execution.

use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, GgufValue, K3Error, ModelMetadata, ModelWeights,
    QuantType, SafeTensorsFile,
};

use crate::common::{
    bounds_err, checked_mul, find_tensor_by_aliases, infer_layer_count_from_metadata,
    infer_paired_2d_dim, lookup_optional_bias_aliases, lookup_tensor_aliases, mismatch_err,
    parse_st_f32, parse_st_usize, project_mat_view, require_matvec_view_aliases,
    require_matvec_view_owned, require_tensor_1d_aliases, require_tensor_1d_owned,
    validate_finite_slice, validate_quant_and_arch,
};
use crate::kernels_compat::{
    apply_rope_inplace, dequantize_token_row, gqa_causal_attention_f32, is_supported_quant,
    rmsnorm_standard, swiglu_f32,
};

const EMBED_RETURN_ALIASES: &[&str] = &[
    "dt.embed_return.weight",
    "embed_return.weight",
    "encoder.embed_return.weight",
];
const EMBED_STATE_ALIASES: &[&str] = &[
    "dt.embed_state.weight",
    "embed_state.weight",
    "encoder.embed_state.weight",
];
const EMBED_ACTION_ALIASES: &[&str] = &[
    "dt.embed_action.weight",
    "embed_action.weight",
    "encoder.embed_action.weight",
];
const EMBED_TIMESTEP_ALIASES: &[&str] = &[
    "dt.embed_timestep.weight",
    "embed_timestep.weight",
    "encoder.embed_timestep.weight",
];
const OUTPUT_NORM_ALIASES: &[&str] = &[
    "output_norm.weight",
    "encoder.ln_f.weight",
    "ln_f.weight",
    "model.norm.weight",
];
const PREDICT_ACTION_WEIGHT_ALIASES: &[&str] = &[
    "dt.predict_action.weight",
    "predict_action.weight",
    "predict_action.0.weight",
];
const PREDICT_ACTION_BIAS_ALIASES: &[&str] = &[
    "dt.predict_action.bias",
    "predict_action.bias",
    "predict_action.0.bias",
];
const LAYER0_FFN_GATE_ALIASES: &[&str] = &[
    "blk.0.ffn_gate.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "layers.0.ffn_gate.weight",
];
const LAYER_PREFIXES: &[&str] = &["blk.", "model.layers.", "encoder.h.", "layers."];

fn dt_layer_aliases(
    layer: usize,
    gguf_suffix: &str,
    llama_hf_suffix: &str,
    gpt_hf_suffix: &str,
) -> [String; 4] {
    [
        format!("blk.{layer}.{gguf_suffix}"),
        format!("model.layers.{layer}.{llama_hf_suffix}"),
        format!("encoder.h.{layer}.{gpt_hf_suffix}"),
        format!("layers.{layer}.{gguf_suffix}"),
    ]
}

#[derive(Debug, Clone, Copy)]
struct DenseAttnLayerSpec {
    hidden_dim: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    rms_eps: f32,
    rope_theta: f32,
}

fn load_attn_projections(
    spec: DenseAttnLayerSpec,
    layer: usize,
    weights: &dyn ModelWeights,
) -> Result<
    (
        Vec<f32>,
        jero_core::TensorView<'_>,
        jero_core::TensorView<'_>,
        jero_core::TensorView<'_>,
        jero_core::TensorView<'_>,
    ),
    K3Error,
> {
    let h = spec.hidden_dim;
    let q_dim = spec.n_heads_q * spec.head_dim;
    let kv_dim = spec.n_heads_kv * spec.head_dim;
    let attn_norm_w = require_tensor_1d_owned(
        weights,
        &dt_layer_aliases(
            layer,
            "attn_norm.weight",
            "input_layernorm.weight",
            "ln_1.weight",
        ),
        h,
    )?;
    let q_view = require_matvec_view_owned(
        weights,
        &dt_layer_aliases(
            layer,
            "attn_q.weight",
            "self_attn.q_proj.weight",
            "attn.q_proj.weight",
        ),
        h,
        q_dim,
    )?;
    let k_view = require_matvec_view_owned(
        weights,
        &dt_layer_aliases(
            layer,
            "attn_k.weight",
            "self_attn.k_proj.weight",
            "attn.k_proj.weight",
        ),
        h,
        kv_dim,
    )?;
    let v_view = require_matvec_view_owned(
        weights,
        &dt_layer_aliases(
            layer,
            "attn_v.weight",
            "self_attn.v_proj.weight",
            "attn.v_proj.weight",
        ),
        h,
        kv_dim,
    )?;
    let o_view = require_matvec_view_owned(
        weights,
        &dt_layer_aliases(
            layer,
            "attn_output.weight",
            "self_attn.o_proj.weight",
            "attn.out_proj.weight",
        ),
        q_dim,
        h,
    )?;
    Ok((attn_norm_w, q_view, k_view, v_view, o_view))
}

fn run_dense_attention_sublayer(
    spec: DenseAttnLayerSpec,
    layer: usize,
    pos: usize,
    seq_len: usize,
    hidden: &mut [f32],
    layer_kv: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    let h = spec.hidden_dim;
    let (attn_norm_w, q_view, k_view, v_view, o_view) =
        load_attn_projections(spec, layer, weights)?;
    let mut norm_hidden = vec![0.0_f32; h];
    rmsnorm_standard(&mut norm_hidden, hidden, &attn_norm_w, spec.rms_eps)?;

    let q_dim = spec.n_heads_q * spec.head_dim;
    let kv_dim = spec.n_heads_kv * spec.head_dim;
    let mut q = project_mat_view(&q_view, &norm_hidden, q_dim, None)?;
    let mut k = project_mat_view(&k_view, &norm_hidden, kv_dim, None)?;
    let v = project_mat_view(&v_view, &norm_hidden, kv_dim, None)?;

    let eff_seq = seq_len.max(pos + 1);
    apply_rope_inplace(&mut q, spec.head_dim, pos, eff_seq, spec.rope_theta)?;
    apply_rope_inplace(&mut k, spec.head_dim, pos, eff_seq, spec.rope_theta)?;

    #[expect(clippy::cast_precision_loss, reason = "head_dim fits cleanly in f32")]
    let scale = 1.0_f32 / (spec.head_dim as f32).sqrt();
    let capacity_slots = layer_kv.len().checked_div(2 * kv_dim).unwrap_or(0);
    if capacity_slots <= pos {
        return Err(bounds_err(
            "kv_cache",
            (pos + 1) * 2 * kv_dim,
            layer_kv.len(),
            format!(
                "decision-transformer kv_cache layer capacity {capacity_slots} <= position {pos}"
            ),
        ));
    }

    let slot = pos % capacity_slots;
    let active_len = (pos + 1).min(capacity_slots);
    let ring_elems = capacity_slots * kv_dim;
    let (k_ring, v_ring) = layer_kv[..2 * ring_elems].split_at_mut(ring_elems);
    k_ring[slot * kv_dim..(slot + 1) * kv_dim].copy_from_slice(&k);
    v_ring[slot * kv_dim..(slot + 1) * kv_dim].copy_from_slice(&v);

    let mut attn_out = vec![0.0_f32; q_dim];
    gqa_causal_attention_f32(
        &mut attn_out,
        &q,
        &k_ring[..active_len * kv_dim],
        &v_ring[..active_len * kv_dim],
        spec.n_heads_q,
        spec.n_heads_kv,
        spec.head_dim,
        active_len,
        scale,
    )?;

    let proj_attn = project_mat_view(&o_view, &attn_out, h, None)?;
    for (dst, &r) in hidden.iter_mut().zip(proj_attn.iter()) {
        *dst += r;
    }
    Ok(())
}

fn run_dense_swiglu_ffn(
    layer: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    norm_ffn_in: &[f32],
    weights: &dyn ModelWeights,
) -> Result<Vec<f32>, K3Error> {
    let gate_view = require_matvec_view_owned(
        weights,
        &dt_layer_aliases(
            layer,
            "ffn_gate.weight",
            "mlp.gate_proj.weight",
            "mlp.gate_proj.weight",
        ),
        hidden_dim,
        ffn_dim,
    )?;
    let up_view = require_matvec_view_owned(
        weights,
        &dt_layer_aliases(
            layer,
            "ffn_up.weight",
            "mlp.up_proj.weight",
            "mlp.up_proj.weight",
        ),
        hidden_dim,
        ffn_dim,
    )?;
    let down_view = require_matvec_view_owned(
        weights,
        &dt_layer_aliases(
            layer,
            "ffn_down.weight",
            "mlp.down_proj.weight",
            "mlp.down_proj.weight",
        ),
        ffn_dim,
        hidden_dim,
    )?;

    let gate = project_mat_view(&gate_view, norm_ffn_in, ffn_dim, None)?;
    let up = project_mat_view(&up_view, norm_ffn_in, ffn_dim, None)?;
    let mut act = vec![0.0_f32; ffn_dim];
    swiglu_f32(&mut act, &gate, &up)?;
    project_mat_view(&down_view, &act, hidden_dim, None)
}

/// Causal offline-RL `DecisionTransformer` recipe operating over `(return_to_go, state, action)`
/// trajectory triples with bounded KV caching (`n_ctx = 64` compatible).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecisionTransformerRecipe {
    /// Continuous environment state dimension.
    pub state_dim: usize,
    /// Discrete action space size (`act_dim`).
    pub act_dim: usize,
    /// Causal transformer hidden width (`n_embd`).
    pub hidden_dim: usize,
    /// Number of causal decoder layers.
    pub num_layers: usize,
    /// Query attention head count.
    pub n_heads_q: usize,
    /// Key/value attention head count (GQA compatible).
    pub n_heads_kv: usize,
    /// Per-head embedding dimension.
    pub head_dim: usize,
    /// `SwiGLU` feed-forward width.
    pub ffn_dim: usize,
    /// Maximum episode timestep index (`embed_timestep` table capacity).
    pub max_ep_len: usize,
    /// `RMSNorm` epsilon (`1e-6` default).
    pub rms_eps: f32,
    /// `RoPE` frequency base (`10_000.0` default).
    pub rope_theta: f32,
}

impl Default for DecisionTransformerRecipe {
    fn default() -> Self {
        Self::new()
    }
}

impl DecisionTransformerRecipe {
    /// Creates a baseline `decision-transformer` recipe fitting comfortably in `n_ctx = 64`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state_dim: 4,
            act_dim: 4,
            hidden_dim: 16,
            num_layers: 2,
            n_heads_q: 2,
            n_heads_kv: 2,
            head_dim: 8,
            ffn_dim: 32,
            max_ep_len: 64,
            rms_eps: 1e-6,
            rope_theta: 10_000.0,
        }
    }

    /// Hydrates a [`DecisionTransformerRecipe`] from [`ModelMetadata`] and tensor descriptors.
    #[must_use]
    pub fn from_metadata(metadata: &ModelMetadata) -> Self {
        let mut recipe = Self::new();
        let layers = infer_layer_count_from_metadata(metadata, LAYER_PREFIXES);
        if layers > 0 {
            recipe.num_layers = layers;
        }
        if metadata.embedding_dim() > 0 {
            recipe.hidden_dim = metadata.embedding_dim();
        } else if let Some(desc) = find_tensor_by_aliases(metadata, OUTPUT_NORM_ALIASES)
            && desc.num_elements() > 0
        {
            recipe.hidden_dim = desc.num_elements();
        }
        if metadata.head_count() > 0 {
            recipe.n_heads_q = metadata.head_count();
        }
        if metadata.head_count_kv() > 0 {
            recipe.n_heads_kv = metadata.head_count_kv().min(recipe.n_heads_q);
        } else {
            recipe.n_heads_kv = recipe.n_heads_q;
        }
        if let Some(kd) = metadata.key_length()
            && kd >= 2
        {
            recipe.head_dim = kd;
        } else if recipe.n_heads_q > 0 && recipe.hidden_dim >= recipe.n_heads_q {
            recipe.head_dim = (recipe.hidden_dim / recipe.n_heads_q).max(2);
        }
        if let Some(ffn) = metadata.feed_forward_length()
            && ffn > 0
        {
            recipe.ffn_dim = ffn;
        } else if let Some(desc) = find_tensor_by_aliases(metadata, LAYER0_FFN_GATE_ALIASES)
            && let Some(ffn) = infer_paired_2d_dim(desc.shape(), recipe.hidden_dim, false)
        {
            recipe.ffn_dim = ffn;
        } else {
            recipe.ffn_dim = recipe.hidden_dim.saturating_mul(2).max(8);
        }
        if let Some(desc) = find_tensor_by_aliases(metadata, EMBED_STATE_ALIASES)
            && let Some(sd) = infer_paired_2d_dim(desc.shape(), recipe.hidden_dim, true)
        {
            recipe.state_dim = sd;
        }
        if let Some(desc) = find_tensor_by_aliases(metadata, PREDICT_ACTION_BIAS_ALIASES)
            && desc.num_elements() > 0
        {
            recipe.act_dim = desc.num_elements();
        } else if let Some(desc) = find_tensor_by_aliases(metadata, PREDICT_ACTION_WEIGHT_ALIASES)
            && let Some(ad) = infer_paired_2d_dim(desc.shape(), recipe.hidden_dim, false)
        {
            recipe.act_dim = ad;
        }
        if let Some(desc) = find_tensor_by_aliases(metadata, EMBED_TIMESTEP_ALIASES)
            && recipe.hidden_dim > 0
            && desc.num_elements() >= recipe.hidden_dim
        {
            recipe.max_ep_len = (desc.num_elements() / recipe.hidden_dim).max(1);
        }
        if let Some(eps) = metadata.rms_norm_eps()
            && eps.is_finite()
            && eps > 0.0
        {
            recipe.rms_eps = eps;
        }
        if let Some(theta) = metadata.rope_freq_base()
            && theta.is_finite()
            && theta > 0.0
        {
            recipe.rope_theta = theta;
        }
        recipe
    }

    /// Hydrates a [`DecisionTransformerRecipe`] from a parsed [`GgufFile`].
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Self {
        let mut recipe = Self::from_metadata(gguf.metadata());
        for (key, dst) in [
            ("decision-transformer.state_dim", &mut recipe.state_dim),
            ("decision-transformer.act_dim", &mut recipe.act_dim),
            ("decision-transformer.max_ep_len", &mut recipe.max_ep_len),
        ] {
            if let Some(v) = gguf.metadata_value(key).and_then(GgufValue::as_usize)
                && v > 0
            {
                *dst = v;
            }
        }
        recipe
    }

    /// Hydrates a [`DecisionTransformerRecipe`] from a parsed [`SafeTensorsFile`], reading
    /// `"__metadata__"` keys and tensor descriptors uniformly.
    #[must_use]
    pub fn from_safetensors(st: &SafeTensorsFile<'_>) -> Self {
        let mut recipe = Self::from_metadata(st.metadata());
        if let Some(v) = parse_st_usize(
            st,
            &[
                "decision-transformer.block_count",
                "n_layer",
                "num_hidden_layers",
            ],
            1,
        ) {
            recipe.num_layers = v;
        }
        if let Some(v) = parse_st_usize(
            st,
            &[
                "decision-transformer.embedding_length",
                "hidden_size",
                "n_embd",
            ],
            4,
        ) {
            recipe.hidden_dim = v;
        }
        if let Some(v) = parse_st_usize(
            st,
            &[
                "decision-transformer.attention.head_count",
                "n_head",
                "num_attention_heads",
            ],
            1,
        ) {
            recipe.n_heads_q = v;
        }
        if let Some(v) = parse_st_usize(
            st,
            &[
                "decision-transformer.attention.head_count_kv",
                "num_key_value_heads",
            ],
            1,
        ) {
            recipe.n_heads_kv = v.min(recipe.n_heads_q);
        } else {
            recipe.n_heads_kv = recipe.n_heads_q;
        }
        if let Some(v) = parse_st_usize(
            st,
            &["decision-transformer.attention.key_length", "head_dim"],
            2,
        ) {
            recipe.head_dim = v;
        } else if recipe.n_heads_q > 0 && recipe.hidden_dim >= recipe.n_heads_q {
            recipe.head_dim = (recipe.hidden_dim / recipe.n_heads_q).max(2);
        }
        if let Some(v) = parse_st_usize(
            st,
            &[
                "decision-transformer.feed_forward_length",
                "n_inner",
                "intermediate_size",
            ],
            4,
        ) {
            recipe.ffn_dim = v;
        }
        if let Some(v) = parse_st_usize(st, &["decision-transformer.state_dim", "state_dim"], 1) {
            recipe.state_dim = v;
        }
        if let Some(v) = parse_st_usize(st, &["decision-transformer.act_dim", "act_dim"], 1) {
            recipe.act_dim = v;
        }
        if let Some(v) = parse_st_usize(st, &["decision-transformer.max_ep_len", "max_ep_len"], 1) {
            recipe.max_ep_len = v;
        }
        if let Some(eps) = parse_st_f32(
            st,
            &[
                "decision-transformer.attention.layer_norm_rms_epsilon",
                "rms_norm_eps",
                "layer_norm_epsilon",
            ],
        ) {
            recipe.rms_eps = eps;
        }
        if let Some(theta) =
            parse_st_f32(st, &["decision-transformer.rope.freq_base", "rope_theta"])
        {
            recipe.rope_theta = theta;
        }
        recipe
    }

    fn validate_hyperparams(&self) -> Result<(), K3Error> {
        if self.state_dim == 0
            || self.act_dim == 0
            || self.hidden_dim < 4
            || self.num_layers == 0
            || self.n_heads_q == 0
            || self.n_heads_kv == 0
            || self.n_heads_q % self.n_heads_kv != 0
            || self.head_dim < 2
            || self.head_dim % 2 != 0
            || self.ffn_dim < 4
            || self.max_ep_len == 0
            || !self.rms_eps.is_finite()
            || self.rms_eps <= 0.0
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
        {
            return Err(mismatch_err(
                "decision-transformer",
                "invalid DecisionTransformer hyperparameters (requires hidden_dim >= 4, ffn_dim >= 4, even head_dim >= 2, and n_heads_q divisible by n_heads_kv)",
            ));
        }
        Ok(())
    }

    fn validate_weights(&self, weights: &dyn ModelWeights) -> Result<(), K3Error> {
        let h = self.hidden_dim;
        let _ = checked_mul("dt_q_dim", self.n_heads_q, self.head_dim)?;
        let _ = checked_mul("dt_kv_dim", self.n_heads_kv, self.head_dim)?;
        let _ = require_matvec_view_aliases(weights, EMBED_RETURN_ALIASES, 1, h)?;
        let _ = require_matvec_view_aliases(weights, EMBED_STATE_ALIASES, self.state_dim, h)?;
        let _ = require_matvec_view_aliases(weights, EMBED_ACTION_ALIASES, h, self.act_dim)?;
        let _ = require_matvec_view_aliases(weights, EMBED_TIMESTEP_ALIASES, h, self.max_ep_len)?;
        let _ = require_tensor_1d_aliases(weights, OUTPUT_NORM_ALIASES, h)?;
        let _ =
            require_matvec_view_aliases(weights, PREDICT_ACTION_WEIGHT_ALIASES, h, self.act_dim)?;
        let _ = lookup_optional_bias_aliases(weights, PREDICT_ACTION_BIAS_ALIASES, self.act_dim)?;
        for layer in 0..self.num_layers {
            self.validate_layer_weights(weights, layer, h)?;
        }
        Ok(())
    }

    fn validate_layer_weights(
        &self,
        weights: &dyn ModelWeights,
        layer: usize,
        h: usize,
    ) -> Result<(), K3Error> {
        let _ = load_attn_projections(self.attn_spec(), layer, weights)?;
        let _ = require_tensor_1d_owned(
            weights,
            &dt_layer_aliases(
                layer,
                "ffn_norm.weight",
                "post_attention_layernorm.weight",
                "ln_2.weight",
            ),
            h,
        )?;
        let _ = require_matvec_view_owned(
            weights,
            &dt_layer_aliases(
                layer,
                "ffn_gate.weight",
                "mlp.gate_proj.weight",
                "mlp.gate_proj.weight",
            ),
            h,
            self.ffn_dim,
        )?;
        let _ = require_matvec_view_owned(
            weights,
            &dt_layer_aliases(
                layer,
                "ffn_up.weight",
                "mlp.up_proj.weight",
                "mlp.up_proj.weight",
            ),
            h,
            self.ffn_dim,
        )?;
        let _ = require_matvec_view_owned(
            weights,
            &dt_layer_aliases(
                layer,
                "ffn_down.weight",
                "mlp.down_proj.weight",
                "mlp.down_proj.weight",
            ),
            self.ffn_dim,
            h,
        )?;
        Ok(())
    }

    fn validate_geometry(&self, weights: &dyn ModelWeights) -> Result<(), K3Error> {
        self.validate_hyperparams()?;
        self.validate_weights(weights)
    }

    const fn attn_spec(&self) -> DenseAttnLayerSpec {
        DenseAttnLayerSpec {
            hidden_dim: self.hidden_dim,
            n_heads_q: self.n_heads_q,
            n_heads_kv: self.n_heads_kv,
            head_dim: self.head_dim,
            rms_eps: self.rms_eps,
            rope_theta: self.rope_theta,
        }
    }

    fn run_token_hidden(
        &self,
        mut hidden: Vec<f32>,
        pos: usize,
        seq_len: usize,
        kv_cache: &mut [f32],
        logits: &mut [f32],
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let spec = self.attn_spec();
        let layer_stride = kv_cache.len() / self.num_layers;
        for layer in 0..self.num_layers {
            let layer_kv = &mut kv_cache[layer * layer_stride..(layer + 1) * layer_stride];
            run_dense_attention_sublayer(
                spec,
                layer,
                pos,
                seq_len,
                &mut hidden,
                layer_kv,
                weights,
            )?;
            let ffn_norm_w = require_tensor_1d_owned(
                weights,
                &dt_layer_aliases(
                    layer,
                    "ffn_norm.weight",
                    "post_attention_layernorm.weight",
                    "ln_2.weight",
                ),
                self.hidden_dim,
            )?;
            let mut norm_ffn = vec![0.0_f32; self.hidden_dim];
            rmsnorm_standard(&mut norm_ffn, &hidden, &ffn_norm_w, self.rms_eps)?;
            let ffn_out =
                run_dense_swiglu_ffn(layer, self.hidden_dim, self.ffn_dim, &norm_ffn, weights)?;
            for (h_val, &r) in hidden.iter_mut().zip(ffn_out.iter()) {
                *h_val += r;
            }
        }
        let out_norm_w = require_tensor_1d_aliases(weights, OUTPUT_NORM_ALIASES, self.hidden_dim)?;
        let mut normed = vec![0.0_f32; self.hidden_dim];
        rmsnorm_standard(&mut normed, &hidden, &out_norm_w, self.rms_eps)?;
        let head_view = require_matvec_view_aliases(
            weights,
            PREDICT_ACTION_WEIGHT_ALIASES,
            self.hidden_dim,
            self.act_dim,
        )?;
        let head_bias =
            lookup_optional_bias_aliases(weights, PREDICT_ACTION_BIAS_ALIASES, self.act_dim)?;
        let action_logits =
            project_mat_view(&head_view, &normed, self.act_dim, head_bias.as_deref())?;
        logits[..self.act_dim].copy_from_slice(&action_logits);
        Ok(())
    }

    fn timestep_embedding(
        &self,
        step_idx: usize,
        weights: &dyn ModelWeights,
    ) -> Result<Vec<f32>, K3Error> {
        let clamped = step_idx.min(self.max_ep_len - 1);
        let ts_view = lookup_tensor_aliases(weights, EMBED_TIMESTEP_ALIASES)?;
        let mut ts = vec![0.0_f32; self.hidden_dim];
        dequantize_token_row(&ts_view, clamped, &mut ts)?;
        validate_finite_slice("dt.embed_timestep.weight", &ts)?;
        Ok(ts)
    }

    fn action_embedding(
        &self,
        action_id: u32,
        weights: &dyn ModelWeights,
    ) -> Result<Vec<f32>, K3Error> {
        let idx = usize::try_from(action_id).unwrap_or(usize::MAX);
        if idx >= self.act_dim {
            return Err(bounds_err(
                "action_id",
                self.act_dim,
                idx,
                format!("action_id {idx} >= act_dim {}", self.act_dim),
            ));
        }
        let act_view = lookup_tensor_aliases(weights, EMBED_ACTION_ALIASES)?;
        let mut emb = vec![0.0_f32; self.hidden_dim];
        dequantize_token_row(&act_view, idx, &mut emb)?;
        validate_finite_slice("dt.embed_action.weight", &emb)?;
        Ok(emb)
    }

    fn build_prefill_sequence(
        &self,
        ctx: &ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<Vec<Vec<f32>>, K3Error> {
        let mut seq = Vec::new();
        if let Some(sp) = ctx.soft_prompt_embeds() {
            let (n_sp, sp_dim) = ctx.soft_prompt_dims().ok_or_else(|| {
                mismatch_err(
                    "decision-transformer",
                    "soft_prompt_embeds missing soft_prompt_dims",
                )
            })?;
            if sp_dim != self.hidden_dim {
                return Err(mismatch_err(
                    "decision-transformer",
                    format!(
                        "soft prompt embed_dim {sp_dim} != hidden_dim {}",
                        self.hidden_dim
                    ),
                ));
            }
            for s in 0..n_sp {
                seq.push(sp[s * sp_dim..(s + 1) * sp_dim].to_vec());
            }
        }

        let actions = ctx.prompt_tokens();
        let ret_view = lookup_tensor_aliases(weights, EMBED_RETURN_ALIASES)?;
        let state_view = lookup_tensor_aliases(weights, EMBED_STATE_ALIASES)?;

        if let Some(features) = ctx.feature_input() {
            let (n_steps, n_cols) = ctx.feature_dims().ok_or_else(|| {
                bounds_err("feature_input", 1, 0, "missing feature_dims".to_owned())
            })?;
            let expected_cols = 1 + self.state_dim;
            if n_cols != expected_cols {
                return Err(bounds_err(
                    "feature_input",
                    expected_cols,
                    n_cols,
                    format!(
                        "decision-transformer expected [n_steps, 1 + state_dim] ({expected_cols} cols), got {n_cols}"
                    ),
                ));
            }
            if actions.len() > n_steps {
                return Err(bounds_err(
                    "prompt_tokens",
                    n_steps,
                    actions.len(),
                    format!(
                        "action count {} exceeds trajectory steps {n_steps}",
                        actions.len()
                    ),
                ));
            }
            for t in 0..n_steps {
                let row = &features[t * n_cols..(t + 1) * n_cols];
                let ts = self.timestep_embedding(t, weights)?;
                let mut r_emb = project_mat_view(&ret_view, &row[..1], self.hidden_dim, None)?;
                let mut s_emb = project_mat_view(&state_view, &row[1..], self.hidden_dim, None)?;
                for (dst, &tv) in r_emb.iter_mut().zip(ts.iter()) {
                    *dst += tv;
                }
                for (dst, &tv) in s_emb.iter_mut().zip(ts.iter()) {
                    *dst += tv;
                }
                seq.push(r_emb);
                seq.push(s_emb);
                if let Some(&act_id) = actions.get(t) {
                    let mut a_emb = self.action_embedding(act_id, weights)?;
                    for (dst, &tv) in a_emb.iter_mut().zip(ts.iter()) {
                        *dst += tv;
                    }
                    seq.push(a_emb);
                }
            }
        } else {
            if actions.is_empty() && seq.is_empty() {
                return Err(mismatch_err(
                    "decision-transformer",
                    "prefill requires feature_input ([return_to_go, state]) or prompt action tokens",
                ));
            }
            for (t, &act_id) in actions.iter().enumerate() {
                let ts = self.timestep_embedding(t, weights)?;
                let mut a_emb = self.action_embedding(act_id, weights)?;
                for (dst, &tv) in a_emb.iter_mut().zip(ts.iter()) {
                    *dst += tv;
                }
                seq.push(a_emb);
            }
        }
        Ok(seq)
    }
}

impl ArchRecipe for DecisionTransformerRecipe {
    fn architecture_id(&self) -> &'static str {
        "decision-transformer"
    }

    fn requires_kv_cache(&self) -> bool {
        true
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn validate_model(
        &self,
        metadata: &ModelMetadata,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.validate_hyperparams()?;
        validate_quant_and_arch(self, metadata, weights)?;
        if weights.tensor_count() > 0 {
            self.validate_weights(weights)?;
        }
        Ok(())
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.validate_geometry(weights)?;
        if ctx.logits().len() < self.act_dim {
            return Err(bounds_err(
                "logits",
                self.act_dim,
                ctx.logits().len(),
                format!(
                    "decision-transformer logits length {} < act_dim {}",
                    ctx.logits().len(),
                    self.act_dim
                ),
            ));
        }

        let seq = self.build_prefill_sequence(ctx, weights)?;
        let start_pos = ctx.position();
        let seq_len = seq.len();
        if start_pos + seq_len > ctx.n_ctx() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: start_pos + seq_len,
                max_context_tokens: ctx.n_ctx(),
            });
        }

        let mut step_logits = vec![0.0_f32; self.act_dim];
        for (offset, hidden) in seq.into_iter().enumerate() {
            self.run_token_hidden(
                hidden,
                start_pos + offset,
                start_pos + seq_len,
                ctx.kv_cache_mut(),
                &mut step_logits,
                weights,
            )?;
            if offset + 1 == seq_len {
                ctx.logits_mut()[..self.act_dim].copy_from_slice(&step_logits);
            }
        }
        ctx.advance_position(seq_len)?;
        Ok(())
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.validate_geometry(weights)?;
        let act_id = ctx.decode_token().ok_or_else(|| {
            mismatch_err(
                "decision-transformer",
                "decode_step called without staged decode_token",
            )
        })?;
        let pos = ctx.position();
        if pos >= ctx.n_ctx() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: pos + 1,
                max_context_tokens: ctx.n_ctx(),
            });
        }
        if ctx.logits().len() < self.act_dim {
            return Err(bounds_err(
                "logits",
                self.act_dim,
                ctx.logits().len(),
                format!(
                    "logits length {} < act_dim {}",
                    ctx.logits().len(),
                    self.act_dim
                ),
            ));
        }

        let ts = self.timestep_embedding(pos / 3, weights)?;
        let mut hidden = self.action_embedding(act_id, weights)?;
        for (dst, &tv) in hidden.iter_mut().zip(ts.iter()) {
            *dst += tv;
        }
        let mut out_logits = vec![0.0_f32; self.act_dim];
        self.run_token_hidden(
            hidden,
            pos,
            pos + 1,
            ctx.kv_cache_mut(),
            &mut out_logits,
            weights,
        )?;
        ctx.logits_mut()[..self.act_dim].copy_from_slice(&out_logits);
        ctx.advance_position(1)?;
        Ok(())
    }
}
