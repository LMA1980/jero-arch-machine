//! Zero-KV `TabPFN-v2` dual-axis (`Feature` + context-restricted `Sample`) tabular classifier.

use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, GgufValue, K3Error, ModelMetadata, ModelWeights,
    QuantType, SafeTensorsFile, TensorView,
};

use crate::common::{
    bounds_err, checked_mul, find_tensor_by_aliases, infer_layer_count_from_metadata,
    infer_paired_2d_dim, mismatch_err, parse_st_f32, parse_st_usize, project_mat_view,
    require_matvec_view_aliases, require_matvec_view_owned, require_tensor_1d_aliases,
    require_tensor_1d_owned, validate_finite_slice, validate_quant_and_arch,
};
use crate::kernels_compat::{
    TabPfnAttentionAxis, TabPfnAttentionShape, dequantize_token_row, gelu_inplace_f32,
    is_supported_quant, layernorm_f32, tabpfn_dual_axis_attention_f32,
    tokenize_continuous_features_f32,
};

type QkvProjection = (Vec<f32>, Vec<f32>, Vec<f32>);

const FEATURE_ENC_WEIGHT_ALIASES: &[&str] = &[
    "tabpfn.feature_enc.weight",
    "encoder.weight",
    "feature_encoder.weight",
    "x_encoder.weight",
];
const FEATURE_ENC_BIAS_ALIASES: &[&str] = &[
    "tabpfn.feature_enc.bias",
    "encoder.bias",
    "feature_encoder.bias",
    "x_encoder.bias",
];
const Y_EMBD_ALIASES: &[&str] = &["tabpfn.y_embd.weight", "y_encoder.weight", "y_embd.weight"];
const Y_UNKNOWN_ALIASES: &[&str] = &[
    "tabpfn.y_unknown.weight",
    "y_unknown.weight",
    "y_encoder.unknown_weight",
];
const OUTPUT_NORM_WEIGHT_ALIASES: &[&str] =
    &["output_norm.weight", "decoder_norm.weight", "norm.weight"];
const OUTPUT_NORM_BIAS_ALIASES: &[&str] = &["output_norm.bias", "decoder_norm.bias", "norm.bias"];
const CLASSIFIER_WEIGHT_ALIASES: &[&str] = &[
    "classifier.weight",
    "decoder_dict.standard.2.weight",
    "decoder.weight",
    "head.weight",
];
const CLASSIFIER_BIAS_ALIASES: &[&str] = &[
    "classifier.bias",
    "decoder_dict.standard.2.bias",
    "decoder.bias",
    "head.bias",
];
const LAYER0_FFN_FC1_BIAS_ALIASES: &[&str] = &[
    "blk.0.ffn_fc1.bias",
    "transformer_encoder.layers.0.mlp.linear1.bias",
    "layers.0.ffn_fc1.bias",
];
const LAYER_PREFIXES: &[&str] = &["blk.", "transformer_encoder.layers.", "layers."];

fn tabpfn_layer_aliases(layer: usize, gguf_suffix: &str, hf_suffix: &str) -> [String; 3] {
    [
        format!("blk.{layer}.{gguf_suffix}"),
        format!("transformer_encoder.layers.{layer}.{hf_suffix}"),
        format!("layers.{layer}.{gguf_suffix}"),
    ]
}

/// Zero-KV `TabPFN-v2` dual-axis (`Feature` + context-restricted `Sample`) tabular classifier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TabPfnV2Recipe {
    /// Number of tabular feature columns (`F`).
    pub num_features: usize,
    /// Per-cell hidden representation width (`d_model = n_heads * head_dim`).
    pub hidden_dim: usize,
    /// Number of dual-axis attention + FFN layers.
    pub num_layers: usize,
    /// Number of attention heads dividing `hidden_dim`.
    pub n_heads: usize,
    /// Intermediate FFN width.
    pub ffn_dim: usize,
    /// Number of discrete target classes.
    pub num_classes: usize,
    /// `LayerNorm` epsilon (`1e-5` default).
    pub norm_eps: f32,
}

impl Default for TabPfnV2Recipe {
    fn default() -> Self {
        Self::new()
    }
}

impl TabPfnV2Recipe {
    /// Creates a baseline `tabpfn-v2` prior-data in-context classification recipe.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            num_features: 4,
            hidden_dim: 16,
            num_layers: 2,
            n_heads: 2,
            ffn_dim: 32,
            num_classes: 2,
            norm_eps: 1e-5,
        }
    }

    /// Hydrates a [`TabPfnV2Recipe`] from [`ModelMetadata`] and GGUF or `SafeTensors` descriptors.
    #[must_use]
    pub fn from_metadata(metadata: &ModelMetadata) -> Self {
        let mut recipe = Self::new();
        if metadata.embedding_dim() > 0 {
            recipe.hidden_dim = metadata.embedding_dim();
        } else if let Some(desc) = find_tensor_by_aliases(metadata, Y_UNKNOWN_ALIASES)
            .or_else(|| find_tensor_by_aliases(metadata, OUTPUT_NORM_WEIGHT_ALIASES))
            && desc.num_elements() > 0
        {
            recipe.hidden_dim = desc.num_elements();
        }
        let layers = infer_layer_count_from_metadata(metadata, LAYER_PREFIXES);
        if layers > 0 {
            recipe.num_layers = layers;
        }
        if metadata.head_count() > 0 {
            recipe.n_heads = metadata.head_count().min(recipe.hidden_dim).max(1);
        }
        if let Some(ffn) = metadata.feed_forward_length()
            && ffn > 0
        {
            recipe.ffn_dim = ffn;
        } else if let Some(desc) = find_tensor_by_aliases(metadata, LAYER0_FFN_FC1_BIAS_ALIASES)
            && desc.num_elements() > 0
        {
            recipe.ffn_dim = desc.num_elements();
        } else {
            recipe.ffn_dim = recipe.hidden_dim.saturating_mul(2).max(4);
        }
        if let Some(desc) = find_tensor_by_aliases(metadata, FEATURE_ENC_WEIGHT_ALIASES)
            && recipe.hidden_dim > 0
            && desc.num_elements() >= recipe.hidden_dim
            && desc.num_elements() % recipe.hidden_dim == 0
        {
            recipe.num_features = desc.num_elements() / recipe.hidden_dim;
        }
        if let Some(desc) = find_tensor_by_aliases(metadata, CLASSIFIER_BIAS_ALIASES)
            && desc.num_elements() >= 2
        {
            recipe.num_classes = desc.num_elements();
        } else if let Some(desc) = find_tensor_by_aliases(metadata, CLASSIFIER_WEIGHT_ALIASES)
            && let Some(classes) = infer_paired_2d_dim(desc.shape(), recipe.hidden_dim, false)
        {
            recipe.num_classes = classes;
        }
        if let Some(eps) = metadata.rms_norm_eps()
            && eps.is_finite()
            && eps > 0.0
        {
            recipe.norm_eps = eps;
        }
        recipe
    }

    /// Hydrates a [`TabPfnV2Recipe`] from a parsed [`GgufFile`], reading optional `tabpfn-v2.*` keys.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Self {
        let mut recipe = Self::from_metadata(gguf.metadata());
        for (key, dst) in [
            ("tabpfn-v2.num_features", &mut recipe.num_features),
            ("tabpfn-v2.num_classes", &mut recipe.num_classes),
            ("tabpfn-v2.feed_forward_length", &mut recipe.ffn_dim),
        ] {
            if let Some(v) = gguf.metadata_value(key).and_then(GgufValue::as_usize)
                && v > 0
            {
                *dst = v;
            }
        }
        recipe
    }

    /// Hydrates a [`TabPfnV2Recipe`] from a parsed [`SafeTensorsFile`], reading `"__metadata__"`
    /// keys and tensor descriptors uniformly.
    #[must_use]
    pub fn from_safetensors(st: &SafeTensorsFile<'_>) -> Self {
        let mut recipe = Self::from_metadata(st.metadata());
        if let Some(v) = parse_st_usize(
            st,
            &[
                "tabpfn-v2.embedding_length",
                "hidden_dim",
                "ninp",
                "emsize",
                "hidden_size",
            ],
            1,
        ) {
            recipe.hidden_dim = v;
        }
        if let Some(v) = parse_st_usize(
            st,
            &[
                "tabpfn-v2.block_count",
                "nlayers",
                "num_layers",
                "num_hidden_layers",
            ],
            1,
        ) {
            recipe.num_layers = v;
        }
        if let Some(v) = parse_st_usize(
            st,
            &[
                "tabpfn-v2.attention.head_count",
                "nhead",
                "n_heads",
                "num_attention_heads",
            ],
            1,
        ) {
            recipe.n_heads = v;
        }
        if let Some(v) = parse_st_usize(
            st,
            &[
                "tabpfn-v2.feed_forward_length",
                "ffn_dim",
                "nhid",
                "intermediate_size",
            ],
            1,
        ) {
            recipe.ffn_dim = v;
        }
        if let Some(v) = parse_st_usize(
            st,
            &[
                "tabpfn-v2.num_features",
                "num_features",
                "features_per_group",
            ],
            1,
        ) {
            recipe.num_features = v;
        }
        if let Some(v) = parse_st_usize(st, &["tabpfn-v2.num_classes", "num_classes", "n_out"], 2) {
            recipe.num_classes = v;
        }
        if let Some(eps) = parse_st_f32(st, &["tabpfn-v2.norm_eps", "layer_norm_eps", "norm_eps"]) {
            recipe.norm_eps = eps;
        }
        recipe
    }

    fn validate_geometry(&self) -> Result<usize, K3Error> {
        if self.num_features == 0
            || self.hidden_dim == 0
            || self.num_layers == 0
            || self.n_heads == 0
            || self.hidden_dim % self.n_heads != 0
            || self.ffn_dim == 0
            || self.num_classes < 2
        {
            return Err(mismatch_err(
                "tabpfn-v2",
                format!(
                    "invalid TabPFN-v2 geometry (features={}, hidden={}, layers={}, heads={}, classes={})",
                    self.num_features,
                    self.hidden_dim,
                    self.num_layers,
                    self.n_heads,
                    self.num_classes
                ),
            ));
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err(mismatch_err(
                "tabpfn-v2",
                format!(
                    "tabpfn-v2 norm_eps ({}) must be finite and > 0",
                    self.norm_eps
                ),
            ));
        }
        Ok(self.hidden_dim / self.n_heads)
    }

    fn validate_weights(&self, weights: &dyn ModelWeights) -> Result<(), K3Error> {
        let _ = self.validate_geometry()?;
        let d = self.hidden_dim;
        let row_elems = checked_mul("tabpfn.feature_enc.weight", self.num_features, d)?;
        let _ = require_tensor_1d_aliases(weights, FEATURE_ENC_WEIGHT_ALIASES, row_elems)?;
        let _ = require_tensor_1d_aliases(weights, FEATURE_ENC_BIAS_ALIASES, row_elems)?;
        let _ = require_matvec_view_aliases(weights, Y_EMBD_ALIASES, d, self.num_classes)?;
        let _ = require_tensor_1d_aliases(weights, Y_UNKNOWN_ALIASES, d)?;
        let _ = require_tensor_1d_aliases(weights, OUTPUT_NORM_WEIGHT_ALIASES, d)?;
        let _ = require_tensor_1d_aliases(weights, OUTPUT_NORM_BIAS_ALIASES, d)?;
        let _ =
            require_matvec_view_aliases(weights, CLASSIFIER_WEIGHT_ALIASES, d, self.num_classes)?;
        let _ = require_tensor_1d_aliases(weights, CLASSIFIER_BIAS_ALIASES, self.num_classes)?;
        for layer in 0..self.num_layers {
            let _ = TabPfnLayerViews::load(weights, layer, d, self.ffn_dim)?;
        }
        Ok(())
    }

    fn project_cell_qkv(
        &self,
        hidden: &[f32],
        norm_w: &[f32],
        norm_b: &[f32],
        q_view: &TensorView<'_>,
        k_view: &TensorView<'_>,
        v_view: &TensorView<'_>,
    ) -> Result<QkvProjection, K3Error> {
        let d = self.hidden_dim;
        let total = hidden.len();
        let mut q = Vec::with_capacity(total);
        let mut k = Vec::with_capacity(total);
        let mut v = Vec::with_capacity(total);
        let mut norm_buf = vec![0.0_f32; d];
        for cell in hidden.chunks_exact(d) {
            layernorm_f32(
                &mut norm_buf,
                cell,
                Some(norm_w),
                Some(norm_b),
                self.norm_eps,
            )?;
            q.extend_from_slice(&project_mat_view(q_view, &norm_buf, d, None)?);
            k.extend_from_slice(&project_mat_view(k_view, &norm_buf, d, None)?);
            v.extend_from_slice(&project_mat_view(v_view, &norm_buf, d, None)?);
        }
        Ok((q, k, v))
    }

    fn run_dual_axis_layer(
        &self,
        weights: &dyn ModelWeights,
        layer: usize,
        shape: TabPfnAttentionShape,
        scale: f32,
        hidden: &mut [f32],
    ) -> Result<(), K3Error> {
        let d = self.hidden_dim;
        let views = TabPfnLayerViews::load(weights, layer, d, self.ffn_dim)?;

        let (q_f, k_f, v_f) = self.project_cell_qkv(
            hidden,
            &views.feat.norm_w,
            &views.feat.norm_b,
            &views.feat.q,
            &views.feat.k,
            &views.feat.v,
        )?;
        let mut feat_attn = vec![0.0_f32; hidden.len()];
        tabpfn_dual_axis_attention_f32(
            &mut feat_attn,
            &q_f,
            &k_f,
            &v_f,
            shape,
            TabPfnAttentionAxis::Feature,
            scale,
        )?;
        for (dst_cell, attn_cell) in hidden.chunks_exact_mut(d).zip(feat_attn.chunks_exact(d)) {
            let proj = project_mat_view(&views.feat.o, attn_cell, d, None)?;
            for (dst, &r) in dst_cell.iter_mut().zip(proj.iter()) {
                *dst += r;
            }
        }

        let (q_s, k_s, v_s) = self.project_cell_qkv(
            hidden,
            &views.samp.norm_w,
            &views.samp.norm_b,
            &views.samp.q,
            &views.samp.k,
            &views.samp.v,
        )?;
        let mut samp_attn = vec![0.0_f32; hidden.len()];
        tabpfn_dual_axis_attention_f32(
            &mut samp_attn,
            &q_s,
            &k_s,
            &v_s,
            shape,
            TabPfnAttentionAxis::Sample,
            scale,
        )?;
        for (dst_cell, attn_cell) in hidden.chunks_exact_mut(d).zip(samp_attn.chunks_exact(d)) {
            let proj = project_mat_view(&views.samp.o, attn_cell, d, None)?;
            for (dst, &r) in dst_cell.iter_mut().zip(proj.iter()) {
                *dst += r;
            }
        }

        let mut norm_buf = vec![0.0_f32; d];
        for cell in hidden.chunks_exact_mut(d) {
            layernorm_f32(
                &mut norm_buf,
                cell,
                Some(&views.ffn_norm_w),
                Some(&views.ffn_norm_b),
                self.norm_eps,
            )?;
            let mut mid =
                project_mat_view(&views.fc1_view, &norm_buf, self.ffn_dim, Some(&views.fc1_b))?;
            gelu_inplace_f32(&mut mid)?;
            let down = project_mat_view(&views.fc2_view, &mid, d, Some(&views.fc2_b))?;
            for (dst, &r) in cell.iter_mut().zip(down.iter()) {
                *dst += r;
            }
        }
        Ok(())
    }

    fn validate_tabpfn_inputs<'a>(
        &self,
        ctx: &'a ExecutionContext<'_>,
    ) -> Result<(&'a [f32], usize, usize, usize), K3Error> {
        let features = ctx.feature_input().ok_or_else(|| {
            bounds_err(
                "feature_input",
                self.num_features,
                0,
                "tabpfn-v2 requires ExecutionContext::feature_input [n_samples, n_features]"
                    .to_owned(),
            )
        })?;
        let (n_samples, n_cols) = ctx.feature_dims().ok_or_else(|| {
            bounds_err(
                "feature_input",
                1,
                0,
                "tabpfn-v2 requires feature_dims".to_owned(),
            )
        })?;
        if n_cols != self.num_features || n_samples == 0 {
            return Err(bounds_err(
                "feature_input",
                self.num_features,
                n_cols,
                format!(
                    "tabpfn-v2 expected feature_dims (_, {}), got ({n_samples}, {n_cols})",
                    self.num_features
                ),
            ));
        }
        validate_finite_slice("feature_input", features)?;

        let n_ctx = ctx.prompt_tokens().len();
        if n_ctx == 0 || n_ctx > n_samples {
            return Err(bounds_err(
                "prompt_tokens",
                n_samples,
                n_ctx,
                format!(
                    "tabpfn-v2 context label count ({n_ctx}) in prompt_tokens must lie within 1..={n_samples}"
                ),
            ));
        }
        let n_query = if n_ctx < n_samples {
            n_samples - n_ctx
        } else {
            n_samples
        };
        let needed_logits = checked_mul("tabpfn_logits", n_query, self.num_classes)?;
        if ctx.logits().len() < needed_logits {
            return Err(bounds_err(
                "logits",
                needed_logits,
                ctx.logits().len(),
                format!(
                    "tabpfn-v2 logits length {} < required {needed_logits} ({n_query} queries x {} classes)",
                    ctx.logits().len(),
                    self.num_classes
                ),
            ));
        }
        Ok((features, n_samples, n_ctx, needed_logits))
    }

    fn encode_tabpfn_cells(
        &self,
        features: &[f32],
        context_labels: &[u32],
        weights: &dyn ModelWeights,
        n_samples: usize,
        n_ctx: usize,
    ) -> Result<Vec<f32>, K3Error> {
        let row_elems = checked_mul(
            "tabpfn.feature_enc.weight",
            self.num_features,
            self.hidden_dim,
        )?;
        let enc_w = require_tensor_1d_aliases(weights, FEATURE_ENC_WEIGHT_ALIASES, row_elems)?;
        let enc_b = require_tensor_1d_aliases(weights, FEATURE_ENC_BIAS_ALIASES, row_elems)?;
        let y_embd_view = require_matvec_view_aliases(
            weights,
            Y_EMBD_ALIASES,
            self.hidden_dim,
            self.num_classes,
        )?;
        let y_unknown = require_tensor_1d_aliases(weights, Y_UNKNOWN_ALIASES, self.hidden_dim)?;

        let total_elems = checked_mul("tabpfn_hidden", n_samples, row_elems)?;
        let mut hidden = vec![0.0_f32; total_elems];
        let mut label_vec = vec![0.0_f32; self.hidden_dim];
        for r in 0..n_samples {
            let x_row = &features[r * self.num_features..(r + 1) * self.num_features];
            let dst_row = &mut hidden[r * row_elems..(r + 1) * row_elems];
            tokenize_continuous_features_f32(
                dst_row,
                x_row,
                &enc_w,
                &enc_b,
                self.num_features,
                self.hidden_dim,
            )?;
            let cond = if r < n_ctx {
                let lbl = usize::try_from(context_labels[r]).unwrap_or(usize::MAX);
                if lbl >= self.num_classes {
                    return Err(bounds_err(
                        "prompt_tokens",
                        self.num_classes,
                        lbl,
                        format!(
                            "context label {lbl} at row {r} out of bounds for num_classes={}",
                            self.num_classes
                        ),
                    ));
                }
                dequantize_token_row(&y_embd_view, lbl, &mut label_vec)?;
                validate_finite_slice("tabpfn.y_embd.weight", &label_vec)?;
                &label_vec[..]
            } else {
                &y_unknown[..]
            };
            for cell in dst_row.chunks_exact_mut(self.hidden_dim) {
                for (d_val, &lv) in cell.iter_mut().zip(cond.iter()) {
                    *d_val += lv;
                }
            }
        }
        Ok(hidden)
    }

    fn pool_tabpfn_logits(
        &self,
        weights: &dyn ModelWeights,
        hidden: &[f32],
        n_samples: usize,
        n_ctx: usize,
        needed_logits: usize,
    ) -> Result<Vec<f32>, K3Error> {
        let out_norm_w =
            require_tensor_1d_aliases(weights, OUTPUT_NORM_WEIGHT_ALIASES, self.hidden_dim)?;
        let out_norm_b =
            require_tensor_1d_aliases(weights, OUTPUT_NORM_BIAS_ALIASES, self.hidden_dim)?;
        let cls_view = require_matvec_view_aliases(
            weights,
            CLASSIFIER_WEIGHT_ALIASES,
            self.hidden_dim,
            self.num_classes,
        )?;
        let cls_bias =
            require_tensor_1d_aliases(weights, CLASSIFIER_BIAS_ALIASES, self.num_classes)?;
        let row_elems = self.num_features * self.hidden_dim;
        let query_range = if n_ctx < n_samples {
            n_ctx..n_samples
        } else {
            0..n_samples
        };

        #[expect(
            clippy::cast_precision_loss,
            reason = "num_features fits cleanly in f32"
        )]
        let inv_f = 1.0_f32 / (self.num_features as f32);
        let mut pooled = vec![0.0_f32; self.hidden_dim];
        let mut normed = vec![0.0_f32; self.hidden_dim];
        let mut out_logits = Vec::with_capacity(needed_logits);

        for r in query_range {
            pooled.fill(0.0);
            let row = &hidden[r * row_elems..(r + 1) * row_elems];
            for cell in row.chunks_exact(self.hidden_dim) {
                for (acc, &val) in pooled.iter_mut().zip(cell.iter()) {
                    *acc += val * inv_f;
                }
            }
            layernorm_f32(
                &mut normed,
                &pooled,
                Some(&out_norm_w),
                Some(&out_norm_b),
                self.norm_eps,
            )?;
            let logits = project_mat_view(&cls_view, &normed, self.num_classes, Some(&cls_bias))?;
            out_logits.extend_from_slice(&logits);
        }

        validate_finite_slice("tabpfn_logits", &out_logits)?;
        Ok(out_logits)
    }
}

struct AxisAttnViews<'w> {
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    q: TensorView<'w>,
    k: TensorView<'w>,
    v: TensorView<'w>,
    o: TensorView<'w>,
}

impl<'w> AxisAttnViews<'w> {
    fn load(
        weights: &'w dyn ModelWeights,
        layer: usize,
        d: usize,
        gguf_prefix: &str,
        hf_prefix: &str,
    ) -> Result<Self, K3Error> {
        Ok(Self {
            norm_w: require_tensor_1d_owned(
                weights,
                &tabpfn_layer_aliases(
                    layer,
                    &format!("{gguf_prefix}_norm.weight"),
                    &format!("{hf_prefix}_norm.weight"),
                ),
                d,
            )?,
            norm_b: require_tensor_1d_owned(
                weights,
                &tabpfn_layer_aliases(
                    layer,
                    &format!("{gguf_prefix}_norm.bias"),
                    &format!("{hf_prefix}_norm.bias"),
                ),
                d,
            )?,
            q: require_matvec_view_owned(
                weights,
                &tabpfn_layer_aliases(
                    layer,
                    &format!("{gguf_prefix}_q.weight"),
                    &format!("{hf_prefix}.q_proj.weight"),
                ),
                d,
                d,
            )?,
            k: require_matvec_view_owned(
                weights,
                &tabpfn_layer_aliases(
                    layer,
                    &format!("{gguf_prefix}_k.weight"),
                    &format!("{hf_prefix}.k_proj.weight"),
                ),
                d,
                d,
            )?,
            v: require_matvec_view_owned(
                weights,
                &tabpfn_layer_aliases(
                    layer,
                    &format!("{gguf_prefix}_v.weight"),
                    &format!("{hf_prefix}.v_proj.weight"),
                ),
                d,
                d,
            )?,
            o: require_matvec_view_owned(
                weights,
                &tabpfn_layer_aliases(
                    layer,
                    &format!("{gguf_prefix}_out.weight"),
                    &format!("{hf_prefix}.out_proj.weight"),
                ),
                d,
                d,
            )?,
        })
    }
}

struct TabPfnLayerViews<'w> {
    feat: AxisAttnViews<'w>,
    samp: AxisAttnViews<'w>,
    ffn_norm_w: Vec<f32>,
    ffn_norm_b: Vec<f32>,
    fc1_view: TensorView<'w>,
    fc1_b: Vec<f32>,
    fc2_view: TensorView<'w>,
    fc2_b: Vec<f32>,
}

impl<'w> TabPfnLayerViews<'w> {
    fn load(
        weights: &'w dyn ModelWeights,
        layer: usize,
        d: usize,
        ffn_dim: usize,
    ) -> Result<Self, K3Error> {
        Ok(Self {
            feat: AxisAttnViews::load(weights, layer, d, "feat", "self_attn_between_features")?,
            samp: AxisAttnViews::load(weights, layer, d, "sample", "self_attn_between_items")?,
            ffn_norm_w: require_tensor_1d_owned(
                weights,
                &tabpfn_layer_aliases(layer, "ffn_norm.weight", "mlp_norm.weight"),
                d,
            )?,
            ffn_norm_b: require_tensor_1d_owned(
                weights,
                &tabpfn_layer_aliases(layer, "ffn_norm.bias", "mlp_norm.bias"),
                d,
            )?,
            fc1_view: require_matvec_view_owned(
                weights,
                &tabpfn_layer_aliases(layer, "ffn_fc1.weight", "mlp.linear1.weight"),
                d,
                ffn_dim,
            )?,
            fc1_b: require_tensor_1d_owned(
                weights,
                &tabpfn_layer_aliases(layer, "ffn_fc1.bias", "mlp.linear1.bias"),
                ffn_dim,
            )?,
            fc2_view: require_matvec_view_owned(
                weights,
                &tabpfn_layer_aliases(layer, "ffn_fc2.weight", "mlp.linear2.weight"),
                ffn_dim,
                d,
            )?,
            fc2_b: require_tensor_1d_owned(
                weights,
                &tabpfn_layer_aliases(layer, "ffn_fc2.bias", "mlp.linear2.bias"),
                d,
            )?,
        })
    }
}

impl ArchRecipe for TabPfnV2Recipe {
    fn architecture_id(&self) -> &'static str {
        "tabpfn-v2"
    }

    fn requires_kv_cache(&self) -> bool {
        false
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        is_supported_quant(quant)
    }

    fn validate_model(
        &self,
        metadata: &ModelMetadata,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let _ = self.validate_geometry()?;
        validate_quant_and_arch(self, metadata, weights)?;
        if weights.tensor_count() > 0 {
            self.validate_weights(weights)?;
        }
        Ok(())
    }

    fn forward_classify(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let head_dim = self.validate_geometry()?;
        let (features, n_samples, n_ctx, needed_logits) = self.validate_tabpfn_inputs(ctx)?;
        let context_labels = ctx.prompt_tokens();
        let mut hidden =
            self.encode_tabpfn_cells(features, context_labels, weights, n_samples, n_ctx)?;

        let shape = TabPfnAttentionShape::new(n_samples, self.num_features, self.n_heads, head_dim)
            .with_context_samples(n_ctx);
        #[expect(clippy::cast_precision_loss, reason = "head_dim fits cleanly in f32")]
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        for layer in 0..self.num_layers {
            self.run_dual_axis_layer(weights, layer, shape, scale, &mut hidden)?;
        }

        let out_logits =
            self.pool_tabpfn_logits(weights, &hidden, n_samples, n_ctx, needed_logits)?;
        ctx.logits_mut()[..needed_logits].copy_from_slice(&out_logits);
        Ok(())
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        self.forward_classify(ctx, weights)
    }

    fn decode_step(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        _weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        Err(mismatch_err(
            "tabpfn-v2",
            "TabPFN-v2 is a zero-KV in-context tabular model and does not support autoregressive decode_step",
        ))
    }
}
