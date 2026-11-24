//! Zero-KV `FT-Transformer` (Feature Tokenizer + bidirectional Transformer) tabular recipe.

use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, GgufValue, K3Error, ModelMetadata, ModelWeights,
    QuantType, SafeTensorsFile, TensorView,
};

use crate::common::{
    bounds_err, checked_mul, find_tensor_by_aliases, infer_layer_count_from_metadata,
    infer_paired_2d_dim, lookup_optional_bias_aliases, lookup_tensor_aliases, mismatch_err,
    parse_st_f32, parse_st_usize, project_mat_view, require_matvec_view_aliases,
    require_matvec_view_owned, require_tensor_1d_aliases, require_tensor_1d_owned,
    run_bidirectional_mha, validate_finite_slice, validate_quant_and_arch,
    validate_raw_unquantized_finite,
};
use crate::kernels_compat::{
    dequantize_token_row, gelu_inplace_f32, is_supported_quant, layernorm_f32,
    tokenize_continuous_features_f32,
};

const CLS_TOKEN_ALIASES: &[&str] = &[
    "ft.cls_token.weight",
    "cls_token.weight",
    "feature_tokenizer.cls_token.weight",
    "cls_token",
];
const CONT_WEIGHT_ALIASES: &[&str] = &[
    "ft.cont_weight",
    "feature_tokenizer.num_tokenizer.weight",
    "num_tokenizer.weight",
    "cont_tokenizer.weight",
];
const CONT_BIAS_ALIASES: &[&str] = &[
    "ft.cont_bias",
    "feature_tokenizer.num_tokenizer.bias",
    "num_tokenizer.bias",
    "cont_tokenizer.bias",
];
const CAT_EMBD_ALIASES: &[&str] = &[
    "ft.cat_embd.weight",
    "feature_tokenizer.cat_tokenizer.weight",
    "cat_tokenizer.weight",
    "cat_embd.weight",
];
const CAT_BIAS_ALIASES: &[&str] = &[
    "ft.cat_bias",
    "feature_tokenizer.cat_tokenizer.bias",
    "cat_tokenizer.bias",
    "cat_bias",
];
const OUTPUT_NORM_WEIGHT_ALIASES: &[&str] = &[
    "output_norm.weight",
    "transformer.head.normalization.weight",
    "head.normalization.weight",
    "norm.weight",
];
const OUTPUT_NORM_BIAS_ALIASES: &[&str] = &[
    "output_norm.bias",
    "transformer.head.normalization.bias",
    "head.normalization.bias",
    "norm.bias",
];
const HEAD_PAIRS: &[(&str, &str)] = &[
    ("ft.head.weight", "ft.head.bias"),
    ("classifier.weight", "classifier.bias"),
    (
        "transformer.head.linear.weight",
        "transformer.head.linear.bias",
    ),
    ("head.linear.weight", "head.linear.bias"),
    ("head.weight", "head.bias"),
];
const HEAD_WEIGHT_ALIASES: &[&str] = &[
    "ft.head.weight",
    "classifier.weight",
    "transformer.head.linear.weight",
    "head.linear.weight",
    "head.weight",
];
const HEAD_BIAS_ALIASES: &[&str] = &[
    "ft.head.bias",
    "classifier.bias",
    "transformer.head.linear.bias",
    "head.linear.bias",
    "head.bias",
];
const LAYER0_FFN_FC1_BIAS_ALIASES: &[&str] = &[
    "blk.0.ffn_fc1.bias",
    "transformer.blocks.0.ffn.linear_first.bias",
    "blocks.0.ffn.linear_first.bias",
    "layers.0.ffn_fc1.bias",
];
const LAYER_PREFIXES: &[&str] = &["blk.", "transformer.blocks.", "blocks.", "layers."];

fn ft_layer_aliases(layer: usize, gguf_suffix: &str, hf_suffix: &str) -> [String; 4] {
    [
        format!("blk.{layer}.{gguf_suffix}"),
        format!("transformer.blocks.{layer}.{hf_suffix}"),
        format!("blocks.{layer}.{hf_suffix}"),
        format!("layers.{layer}.{gguf_suffix}"),
    ]
}

fn select_head_pair(weights: &dyn ModelWeights) -> (&'static str, &'static str) {
    for &(w, b) in HEAD_PAIRS {
        if weights.contains_tensor(w) {
            return (w, b);
        }
    }
    ("ft.head.weight", "ft.head.bias")
}

/// Zero-KV `FT-Transformer` (Feature Tokenizer + bidirectional Transformer) tabular recipe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FtTransformerRecipe {
    /// Number of continuous numerical input features.
    pub num_continuous: usize,
    /// Number of categorical input feature columns.
    pub num_categorical: usize,
    /// Per-feature token embedding width (`d_token`).
    pub d_token: usize,
    /// Number of bidirectional Transformer encoder layers.
    pub num_layers: usize,
    /// Number of attention heads dividing `d_token`.
    pub n_heads: usize,
    /// Feed-forward intermediate width per encoder block.
    pub ffn_dim: usize,
    /// Output scoring/logit dimension (`1` for regression/binary score or `num_classes`).
    pub out_dim: usize,
    /// `LayerNorm` epsilon (`1e-5` default).
    pub norm_eps: f32,
}

impl Default for FtTransformerRecipe {
    fn default() -> Self {
        Self::new()
    }
}

impl FtTransformerRecipe {
    /// Creates a baseline `ft-transformer` recipe.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            num_continuous: 4,
            num_categorical: 0,
            d_token: 16,
            num_layers: 2,
            n_heads: 2,
            ffn_dim: 32,
            out_dim: 2,
            norm_eps: 1e-5,
        }
    }

    /// Hydrates an [`FtTransformerRecipe`] from [`ModelMetadata`] and tensor descriptors.
    #[must_use]
    pub fn from_metadata(metadata: &ModelMetadata) -> Self {
        let mut recipe = Self::new();
        if metadata.embedding_dim() > 0 {
            recipe.d_token = metadata.embedding_dim();
        } else if let Some(desc) = find_tensor_by_aliases(metadata, CLS_TOKEN_ALIASES)
            .or_else(|| find_tensor_by_aliases(metadata, OUTPUT_NORM_WEIGHT_ALIASES))
            && desc.num_elements() > 0
        {
            recipe.d_token = desc.num_elements();
        }
        let layers = infer_layer_count_from_metadata(metadata, LAYER_PREFIXES);
        if layers > 0 {
            recipe.num_layers = layers;
        }
        if metadata.head_count() > 0 {
            recipe.n_heads = metadata.head_count().min(recipe.d_token).max(1);
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
            recipe.ffn_dim = recipe.d_token.saturating_mul(2).max(4);
        }
        if let Some(desc) = find_tensor_by_aliases(metadata, CONT_WEIGHT_ALIASES)
            && recipe.d_token > 0
            && desc.num_elements() >= recipe.d_token
            && desc.num_elements() % recipe.d_token == 0
        {
            recipe.num_continuous = desc.num_elements() / recipe.d_token;
        } else if find_tensor_by_aliases(metadata, CAT_EMBD_ALIASES).is_some() {
            recipe.num_continuous = 0;
        }
        if let Some(desc) = find_tensor_by_aliases(metadata, CAT_BIAS_ALIASES)
            && recipe.d_token > 0
            && desc.num_elements() >= recipe.d_token
            && desc.num_elements() % recipe.d_token == 0
        {
            recipe.num_categorical = desc.num_elements() / recipe.d_token;
        }
        if let Some(desc) = find_tensor_by_aliases(metadata, HEAD_BIAS_ALIASES)
            && desc.num_elements() > 0
        {
            recipe.out_dim = desc.num_elements();
        } else if let Some(desc) = find_tensor_by_aliases(metadata, HEAD_WEIGHT_ALIASES)
            && let Some(out_dim) = infer_paired_2d_dim(desc.shape(), recipe.d_token, false)
        {
            recipe.out_dim = out_dim;
        }
        if let Some(eps) = metadata.rms_norm_eps()
            && eps.is_finite()
            && eps > 0.0
        {
            recipe.norm_eps = eps;
        }
        recipe
    }

    /// Hydrates an [`FtTransformerRecipe`] from a parsed [`GgufFile`], reading optional `ft.*` keys.
    #[must_use]
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Self {
        let mut recipe = Self::from_metadata(gguf.metadata());
        for (prefix, field) in [
            ("ft-transformer.num_continuous", &mut recipe.num_continuous),
            (
                "ft-transformer.num_categorical",
                &mut recipe.num_categorical,
            ),
            ("ft-transformer.out_dim", &mut recipe.out_dim),
        ] {
            if let Some(v) = gguf.metadata_value(prefix).and_then(GgufValue::as_usize) {
                *field = v;
            }
        }
        if let Some(v) = gguf
            .metadata_value("ft-transformer.feed_forward_length")
            .and_then(GgufValue::as_usize)
            && v > 0
        {
            recipe.ffn_dim = v;
        }
        recipe
    }

    /// Hydrates an [`FtTransformerRecipe`] from a parsed [`SafeTensorsFile`], reading
    /// `"__metadata__"` keys and tensor descriptors uniformly.
    #[must_use]
    pub fn from_safetensors(st: &SafeTensorsFile<'_>) -> Self {
        let mut recipe = Self::from_metadata(st.metadata());
        if let Some(v) = parse_st_usize(
            st,
            &["ft-transformer.embedding_length", "d_token", "hidden_size"],
            1,
        ) {
            recipe.d_token = v;
        }
        if let Some(v) = parse_st_usize(
            st,
            &[
                "ft-transformer.block_count",
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
                "ft-transformer.attention.head_count",
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
                "ft-transformer.feed_forward_length",
                "ffn_dim",
                "d_ffn",
                "intermediate_size",
            ],
            1,
        ) {
            recipe.ffn_dim = v;
        }
        if let Some(v) = parse_st_usize(
            st,
            &[
                "ft-transformer.num_continuous",
                "num_continuous",
                "n_num_features",
            ],
            0,
        ) {
            recipe.num_continuous = v;
        }
        if let Some(v) = parse_st_usize(
            st,
            &[
                "ft-transformer.num_categorical",
                "num_categorical",
                "n_cat_features",
            ],
            0,
        ) {
            recipe.num_categorical = v;
        }
        if let Some(v) = parse_st_usize(
            st,
            &["ft-transformer.out_dim", "out_dim", "d_out", "num_classes"],
            1,
        ) {
            recipe.out_dim = v;
        }
        if let Some(eps) = parse_st_f32(
            st,
            &["ft-transformer.norm_eps", "layer_norm_eps", "norm_eps"],
        ) {
            recipe.norm_eps = eps;
        }
        recipe
    }

    fn validate_geometry(&self) -> Result<usize, K3Error> {
        let total_features = self.num_continuous + self.num_categorical;
        if total_features == 0
            || self.d_token == 0
            || self.num_layers == 0
            || self.n_heads == 0
            || self.d_token % self.n_heads != 0
            || self.ffn_dim == 0
            || self.out_dim == 0
        {
            return Err(mismatch_err(
                "ft-transformer",
                format!(
                    "invalid FT-Transformer geometry (cont={}, cat={}, d_token={}, heads={}, ffn={}, out={})",
                    self.num_continuous,
                    self.num_categorical,
                    self.d_token,
                    self.n_heads,
                    self.ffn_dim,
                    self.out_dim
                ),
            ));
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err(mismatch_err(
                "ft-transformer",
                format!(
                    "ft-transformer norm_eps ({}) must be finite and > 0",
                    self.norm_eps
                ),
            ));
        }
        Ok(self.d_token / self.n_heads)
    }

    fn validate_weights(&self, weights: &dyn ModelWeights) -> Result<(), K3Error> {
        let _ = self.validate_geometry()?;
        let _ = self.load_shared_tensors(weights)?;
        let d = self.d_token;
        for layer in 0..self.num_layers {
            let _ = require_tensor_1d_owned(
                weights,
                &ft_layer_aliases(layer, "attn_norm.weight", "attention_normalization.weight"),
                d,
            )?;
            let _ = require_tensor_1d_owned(
                weights,
                &ft_layer_aliases(layer, "attn_norm.bias", "attention_normalization.bias"),
                d,
            )?;
            let _ = require_matvec_view_owned(
                weights,
                &ft_layer_aliases(layer, "attn_q.weight", "attention.W_q.weight"),
                d,
                d,
            )?;
            let _ = require_matvec_view_owned(
                weights,
                &ft_layer_aliases(layer, "attn_k.weight", "attention.W_k.weight"),
                d,
                d,
            )?;
            let _ = require_matvec_view_owned(
                weights,
                &ft_layer_aliases(layer, "attn_v.weight", "attention.W_v.weight"),
                d,
                d,
            )?;
            let _ = require_matvec_view_owned(
                weights,
                &ft_layer_aliases(layer, "attn_output.weight", "attention.W_out.weight"),
                d,
                d,
            )?;
            let _ = require_tensor_1d_owned(
                weights,
                &ft_layer_aliases(layer, "ffn_norm.weight", "ffn_normalization.weight"),
                d,
            )?;
            let _ = require_tensor_1d_owned(
                weights,
                &ft_layer_aliases(layer, "ffn_norm.bias", "ffn_normalization.bias"),
                d,
            )?;
            let _ = require_matvec_view_owned(
                weights,
                &ft_layer_aliases(layer, "ffn_fc1.weight", "ffn.linear_first.weight"),
                d,
                self.ffn_dim,
            )?;
            let _ = require_tensor_1d_owned(
                weights,
                &ft_layer_aliases(layer, "ffn_fc1.bias", "ffn.linear_first.bias"),
                self.ffn_dim,
            )?;
            let _ = require_matvec_view_owned(
                weights,
                &ft_layer_aliases(layer, "ffn_fc2.weight", "ffn.linear_second.weight"),
                self.ffn_dim,
                d,
            )?;
            let _ = require_tensor_1d_owned(
                weights,
                &ft_layer_aliases(layer, "ffn_fc2.bias", "ffn.linear_second.bias"),
                d,
            )?;
        }
        Ok(())
    }

    fn load_shared_tensors<'w>(
        &self,
        weights: &'w dyn ModelWeights,
    ) -> Result<FtSharedTensors<'w>, K3Error> {
        let cls_embd = require_tensor_1d_aliases(weights, CLS_TOKEN_ALIASES, self.d_token)?;
        let cont_wb = if self.num_continuous > 0 {
            let need = checked_mul("ft.cont_weight", self.num_continuous, self.d_token)?;
            let w = require_tensor_1d_aliases(weights, CONT_WEIGHT_ALIASES, need)?;
            let b = require_tensor_1d_aliases(weights, CONT_BIAS_ALIASES, need)?;
            Some((w, b))
        } else {
            None
        };
        let cat_view = if self.num_categorical > 0 {
            let view = lookup_tensor_aliases(weights, CAT_EMBD_ALIASES)?;
            let shape = view.shape();
            if shape.len() == 2 && shape[0] != self.d_token && shape[1] != self.d_token {
                return Err(bounds_err(
                    view.name(),
                    self.d_token,
                    view.num_elements(),
                    format!(
                        "{} expected dimension {} in shape {shape:?}",
                        view.name(),
                        self.d_token
                    ),
                ));
            }
            if view.num_elements() < self.d_token || view.num_elements() % self.d_token != 0 {
                return Err(bounds_err(
                    view.name(),
                    self.d_token,
                    view.num_elements(),
                    format!("{} must hold whole d_token rows", view.name()),
                ));
            }
            validate_raw_unquantized_finite(&view, view.name())?;
            Some(view)
        } else {
            None
        };
        let cat_bias = if self.num_categorical > 0 {
            let bias_len = checked_mul("ft.cat_bias", self.num_categorical, self.d_token)?;
            lookup_optional_bias_aliases(weights, CAT_BIAS_ALIASES, bias_len)?
        } else {
            None
        };

        let out_norm_w =
            require_tensor_1d_aliases(weights, OUTPUT_NORM_WEIGHT_ALIASES, self.d_token)?;
        let out_norm_b =
            require_tensor_1d_aliases(weights, OUTPUT_NORM_BIAS_ALIASES, self.d_token)?;
        let (head_name, bias_name) = select_head_pair(weights);
        let head_view =
            require_matvec_view_aliases(weights, &[head_name], self.d_token, self.out_dim)?;
        let head_bias = require_tensor_1d_aliases(weights, &[bias_name], self.out_dim)?;

        Ok(FtSharedTensors {
            cls_embd,
            cont_wb,
            cat_view,
            cat_bias,
            out_norm_w,
            out_norm_b,
            head_view,
            head_bias,
        })
    }

    fn encode_row_tokens(
        &self,
        row_features: &[f32],
        prompt_cat_ids: &[u32],
        shared: &FtSharedTensors<'_>,
    ) -> Result<Vec<f32>, K3Error> {
        let seq_len = 1 + self.num_continuous + self.num_categorical;
        let mut seq = vec![0.0_f32; seq_len * self.d_token];
        seq[..self.d_token].copy_from_slice(&shared.cls_embd);

        let mut token_offset = self.d_token;
        if self.num_continuous > 0 {
            let (cw, cb) = shared.cont_wb.as_ref().ok_or_else(|| {
                bounds_err(
                    "ft.cont_weight",
                    1,
                    0,
                    "missing continuous tokenizer weights".to_owned(),
                )
            })?;
            let cont_slice = &row_features[..self.num_continuous];
            let cont_len = self.num_continuous * self.d_token;
            tokenize_continuous_features_f32(
                &mut seq[token_offset..token_offset + cont_len],
                cont_slice,
                cw,
                cb,
                self.num_continuous,
                self.d_token,
            )?;
            token_offset += cont_len;
        }

        if self.num_categorical > 0 {
            let cat_view = shared.cat_view.as_ref().ok_or_else(|| {
                bounds_err(
                    "ft.cat_embd.weight",
                    1,
                    0,
                    "missing categorical embedding table".to_owned(),
                )
            })?;
            let cat_rows = cat_view.num_elements() / self.d_token;
            for cat_idx in 0..self.num_categorical {
                let id_usize = if row_features.len() == self.num_continuous + self.num_categorical {
                    let raw = row_features[self.num_continuous + cat_idx];
                    #[expect(
                        clippy::float_cmp,
                        reason = "categorical ID validated non-negative and exact integer before cast"
                    )]
                    if !raw.is_finite() || raw < 0.0 || raw.trunc() != raw {
                        return Err(bounds_err(
                            "feature_input",
                            cat_idx,
                            row_features.len(),
                            format!(
                                "categorical feature at col {} must be non-negative integer, got {raw}",
                                self.num_continuous + cat_idx
                            ),
                        ));
                    }
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "validated exact non-negative finite integer"
                    )]
                    {
                        raw as usize
                    }
                } else {
                    usize::try_from(prompt_cat_ids[cat_idx]).map_err(|_| {
                        bounds_err(
                            "prompt_tokens",
                            cat_idx,
                            prompt_cat_ids.len(),
                            "categorical ID overflow".to_owned(),
                        )
                    })?
                };
                if id_usize >= cat_rows {
                    return Err(bounds_err(
                        cat_view.name(),
                        id_usize + 1,
                        cat_rows,
                        format!(
                            "categorical index {id_usize} out of bounds for table rows {cat_rows}"
                        ),
                    ));
                }
                let dst = &mut seq[token_offset + cat_idx * self.d_token
                    ..token_offset + (cat_idx + 1) * self.d_token];
                dequantize_token_row(cat_view, id_usize, dst)?;
                if let Some(bias) = shared.cat_bias.as_deref() {
                    let b_slice = &bias[cat_idx * self.d_token..(cat_idx + 1) * self.d_token];
                    for (d, &bv) in dst.iter_mut().zip(b_slice.iter()) {
                        *d += bv;
                    }
                }
                validate_finite_slice("ft_cat_token", dst)?;
            }
        }
        Ok(seq)
    }

    fn run_encoder_layer(
        &self,
        weights: &dyn ModelWeights,
        layer: usize,
        seq_len: usize,
        head_dim: usize,
        hidden: &mut [f32],
    ) -> Result<(), K3Error> {
        self.run_encoder_attn_sublayer(weights, layer, seq_len, head_dim, hidden)?;
        self.run_encoder_ffn_sublayer(weights, layer, hidden)
    }

    fn run_encoder_attn_sublayer(
        &self,
        weights: &dyn ModelWeights,
        layer: usize,
        seq_len: usize,
        head_dim: usize,
        hidden: &mut [f32],
    ) -> Result<(), K3Error> {
        let d = self.d_token;
        let attn_norm_w = require_tensor_1d_owned(
            weights,
            &ft_layer_aliases(layer, "attn_norm.weight", "attention_normalization.weight"),
            d,
        )?;
        let attn_norm_b = require_tensor_1d_owned(
            weights,
            &ft_layer_aliases(layer, "attn_norm.bias", "attention_normalization.bias"),
            d,
        )?;
        let q_view = require_matvec_view_owned(
            weights,
            &ft_layer_aliases(layer, "attn_q.weight", "attention.W_q.weight"),
            d,
            d,
        )?;
        let k_view = require_matvec_view_owned(
            weights,
            &ft_layer_aliases(layer, "attn_k.weight", "attention.W_k.weight"),
            d,
            d,
        )?;
        let v_view = require_matvec_view_owned(
            weights,
            &ft_layer_aliases(layer, "attn_v.weight", "attention.W_v.weight"),
            d,
            d,
        )?;
        let o_view = require_matvec_view_owned(
            weights,
            &ft_layer_aliases(layer, "attn_output.weight", "attention.W_out.weight"),
            d,
            d,
        )?;

        let mut q_seq = Vec::with_capacity(seq_len * d);
        let mut k_seq = Vec::with_capacity(seq_len * d);
        let mut v_seq = Vec::with_capacity(seq_len * d);
        let mut norm_buf = vec![0.0_f32; d];
        for tok in hidden.chunks_exact(d) {
            layernorm_f32(
                &mut norm_buf,
                tok,
                Some(&attn_norm_w),
                Some(&attn_norm_b),
                self.norm_eps,
            )?;
            q_seq.extend_from_slice(&project_mat_view(&q_view, &norm_buf, d, None)?);
            k_seq.extend_from_slice(&project_mat_view(&k_view, &norm_buf, d, None)?);
            v_seq.extend_from_slice(&project_mat_view(&v_view, &norm_buf, d, None)?);
        }

        let mha = run_bidirectional_mha(&q_seq, &k_seq, &v_seq, seq_len, self.n_heads, head_dim)?;
        for (t_idx, tok_out) in mha.chunks_exact(d).enumerate() {
            let proj = project_mat_view(&o_view, tok_out, d, None)?;
            for (dst, &r) in hidden[t_idx * d..(t_idx + 1) * d]
                .iter_mut()
                .zip(proj.iter())
            {
                *dst += r;
            }
        }
        Ok(())
    }

    fn run_encoder_ffn_sublayer(
        &self,
        weights: &dyn ModelWeights,
        layer: usize,
        hidden: &mut [f32],
    ) -> Result<(), K3Error> {
        let d = self.d_token;
        let ffn_norm_w = require_tensor_1d_owned(
            weights,
            &ft_layer_aliases(layer, "ffn_norm.weight", "ffn_normalization.weight"),
            d,
        )?;
        let ffn_norm_b = require_tensor_1d_owned(
            weights,
            &ft_layer_aliases(layer, "ffn_norm.bias", "ffn_normalization.bias"),
            d,
        )?;
        let fc1_view = require_matvec_view_owned(
            weights,
            &ft_layer_aliases(layer, "ffn_fc1.weight", "ffn.linear_first.weight"),
            d,
            self.ffn_dim,
        )?;
        let fc1_b = require_tensor_1d_owned(
            weights,
            &ft_layer_aliases(layer, "ffn_fc1.bias", "ffn.linear_first.bias"),
            self.ffn_dim,
        )?;
        let fc2_view = require_matvec_view_owned(
            weights,
            &ft_layer_aliases(layer, "ffn_fc2.weight", "ffn.linear_second.weight"),
            self.ffn_dim,
            d,
        )?;
        let fc2_b = require_tensor_1d_owned(
            weights,
            &ft_layer_aliases(layer, "ffn_fc2.bias", "ffn.linear_second.bias"),
            d,
        )?;

        let mut norm_buf = vec![0.0_f32; d];
        for tok in hidden.chunks_exact_mut(d) {
            layernorm_f32(
                &mut norm_buf,
                tok,
                Some(&ffn_norm_w),
                Some(&ffn_norm_b),
                self.norm_eps,
            )?;
            let mut mid = project_mat_view(&fc1_view, &norm_buf, self.ffn_dim, Some(&fc1_b))?;
            gelu_inplace_f32(&mut mid)?;
            let down = project_mat_view(&fc2_view, &mid, d, Some(&fc2_b))?;
            for (dst, &r) in tok.iter_mut().zip(down.iter()) {
                *dst += r;
            }
        }
        Ok(())
    }

    fn validate_ft_inputs<'a>(
        &self,
        ctx: &'a ExecutionContext<'_>,
    ) -> Result<(&'a [f32], usize, usize, usize), K3Error> {
        let features = ctx.feature_input().ok_or_else(|| {
            bounds_err(
                "feature_input",
                self.num_continuous + self.num_categorical,
                0,
                "ft-transformer requires ExecutionContext::feature_input tabular matrix".to_owned(),
            )
        })?;
        let (num_rows, num_cols) = ctx.feature_dims().ok_or_else(|| {
            bounds_err(
                "feature_input",
                1,
                0,
                "ft-transformer requires feature_dims".to_owned(),
            )
        })?;
        validate_finite_slice("feature_input", features)?;

        let prompt_cats = ctx.prompt_tokens();
        let expects_combined = self.num_continuous + self.num_categorical;
        let uses_prompt_cats =
            self.num_categorical > 0 && num_cols == self.num_continuous && num_rows == 1;
        if num_cols != expects_combined && !uses_prompt_cats {
            return Err(bounds_err(
                "feature_input",
                expects_combined,
                num_cols,
                format!(
                    "ft-transformer expected {expects_combined} feature cols (or {} continuous cols with prompt categorical IDs), got {num_cols}",
                    self.num_continuous
                ),
            ));
        }
        if uses_prompt_cats && prompt_cats.len() != self.num_categorical {
            return Err(bounds_err(
                "prompt_tokens",
                self.num_categorical,
                prompt_cats.len(),
                format!(
                    "ft-transformer expected {} categorical IDs in prompt_tokens, got {}",
                    self.num_categorical,
                    prompt_cats.len()
                ),
            ));
        }

        let needed_logits = checked_mul("ft_logits", num_rows, self.out_dim)?;
        if ctx.logits().len() < needed_logits {
            return Err(bounds_err(
                "logits",
                needed_logits,
                ctx.logits().len(),
                format!(
                    "ft-transformer logits slice length {} < required {needed_logits}",
                    ctx.logits().len()
                ),
            ));
        }
        Ok((features, num_rows, num_cols, needed_logits))
    }
}

struct FtSharedTensors<'w> {
    cls_embd: Vec<f32>,
    cont_wb: Option<(Vec<f32>, Vec<f32>)>,
    cat_view: Option<TensorView<'w>>,
    cat_bias: Option<Vec<f32>>,
    out_norm_w: Vec<f32>,
    out_norm_b: Vec<f32>,
    head_view: TensorView<'w>,
    head_bias: Vec<f32>,
}

impl ArchRecipe for FtTransformerRecipe {
    fn architecture_id(&self) -> &'static str {
        "ft-transformer"
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
        let (features, num_rows, num_cols, needed_logits) = self.validate_ft_inputs(ctx)?;
        let prompt_cats = ctx.prompt_tokens();
        let shared = self.load_shared_tensors(weights)?;
        let seq_len = 1 + self.num_continuous + self.num_categorical;

        let mut all_logits = Vec::with_capacity(needed_logits);
        for row_idx in 0..num_rows {
            let row = &features[row_idx * num_cols..(row_idx + 1) * num_cols];
            let mut hidden = self.encode_row_tokens(row, prompt_cats, &shared)?;
            for layer in 0..self.num_layers {
                self.run_encoder_layer(weights, layer, seq_len, head_dim, &mut hidden)?;
            }
            let mut cls_normed = vec![0.0_f32; self.d_token];
            layernorm_f32(
                &mut cls_normed,
                &hidden[..self.d_token],
                Some(&shared.out_norm_w),
                Some(&shared.out_norm_b),
                self.norm_eps,
            )?;
            let row_logits = project_mat_view(
                &shared.head_view,
                &cls_normed,
                self.out_dim,
                Some(&shared.head_bias),
            )?;
            all_logits.extend_from_slice(&row_logits);
        }

        validate_finite_slice("ft_logits", &all_logits)?;
        ctx.logits_mut()[..needed_logits].copy_from_slice(&all_logits);
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
            "ft-transformer",
            "FT-Transformer is a zero-KV single-pass tabular model and does not support autoregressive decode_step",
        ))
    }
}
