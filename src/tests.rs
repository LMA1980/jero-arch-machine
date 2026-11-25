#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "bounded synthetic test fixtures fit cleanly in u32 and f32"
)]

use jero_core::safetensors::test_helpers::create_synthetic_safetensors;
use jero_core::test_helpers::create_synthetic_gguf_v3_with_kvs;
use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, GgufValue, K3Error, ModelWeights, QuantType,
    SafeTensorsFile, sanitize_and_normalize_features,
};

use crate::kernels_compat::{
    TabPfnAttentionAxis, TabPfnAttentionShape, tabpfn_dual_axis_attention_f32,
};
use crate::{
    CRATE_LICENSE, CRATE_NAME, DecisionTransformerRecipe, FtTransformerRecipe, MmprojLatentBridge,
    TabPfnV2Recipe, TtmRecipe, crate_description, forward_with_soft_prompt,
    project_machine_latent_to_soft_prompt,
};

fn build_gguf_f32<'a>(
    arch: &str,
    extra_kvs: &[(&'a str, GgufValue<'a>)],
    tensors: &[(&str, &[usize], Vec<f32>)],
) -> Vec<u8> {
    let specs: Vec<(&str, QuantType, &[usize])> = tensors
        .iter()
        .map(|(name, shape, _)| (*name, QuantType::F32, *shape))
        .collect();
    let mut buf = create_synthetic_gguf_v3_with_kvs(arch, 32, extra_kvs, &specs);
    let offsets: Vec<(String, usize, usize)> = {
        let gguf = GgufFile::parse(&buf).expect("parse scaffold gguf");
        gguf.metadata()
            .tensors()
            .iter()
            .map(|d| (d.name().to_owned(), d.absolute_offset(), d.byte_len()))
            .collect()
    };
    for ((name, _, values), (off_name, abs_off, byte_len)) in tensors.iter().zip(offsets.iter()) {
        assert_eq!(name, off_name);
        assert_eq!(values.len() * 4, *byte_len);
        for (idx, &v) in values.iter().enumerate() {
            let start = abs_off + idx * 4;
            buf[start..start + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    buf
}

fn build_safetensors_f32(
    custom_metadata: &[(&str, &str)],
    tensors: &[(&str, &[usize], Vec<f32>)],
) -> Vec<u8> {
    let raw_storage: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, _, vals)| vals.iter().flat_map(|v| v.to_le_bytes()).collect())
        .collect();
    let st_specs: Vec<(&str, QuantType, &[usize], &[u8])> = tensors
        .iter()
        .zip(raw_storage.iter())
        .map(|((name, shape, _), raw)| (*name, QuantType::F32, *shape, raw.as_slice()))
        .collect();
    create_synthetic_safetensors(custom_metadata, &st_specs)
}

fn assert_f32_slice_parity(lhs: &[f32], rhs: &[f32], label: &str) {
    assert_eq!(lhs.len(), rhs.len(), "{label}: length mismatch");
    for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "{label}[{idx}]: non-finite ({a}, {b})"
        );
        #[cfg(not(miri))]
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{label}[{idx}]: native output must match bit-exact ({a} vs {b})"
        );
        #[cfg(miri)]
        {
            let tol = 1e-5_f32 * a.abs().max(b.abs()).max(1.0_f32);
            assert!(
                (a - b).abs() <= tol,
                "{label}[{idx}]: Miri output exceeded tolerance ({a} vs {b}, tol={tol})"
            );
        }
    }
}

fn make_ttm_tensors(recipe: &TtmRecipe) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
    let n_p = recipe.num_patches();
    let c = recipe.num_channels;
    let th = recipe.time_hidden_dim;
    let ch = recipe.channel_hidden_dim;
    assert_eq!(recipe.num_mixer_layers, 1);
    vec![
        (
            "ttm.patch_proj.weight",
            vec![recipe.patch_length],
            vec![1.0 / (recipe.patch_length as f32); recipe.patch_length],
        ),
        ("ttm.patch_proj.bias", vec![1], vec![0.0]),
        (
            "ttm.forecast_head.weight",
            vec![n_p, recipe.forecast_length],
            (0..n_p * recipe.forecast_length)
                .map(|i| 0.1 + ((i % 5) as f32) * 0.03)
                .collect(),
        ),
        (
            "ttm.forecast_head.bias",
            vec![recipe.forecast_length],
            vec![0.02; recipe.forecast_length],
        ),
        ("blk.0.time_norm.weight", vec![n_p], vec![1.0; n_p]),
        ("blk.0.time_norm.bias", vec![n_p], vec![0.0; n_p]),
        (
            "blk.0.time_mlp.fc1.weight",
            vec![n_p, th],
            vec![0.05; n_p * th],
        ),
        ("blk.0.time_mlp.fc1.bias", vec![th], vec![0.01; th]),
        (
            "blk.0.time_mlp.fc2.weight",
            vec![th, n_p],
            vec![0.04; th * n_p],
        ),
        ("blk.0.time_mlp.fc2.bias", vec![n_p], vec![0.0; n_p]),
        ("blk.0.channel_norm.weight", vec![c], vec![1.0; c]),
        ("blk.0.channel_norm.bias", vec![c], vec![0.0; c]),
        (
            "blk.0.channel_mlp.fc1.weight",
            vec![c, ch],
            vec![0.06; c * ch],
        ),
        ("blk.0.channel_mlp.fc1.bias", vec![ch], vec![0.01; ch]),
        (
            "blk.0.channel_mlp.fc2.weight",
            vec![ch, c],
            vec![0.03; ch * c],
        ),
        ("blk.0.channel_mlp.fc2.bias", vec![c], vec![0.0; c]),
    ]
}

fn ttm_hf_alias_name(gguf_name: &str) -> &'static str {
    match gguf_name {
        "ttm.patch_proj.weight" => "backbone.patcher.weight",
        "ttm.patch_proj.bias" => "backbone.patcher.bias",
        "ttm.forecast_head.weight" => "head.projection.weight",
        "ttm.forecast_head.bias" => "head.projection.bias",
        "blk.0.time_norm.weight" => "backbone.encoder.layers.0.patch_mixer.norm.weight",
        "blk.0.time_norm.bias" => "backbone.encoder.layers.0.patch_mixer.norm.bias",
        "blk.0.time_mlp.fc1.weight" => "backbone.encoder.layers.0.patch_mixer.mlp.fc1.weight",
        "blk.0.time_mlp.fc1.bias" => "backbone.encoder.layers.0.patch_mixer.mlp.fc1.bias",
        "blk.0.time_mlp.fc2.weight" => "backbone.encoder.layers.0.patch_mixer.mlp.fc2.weight",
        "blk.0.time_mlp.fc2.bias" => "backbone.encoder.layers.0.patch_mixer.mlp.fc2.bias",
        "blk.0.channel_norm.weight" => "backbone.encoder.layers.0.channel_mixer.norm.weight",
        "blk.0.channel_norm.bias" => "backbone.encoder.layers.0.channel_mixer.norm.bias",
        "blk.0.channel_mlp.fc1.weight" => "backbone.encoder.layers.0.channel_mixer.mlp.fc1.weight",
        "blk.0.channel_mlp.fc1.bias" => "backbone.encoder.layers.0.channel_mixer.mlp.fc1.bias",
        "blk.0.channel_mlp.fc2.weight" => "backbone.encoder.layers.0.channel_mixer.mlp.fc2.weight",
        "blk.0.channel_mlp.fc2.bias" => "backbone.encoder.layers.0.channel_mixer.mlp.fc2.bias",
        other => panic!("unexpected ttm tensor {other}"),
    }
}

fn ft_layer0_tensors(recipe: &FtTransformerRecipe) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
    let d = recipe.d_token;
    vec![
        ("blk.0.attn_norm.weight", vec![d], vec![1.0; d]),
        ("blk.0.attn_norm.bias", vec![d], vec![0.0; d]),
        (
            "blk.0.attn_q.weight",
            vec![d, d],
            (0..d * d)
                .map(|i| 0.04 + ((i % 11) as f32) * 0.015)
                .collect(),
        ),
        (
            "blk.0.attn_k.weight",
            vec![d, d],
            (0..d * d)
                .map(|i| 0.03 + (((i + 3) % 9) as f32) * 0.015)
                .collect(),
        ),
        (
            "blk.0.attn_v.weight",
            vec![d, d],
            (0..d * d)
                .map(|i| 0.05 + (((i * 3) % 13) as f32 - 4.0) * 0.02)
                .collect(),
        ),
        (
            "blk.0.attn_output.weight",
            vec![d, d],
            (0..d * d)
                .map(|i| 0.04 + (((i * 5) % 11) as f32 - 3.0) * 0.02)
                .collect(),
        ),
        ("blk.0.ffn_norm.weight", vec![d], vec![1.0; d]),
        ("blk.0.ffn_norm.bias", vec![d], vec![0.0; d]),
        (
            "blk.0.ffn_fc1.weight",
            vec![d, recipe.ffn_dim],
            (0..d * recipe.ffn_dim)
                .map(|i| 0.03 + ((i % 9) as f32 - 3.0) * 0.02)
                .collect(),
        ),
        (
            "blk.0.ffn_fc1.bias",
            vec![recipe.ffn_dim],
            vec![0.01; recipe.ffn_dim],
        ),
        (
            "blk.0.ffn_fc2.weight",
            vec![recipe.ffn_dim, d],
            (0..recipe.ffn_dim * d)
                .map(|i| 0.02 + ((i % 7) as f32 - 2.0) * 0.02)
                .collect(),
        ),
        ("blk.0.ffn_fc2.bias", vec![d], vec![0.0; d]),
    ]
}

fn make_ft_tensors(
    recipe: &FtTransformerRecipe,
    cat_vocab: usize,
) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
    let d = recipe.d_token;
    let cont_len = recipe.num_continuous * d;
    let mut tensors = vec![
        (
            "ft.cls_token.weight",
            vec![d],
            (0..d).map(|i| 0.1 + (i as f32) * 0.07).collect(),
        ),
        (
            "ft.cont_weight",
            vec![cont_len],
            (0..cont_len)
                .map(|i| 0.15 + ((i % 7) as f32) * 0.04)
                .collect(),
        ),
        (
            "ft.cont_bias",
            vec![cont_len],
            (0..cont_len).map(|i| ((i % 5) as f32) * 0.02).collect(),
        ),
        ("output_norm.weight", vec![d], vec![1.0; d]),
        ("output_norm.bias", vec![d], vec![0.0; d]),
        (
            "classifier.weight",
            vec![d, recipe.out_dim],
            (0..d * recipe.out_dim)
                .map(|i| {
                    let in_idx = i / recipe.out_dim;
                    let out_idx = i % recipe.out_dim;
                    let sign = if (in_idx + out_idx) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    sign * (0.1 + (in_idx as f32) * 0.05) + if out_idx == 0 { 0.15 } else { -0.1 }
                })
                .collect(),
        ),
        (
            "classifier.bias",
            vec![recipe.out_dim],
            vec![0.05; recipe.out_dim],
        ),
    ];
    tensors.extend(ft_layer0_tensors(recipe));
    if recipe.num_categorical > 0 {
        tensors.push((
            "ft.cat_embd.weight",
            vec![d, cat_vocab],
            (0..d * cat_vocab)
                .map(|i| {
                    let cat_id = i / d;
                    let dim = i % d;
                    if dim == (cat_id % d) {
                        1.2 + (cat_id as f32) * 0.3
                    } else {
                        (((dim + cat_id) % 3) as f32) * 0.15 - 0.1
                    }
                })
                .collect(),
        ));
        tensors.push((
            "ft.cat_bias",
            vec![recipe.num_categorical * d],
            vec![0.02; recipe.num_categorical * d],
        ));
    }
    tensors
}

fn build_ft_gguf(recipe: &FtTransformerRecipe, cat_vocab: usize) -> Vec<u8> {
    let tensors = make_ft_tensors(recipe, cat_vocab);
    let refs: Vec<(&str, &[usize], Vec<f32>)> = tensors
        .iter()
        .map(|(n, s, v)| (*n, s.as_slice(), v.clone()))
        .collect();
    let kvs = [
        ("ft-transformer.block_count", GgufValue::Uint32(1)),
        (
            "ft-transformer.embedding_length",
            GgufValue::Uint32(recipe.d_token as u32),
        ),
        (
            "ft-transformer.attention.head_count",
            GgufValue::Uint32(recipe.n_heads as u32),
        ),
        (
            "ft-transformer.feed_forward_length",
            GgufValue::Uint32(recipe.ffn_dim as u32),
        ),
        (
            "ft-transformer.num_continuous",
            GgufValue::Uint32(recipe.num_continuous as u32),
        ),
        (
            "ft-transformer.num_categorical",
            GgufValue::Uint32(recipe.num_categorical as u32),
        ),
        (
            "ft-transformer.out_dim",
            GgufValue::Uint32(recipe.out_dim as u32),
        ),
    ];
    build_gguf_f32("ft-transformer", &kvs, &refs)
}

fn ft_hf_alias_name(gguf_name: &str) -> &'static str {
    match gguf_name {
        "ft.cls_token.weight" => "feature_tokenizer.cls_token.weight",
        "ft.cont_weight" => "feature_tokenizer.num_tokenizer.weight",
        "ft.cont_bias" => "feature_tokenizer.num_tokenizer.bias",
        "ft.cat_embd.weight" => "feature_tokenizer.cat_tokenizer.weight",
        "ft.cat_bias" => "feature_tokenizer.cat_tokenizer.bias",
        "output_norm.weight" => "transformer.head.normalization.weight",
        "output_norm.bias" => "transformer.head.normalization.bias",
        "classifier.weight" | "ft.head.weight" => "transformer.head.linear.weight",
        "classifier.bias" | "ft.head.bias" => "transformer.head.linear.bias",
        "blk.0.attn_norm.weight" => "transformer.blocks.0.attention_normalization.weight",
        "blk.0.attn_norm.bias" => "transformer.blocks.0.attention_normalization.bias",
        "blk.0.attn_q.weight" => "transformer.blocks.0.attention.W_q.weight",
        "blk.0.attn_k.weight" => "transformer.blocks.0.attention.W_k.weight",
        "blk.0.attn_v.weight" => "transformer.blocks.0.attention.W_v.weight",
        "blk.0.attn_output.weight" => "transformer.blocks.0.attention.W_out.weight",
        "blk.0.ffn_norm.weight" => "transformer.blocks.0.ffn_normalization.weight",
        "blk.0.ffn_norm.bias" => "transformer.blocks.0.ffn_normalization.bias",
        "blk.0.ffn_fc1.weight" => "transformer.blocks.0.ffn.linear_first.weight",
        "blk.0.ffn_fc1.bias" => "transformer.blocks.0.ffn.linear_first.bias",
        "blk.0.ffn_fc2.weight" => "transformer.blocks.0.ffn.linear_second.weight",
        "blk.0.ffn_fc2.bias" => "transformer.blocks.0.ffn.linear_second.bias",
        other => panic!("unexpected ft tensor {other}"),
    }
}

fn make_tabpfn_tensors(recipe: &TabPfnV2Recipe) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
    let d = recipe.hidden_dim;
    let enc_len = recipe.num_features * d;
    vec![
        (
            "tabpfn.feature_enc.weight",
            vec![enc_len],
            (0..enc_len)
                .map(|i| 0.1 + ((i % 5) as f32) * 0.03)
                .collect(),
        ),
        (
            "tabpfn.feature_enc.bias",
            vec![enc_len],
            vec![0.02; enc_len],
        ),
        (
            "tabpfn.y_embd.weight",
            vec![d, recipe.num_classes],
            (0..d * recipe.num_classes)
                .map(|i| if i < d { 0.4 } else { -0.4 })
                .collect(),
        ),
        ("tabpfn.y_unknown.weight", vec![d], vec![0.0; d]),
        ("blk.0.feat_norm.weight", vec![d], vec![1.0; d]),
        ("blk.0.feat_norm.bias", vec![d], vec![0.0; d]),
        ("blk.0.feat_q.weight", vec![d, d], vec![0.08; d * d]),
        ("blk.0.feat_k.weight", vec![d, d], vec![0.07; d * d]),
        ("blk.0.feat_v.weight", vec![d, d], vec![0.09; d * d]),
        ("blk.0.feat_out.weight", vec![d, d], vec![0.05; d * d]),
        ("blk.0.sample_norm.weight", vec![d], vec![1.0; d]),
        ("blk.0.sample_norm.bias", vec![d], vec![0.0; d]),
        ("blk.0.sample_q.weight", vec![d, d], vec![0.1; d * d]),
        ("blk.0.sample_k.weight", vec![d, d], vec![0.09; d * d]),
        ("blk.0.sample_v.weight", vec![d, d], vec![0.11; d * d]),
        ("blk.0.sample_out.weight", vec![d, d], vec![0.07; d * d]),
        ("blk.0.ffn_norm.weight", vec![d], vec![1.0; d]),
        ("blk.0.ffn_norm.bias", vec![d], vec![0.0; d]),
        (
            "blk.0.ffn_fc1.weight",
            vec![d, recipe.ffn_dim],
            vec![0.05; d * recipe.ffn_dim],
        ),
        (
            "blk.0.ffn_fc1.bias",
            vec![recipe.ffn_dim],
            vec![0.01; recipe.ffn_dim],
        ),
        (
            "blk.0.ffn_fc2.weight",
            vec![recipe.ffn_dim, d],
            vec![0.04; recipe.ffn_dim * d],
        ),
        ("blk.0.ffn_fc2.bias", vec![d], vec![0.0; d]),
        ("output_norm.weight", vec![d], vec![1.0; d]),
        ("output_norm.bias", vec![d], vec![0.0; d]),
        (
            "classifier.weight",
            vec![d, recipe.num_classes],
            (0..d * recipe.num_classes)
                .map(|i| if i < d { 0.2 } else { -0.25 })
                .collect(),
        ),
        (
            "classifier.bias",
            vec![recipe.num_classes],
            vec![0.01; recipe.num_classes],
        ),
    ]
}

fn build_tabpfn_gguf(recipe: &TabPfnV2Recipe) -> Vec<u8> {
    let tensors = make_tabpfn_tensors(recipe);
    let refs: Vec<(&str, &[usize], Vec<f32>)> = tensors
        .iter()
        .map(|(n, s, v)| (*n, s.as_slice(), v.clone()))
        .collect();
    let kvs = [
        ("tabpfn-v2.block_count", GgufValue::Uint32(1)),
        (
            "tabpfn-v2.embedding_length",
            GgufValue::Uint32(recipe.hidden_dim as u32),
        ),
        (
            "tabpfn-v2.attention.head_count",
            GgufValue::Uint32(recipe.n_heads as u32),
        ),
        (
            "tabpfn-v2.feed_forward_length",
            GgufValue::Uint32(recipe.ffn_dim as u32),
        ),
        (
            "tabpfn-v2.num_features",
            GgufValue::Uint32(recipe.num_features as u32),
        ),
        (
            "tabpfn-v2.num_classes",
            GgufValue::Uint32(recipe.num_classes as u32),
        ),
    ];
    build_gguf_f32("tabpfn-v2", &kvs, &refs)
}

fn tabpfn_hf_alias_name(gguf_name: &str) -> &'static str {
    match gguf_name {
        "tabpfn.feature_enc.weight" => "encoder.weight",
        "tabpfn.feature_enc.bias" => "encoder.bias",
        "tabpfn.y_embd.weight" => "y_encoder.weight",
        "tabpfn.y_unknown.weight" => "y_unknown.weight",
        "blk.0.feat_norm.weight" => {
            "transformer_encoder.layers.0.self_attn_between_features_norm.weight"
        }
        "blk.0.feat_norm.bias" => {
            "transformer_encoder.layers.0.self_attn_between_features_norm.bias"
        }
        "blk.0.feat_q.weight" => {
            "transformer_encoder.layers.0.self_attn_between_features.q_proj.weight"
        }
        "blk.0.feat_k.weight" => {
            "transformer_encoder.layers.0.self_attn_between_features.k_proj.weight"
        }
        "blk.0.feat_v.weight" => {
            "transformer_encoder.layers.0.self_attn_between_features.v_proj.weight"
        }
        "blk.0.feat_out.weight" => {
            "transformer_encoder.layers.0.self_attn_between_features.out_proj.weight"
        }
        "blk.0.sample_norm.weight" => {
            "transformer_encoder.layers.0.self_attn_between_items_norm.weight"
        }
        "blk.0.sample_norm.bias" => {
            "transformer_encoder.layers.0.self_attn_between_items_norm.bias"
        }
        "blk.0.sample_q.weight" => {
            "transformer_encoder.layers.0.self_attn_between_items.q_proj.weight"
        }
        "blk.0.sample_k.weight" => {
            "transformer_encoder.layers.0.self_attn_between_items.k_proj.weight"
        }
        "blk.0.sample_v.weight" => {
            "transformer_encoder.layers.0.self_attn_between_items.v_proj.weight"
        }
        "blk.0.sample_out.weight" => {
            "transformer_encoder.layers.0.self_attn_between_items.out_proj.weight"
        }
        "blk.0.ffn_norm.weight" => "transformer_encoder.layers.0.mlp_norm.weight",
        "blk.0.ffn_norm.bias" => "transformer_encoder.layers.0.mlp_norm.bias",
        "blk.0.ffn_fc1.weight" => "transformer_encoder.layers.0.mlp.linear1.weight",
        "blk.0.ffn_fc1.bias" => "transformer_encoder.layers.0.mlp.linear1.bias",
        "blk.0.ffn_fc2.weight" => "transformer_encoder.layers.0.mlp.linear2.weight",
        "blk.0.ffn_fc2.bias" => "transformer_encoder.layers.0.mlp.linear2.bias",
        "output_norm.weight" => "decoder_norm.weight",
        "output_norm.bias" => "decoder_norm.bias",
        "classifier.weight" => "decoder_dict.standard.2.weight",
        "classifier.bias" => "decoder_dict.standard.2.bias",
        other => panic!("unexpected tabpfn tensor {other}"),
    }
}

fn dt_embedding_and_head_tensors(
    recipe: &DecisionTransformerRecipe,
) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
    let h = recipe.hidden_dim;
    vec![
        ("dt.embed_return.weight", vec![1, h], vec![0.2_f32; h]),
        (
            "dt.embed_state.weight",
            vec![recipe.state_dim, h],
            vec![0.15_f32; recipe.state_dim * h],
        ),
        (
            "dt.embed_action.weight",
            vec![h, recipe.act_dim],
            (0..h * recipe.act_dim)
                .map(|i| 0.05 + f32::from(u16::try_from(i / h).unwrap()) * 0.1)
                .collect(),
        ),
        (
            "dt.embed_timestep.weight",
            vec![h, recipe.max_ep_len],
            (0..h * recipe.max_ep_len)
                .map(|i| f32::from(u16::try_from(i / h).unwrap()) * 0.01)
                .collect(),
        ),
        ("output_norm.weight", vec![h], vec![1.0; h]),
        (
            "dt.predict_action.weight",
            vec![h, recipe.act_dim],
            (0..h * recipe.act_dim)
                .map(|i| if (i / h) % 2 == 0 { 0.25 } else { -0.15 })
                .collect(),
        ),
        (
            "dt.predict_action.bias",
            vec![recipe.act_dim],
            vec![0.02; recipe.act_dim],
        ),
    ]
}

fn dt_layer0_tensors(
    recipe: &DecisionTransformerRecipe,
) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
    let h = recipe.hidden_dim;
    let q_dim = recipe.n_heads_q * recipe.head_dim;
    let kv_dim = recipe.n_heads_kv * recipe.head_dim;
    vec![
        ("blk.0.attn_norm.weight", vec![h], vec![1.0; h]),
        ("blk.0.attn_q.weight", vec![h, q_dim], vec![0.08; h * q_dim]),
        (
            "blk.0.attn_k.weight",
            vec![h, kv_dim],
            vec![0.07; h * kv_dim],
        ),
        (
            "blk.0.attn_v.weight",
            vec![h, kv_dim],
            vec![0.09; h * kv_dim],
        ),
        (
            "blk.0.attn_output.weight",
            vec![q_dim, h],
            vec![0.06; q_dim * h],
        ),
        ("blk.0.ffn_norm.weight", vec![h], vec![1.0; h]),
        (
            "blk.0.ffn_gate.weight",
            vec![h, recipe.ffn_dim],
            vec![0.05; h * recipe.ffn_dim],
        ),
        (
            "blk.0.ffn_up.weight",
            vec![h, recipe.ffn_dim],
            vec![0.05; h * recipe.ffn_dim],
        ),
        (
            "blk.0.ffn_down.weight",
            vec![recipe.ffn_dim, h],
            vec![0.04; recipe.ffn_dim * h],
        ),
    ]
}

fn make_dt_tensors(
    recipe: &DecisionTransformerRecipe,
) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
    let mut tensors = dt_embedding_and_head_tensors(recipe);
    tensors.extend(dt_layer0_tensors(recipe));
    tensors
}

fn build_dt_gguf(recipe: &DecisionTransformerRecipe) -> Vec<u8> {
    let tensors = make_dt_tensors(recipe);
    let refs: Vec<(&str, &[usize], Vec<f32>)> = tensors
        .iter()
        .map(|(n, s, v)| (*n, s.as_slice(), v.clone()))
        .collect();
    let kvs = [
        ("decision-transformer.block_count", GgufValue::Uint32(1)),
        (
            "decision-transformer.embedding_length",
            GgufValue::Uint32(u32::try_from(recipe.hidden_dim).unwrap()),
        ),
        (
            "decision-transformer.attention.head_count",
            GgufValue::Uint32(u32::try_from(recipe.n_heads_q).unwrap()),
        ),
        (
            "decision-transformer.attention.head_count_kv",
            GgufValue::Uint32(u32::try_from(recipe.n_heads_kv).unwrap()),
        ),
        (
            "decision-transformer.feed_forward_length",
            GgufValue::Uint32(u32::try_from(recipe.ffn_dim).unwrap()),
        ),
        (
            "decision-transformer.state_dim",
            GgufValue::Uint32(u32::try_from(recipe.state_dim).unwrap()),
        ),
        (
            "decision-transformer.act_dim",
            GgufValue::Uint32(u32::try_from(recipe.act_dim).unwrap()),
        ),
        (
            "decision-transformer.max_ep_len",
            GgufValue::Uint32(u32::try_from(recipe.max_ep_len).unwrap()),
        ),
    ];
    build_gguf_f32("decision-transformer", &kvs, &refs)
}

fn dt_hf_alias_name(gguf_name: &str) -> &'static str {
    match gguf_name {
        "dt.embed_return.weight" => "embed_return.weight",
        "dt.embed_state.weight" => "embed_state.weight",
        "dt.embed_action.weight" => "embed_action.weight",
        "dt.embed_timestep.weight" => "embed_timestep.weight",
        "output_norm.weight" => "encoder.ln_f.weight",
        "dt.predict_action.weight" => "predict_action.weight",
        "dt.predict_action.bias" => "predict_action.bias",
        "blk.0.attn_norm.weight" => "model.layers.0.input_layernorm.weight",
        "blk.0.attn_q.weight" => "model.layers.0.self_attn.q_proj.weight",
        "blk.0.attn_k.weight" => "model.layers.0.self_attn.k_proj.weight",
        "blk.0.attn_v.weight" => "model.layers.0.self_attn.v_proj.weight",
        "blk.0.attn_output.weight" => "model.layers.0.self_attn.o_proj.weight",
        "blk.0.ffn_norm.weight" => "model.layers.0.post_attention_layernorm.weight",
        "blk.0.ffn_gate.weight" => "model.layers.0.mlp.gate_proj.weight",
        "blk.0.ffn_up.weight" => "model.layers.0.mlp.up_proj.weight",
        "blk.0.ffn_down.weight" => "model.layers.0.mlp.down_proj.weight",
        other => panic!("unexpected dt tensor {other}"),
    }
}

/// Converts a GGUF 2D matrix shape `[in_dim, out_dim]` to `PyTorch` `SafeTensors` `[out_dim, in_dim]`,
/// preserving exact flat row-major storage.
fn to_hf_2d_shape(shape: &[usize]) -> Vec<usize> {
    if shape.len() == 2 {
        vec![shape[1], shape[0]]
    } else {
        shape.to_vec()
    }
}

#[test]
fn crate_metadata_matches_policy() {
    assert_eq!(CRATE_NAME, "jero-arch-machine");
    assert_eq!(
        CRATE_LICENSE,
        "LGPL-3.0-only WITH LGPL-3.0-linking-exception"
    );
    assert_ne!(crate_description(), "");
    assert_eq!(jero_core::DEFAULT_ALIGNMENT, 32);
}

#[test]
fn test_ttm_recipe_telemetry_forecast_and_safetensors_parity() {
    let recipe = TtmRecipe {
        context_length: 16,
        patch_length: 4,
        forecast_length: 6,
        num_channels: 3,
        num_mixer_layers: 1,
        time_hidden_dim: 8,
        channel_hidden_dim: 6,
        norm_eps: 1e-5,
    };
    assert_eq!(recipe.architecture_id(), "ttm");
    assert!(!recipe.requires_kv_cache());
    assert!(recipe.supports_quant(QuantType::F32));
    assert!(recipe.supports_quant(QuantType::F16));
    assert!(recipe.supports_quant(QuantType::BF16));

    let tensor_data = make_ttm_tensors(&recipe);
    let gguf_refs: Vec<(&str, &[usize], Vec<f32>)> = tensor_data
        .iter()
        .map(|(n, s, v)| (*n, s.as_slice(), v.clone()))
        .collect();
    let kvs = [
        ("ttm.context_length", GgufValue::Uint32(16)),
        ("ttm.block_count", GgufValue::Uint32(1)),
        ("ttm.patch_length", GgufValue::Uint32(4)),
        ("ttm.forecast_length", GgufValue::Uint32(6)),
        ("ttm.num_channels", GgufValue::Uint32(3)),
        ("ttm.time_hidden_dim", GgufValue::Uint32(8)),
        ("ttm.channel_hidden_dim", GgufValue::Uint32(6)),
    ];
    let gguf_bytes = build_gguf_f32("ttm", &kvs, &gguf_refs);
    let gguf = GgufFile::parse(&gguf_bytes).expect("parse ttm gguf");
    let hydrated_gguf = TtmRecipe::from_gguf(&gguf);
    assert_eq!(hydrated_gguf, recipe);

    // Build SafeTensors with HF tensor aliases and PyTorch [out_dim, in_dim] shapes.
    let st_shapes: Vec<(&'static str, Vec<usize>, Vec<f32>)> = tensor_data
        .iter()
        .map(|(n, s, v)| (ttm_hf_alias_name(n), to_hf_2d_shape(s), v.clone()))
        .collect();
    let st_refs: Vec<(&str, &[usize], Vec<f32>)> = st_shapes
        .iter()
        .map(|(n, s, v)| (*n, s.as_slice(), v.clone()))
        .collect();
    let st_bytes = build_safetensors_f32(&[("general.architecture", "ttm")], &st_refs);
    let st = SafeTensorsFile::from_bytes(&st_bytes).expect("parse ttm safetensors");
    let hydrated_st = TtmRecipe::from_safetensors(&st);
    assert_eq!(hydrated_st, recipe);

    hydrated_gguf
        .validate_model(gguf.metadata(), &gguf)
        .expect("gguf validate_model");
    hydrated_st
        .validate_model(st.metadata(), &st)
        .expect("safetensors validate_model");

    let raw_series: Vec<f32> = (0..16 * 3).map(|i| (i as f32) * 0.05).collect();
    let mut series = vec![0.0_f32; raw_series.len()];
    sanitize_and_normalize_features(
        &mut series,
        &raw_series,
        &[0.5, 0.5, 0.5],
        &[1.0, 1.0, 1.0],
        5.0,
    )
    .expect("sanitize features");

    let mut logits_gguf = vec![0.0_f32; 6 * 3];
    let mut logits_st = vec![0.0_f32; 6 * 3];
    let mut kv = [];

    let mut ctx_gguf =
        ExecutionContext::new_feature_input(&series, 16, 3, 64, &mut logits_gguf, &mut kv).unwrap();
    hydrated_gguf
        .forward_classify(&mut ctx_gguf, &gguf)
        .expect("ttm gguf forecast");

    let mut ctx_st =
        ExecutionContext::new_feature_input(&series, 16, 3, 64, &mut logits_st, &mut kv).unwrap();
    hydrated_st
        .forward_classify(&mut ctx_st, &st)
        .expect("ttm safetensors forecast");
    assert!(hydrated_st.decode_step(&mut ctx_st, &st).is_err());

    assert!(logits_gguf.iter().any(|v| v.abs() > 1e-4));
    assert_f32_slice_parity(&logits_gguf, &logits_st, "ttm_gguf_vs_safetensors");
}

#[test]
fn test_ttm_zero_shot_forecast_parity_and_fail_closed() {
    let recipe = TtmRecipe {
        context_length: 8,
        patch_length: 2,
        forecast_length: 4,
        num_channels: 2,
        num_mixer_layers: 1,
        time_hidden_dim: 6,
        channel_hidden_dim: 4,
        norm_eps: 1e-5,
    };
    let tensor_data = make_ttm_tensors(&recipe);
    let tensor_refs: Vec<(&str, &[usize], Vec<f32>)> = tensor_data
        .iter()
        .map(|(n, s, v)| (*n, s.as_slice(), v.clone()))
        .collect();
    let kvs = [("ttm.block_count", GgufValue::Uint32(1))];
    let bytes = build_gguf_f32("ttm", &kvs, &tensor_refs);
    let gguf = GgufFile::parse(&bytes).expect("parse ttm gguf");
    assert_eq!(TtmRecipe::from_gguf(&gguf), recipe);

    let series_a: Vec<f32> = (0..8 * 2).map(|i| 0.2 + (i as f32) * 0.04).collect();
    let series_b: Vec<f32> = (0..8 * 2).map(|i| 1.5 - (i as f32) * 0.06).collect();
    let mut out_a = vec![0.0_f32; 8];
    let mut out_b = vec![0.0_f32; 8];
    let mut kv = [];

    let mut ctx_a =
        ExecutionContext::new_feature_input(&series_a, 8, 2, 64, &mut out_a, &mut kv).unwrap();
    recipe.forward_classify(&mut ctx_a, &gguf).unwrap();

    let mut ctx_b =
        ExecutionContext::new_feature_input(&series_b, 8, 2, 64, &mut out_b, &mut kv).unwrap();
    recipe.forward_classify(&mut ctx_b, &gguf).unwrap();
    assert_ne!(out_a, out_b);

    // NaN rejection at context boundary AND fail-closed empty SafeTensors/GGUF rejection
    let mut bad_series = series_a;
    bad_series[3] = f32::NAN;
    let mut bad_out = vec![0.0_f32; 8];
    assert!(matches!(
        ExecutionContext::new_feature_input(&bad_series, 8, 2, 64, &mut bad_out, &mut kv),
        Err(K3Error::TensorBoundsError { .. })
    ));

    let empty_gguf_bytes = create_synthetic_gguf_v3_with_kvs("ttm", 32, &[], &[]);
    let empty_gguf = GgufFile::parse(&empty_gguf_bytes).unwrap();
    let mut ctx_miss =
        ExecutionContext::new_feature_input(&[1.0_f32; 16], 8, 2, 64, &mut bad_out, &mut kv)
            .unwrap();
    assert!(matches!(
        recipe.forward_classify(&mut ctx_miss, &empty_gguf),
        Err(K3Error::TensorBoundsError { .. })
    ));

    let empty_st_bytes = create_synthetic_safetensors(&[("general.architecture", "ttm")], &[]);
    let empty_st = SafeTensorsFile::from_bytes(&empty_st_bytes).unwrap();
    assert!(matches!(
        recipe.forward_classify(&mut ctx_miss, &empty_st),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // Missing ttm.patch_proj.weight fails closed instead of synthesizing identity
    let no_patch_refs: Vec<(&str, &[usize], Vec<f32>)> = tensor_refs
        .iter()
        .filter(|(n, _, _)| *n != "ttm.patch_proj.weight")
        .map(|(n, s, v)| (*n, *s, v.clone()))
        .collect();
    let no_patch_st_bytes =
        build_safetensors_f32(&[("general.architecture", "ttm")], &no_patch_refs);
    let no_patch_st = SafeTensorsFile::from_bytes(&no_patch_st_bytes).unwrap();
    assert!(matches!(
        recipe.forward_classify(&mut ctx_miss, &no_patch_st),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

#[test]
fn test_ft_transformer_recipe_forward_and_safetensors_parity() {
    let recipe = FtTransformerRecipe {
        num_continuous: 3,
        num_categorical: 0,
        d_token: 8,
        num_layers: 1,
        n_heads: 2,
        ffn_dim: 16,
        out_dim: 2,
        norm_eps: 1e-5,
    };
    let gguf_bytes = build_ft_gguf(&recipe, 0);
    let gguf = GgufFile::parse(&gguf_bytes).unwrap();
    assert_eq!(FtTransformerRecipe::from_gguf(&gguf), recipe);
    assert!(!recipe.requires_kv_cache());

    // Build SafeTensors with HF `rtdl` tensor names & PyTorch [out_dim, in_dim] linear weights
    let base_tensors = make_ft_tensors(&recipe, 0);
    let st_shapes: Vec<(&'static str, Vec<usize>, Vec<f32>)> = base_tensors
        .iter()
        .map(|(n, s, v)| (ft_hf_alias_name(n), to_hf_2d_shape(s), v.clone()))
        .collect();
    let st_refs: Vec<(&str, &[usize], Vec<f32>)> = st_shapes
        .iter()
        .map(|(n, s, v)| (*n, s.as_slice(), v.clone()))
        .collect();
    let st_bytes = build_safetensors_f32(
        &[
            ("general.architecture", "ft-transformer"),
            ("n_heads", "2"),
            ("num_continuous", "3"),
            ("num_categorical", "0"),
        ],
        &st_refs,
    );
    let st = SafeTensorsFile::from_bytes(&st_bytes).unwrap();
    let st_recipe = FtTransformerRecipe::from_safetensors(&st);
    assert_eq!(st_recipe, recipe);

    let mut logits_gguf = [0.0_f32; 2];
    let mut logits_st = [0.0_f32; 2];
    let mut kv = [];
    let row = [1.2_f32, -0.4, 0.8];

    let mut ctx_g =
        ExecutionContext::new_feature_input(&row, 1, 3, 64, &mut logits_gguf, &mut kv).unwrap();
    recipe.forward_classify(&mut ctx_g, &gguf).unwrap();

    let mut ctx_s =
        ExecutionContext::new_feature_input(&row, 1, 3, 64, &mut logits_st, &mut kv).unwrap();
    st_recipe.forward_classify(&mut ctx_s, &st).unwrap();

    assert_ne!(logits_gguf[0], logits_gguf[1]);
    assert_f32_slice_parity(&logits_gguf, &logits_st, "ft_gguf_vs_safetensors");
}

#[test]
fn test_ft_transformer_continuous_categorical_and_negative_checks() {
    let recipe = FtTransformerRecipe {
        num_continuous: 2,
        num_categorical: 2,
        d_token: 8,
        num_layers: 1,
        n_heads: 2,
        ffn_dim: 16,
        out_dim: 2,
        norm_eps: 1e-5,
    };
    let bytes = build_ft_gguf(&recipe, 4);
    let gguf = GgufFile::parse(&bytes).unwrap();
    assert_eq!(FtTransformerRecipe::from_gguf(&gguf), recipe);
    let mut kv = [];

    let row_cat0 = [0.5_f32, -0.3, 0.0, 1.0];
    let row_cat3 = [0.5_f32, -0.3, 3.0, 2.0];
    let mut out_a = [0.0_f32; 2];
    let mut out_b = [0.0_f32; 2];
    {
        let mut ctx_a =
            ExecutionContext::new_feature_input(&row_cat0, 1, 4, 64, &mut out_a, &mut kv).unwrap();
        recipe.forward_classify(&mut ctx_a, &gguf).unwrap();
    }
    {
        let mut ctx_b =
            ExecutionContext::new_feature_input(&row_cat3, 1, 4, 64, &mut out_b, &mut kv).unwrap();
        recipe.forward_classify(&mut ctx_b, &gguf).unwrap();
    }
    assert_ne!(out_a, out_b);

    // Negative: fractional categorical ID fails closed
    let bad_cat = [0.5_f32, -0.3, 1.5, 2.0];
    {
        let mut ctx_bad_cat =
            ExecutionContext::new_feature_input(&bad_cat, 1, 4, 64, &mut out_a, &mut kv).unwrap();
        assert!(matches!(
            recipe.forward_classify(&mut ctx_bad_cat, &gguf),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    // Negative: missing ft.head.weight / classifier.weight fails closed
    let empty_st_bytes = create_synthetic_safetensors(&[], &[]);
    let empty_st = SafeTensorsFile::from_bytes(&empty_st_bytes).unwrap();
    {
        let mut ctx_empty =
            ExecutionContext::new_feature_input(&row_cat0, 1, 4, 64, &mut out_a, &mut kv).unwrap();
        assert!(matches!(
            recipe.forward_classify(&mut ctx_empty, &empty_st),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    // Negative: missing ft.head.bias when ft.head.weight is present fails closed
    let partial_st = build_safetensors_f32(
        &[("general.architecture", "ft-transformer")],
        &[
            ("ft.cls_token.weight", &[8], vec![0.1; 8]),
            ("ft.cont_weight", &[16], vec![0.1; 16]),
            ("ft.cont_bias", &[16], vec![0.0; 16]),
            ("output_norm.weight", &[8], vec![1.0; 8]),
            ("output_norm.bias", &[8], vec![0.0; 8]),
            ("ft.head.weight", &[8, 2], vec![0.1; 16]),
        ],
    );
    let partial = SafeTensorsFile::from_bytes(&partial_st).unwrap();
    let small_ft = FtTransformerRecipe {
        num_continuous: 2,
        num_categorical: 0,
        ..recipe
    };
    let mut ctx_small =
        ExecutionContext::new_feature_input(&[0.5_f32, -0.3], 1, 2, 64, &mut out_a, &mut kv)
            .unwrap();
    assert!(matches!(
        small_ft.forward_classify(&mut ctx_small, &partial),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

#[test]
fn test_tabpfn_v2_dual_axis_attention() {
    let shape = TabPfnAttentionShape::new(4, 3, 2, 4).with_context_samples(2);
    let total = shape.total_elements().unwrap();
    let q: Vec<f32> = (0..total).map(|i| 0.1 + ((i % 7) as f32) * 0.05).collect();
    let k: Vec<f32> = (0..total).map(|i| 0.2 - ((i % 5) as f32) * 0.04).collect();
    let mut v_a: Vec<f32> = (0..total).map(|i| ((i % 9) as f32) * 0.1).collect();
    let mut v_b = v_a.clone();
    // Corrupt ONLY query row 3 (index >= n_ctx = 2) values in V
    for slot in &mut v_b[3 * 3 * 8..] {
        *slot = 999.0;
    }
    let mut out_a = vec![0.0_f32; total];
    let mut out_b = vec![0.0_f32; total];
    tabpfn_dual_axis_attention_f32(
        &mut out_a,
        &q,
        &k,
        &v_a,
        shape,
        TabPfnAttentionAxis::Sample,
        0.5,
    )
    .unwrap();
    tabpfn_dual_axis_attention_f32(
        &mut out_b,
        &q,
        &k,
        &v_b,
        shape,
        TabPfnAttentionAxis::Sample,
        0.5,
    )
    .unwrap();
    #[cfg(not(miri))]
    assert_eq!(
        out_a, out_b,
        "query V entries must never leak into sample attention"
    );
    for (&a, &b) in out_a.iter().zip(out_b.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "query V entries leaked into sample attention: {a} vs {b}"
        );
    }
    v_a[0] += 0.5;
    let mut out_feat = vec![0.0_f32; total];
    tabpfn_dual_axis_attention_f32(
        &mut out_feat,
        &q,
        &k,
        &v_a,
        shape,
        TabPfnAttentionAxis::Feature,
        0.5,
    )
    .unwrap();
    assert!(out_feat.iter().all(|v| v.is_finite()));
}

#[test]
fn test_tabpfn_v2_in_context_and_safetensors_parity() {
    let recipe = TabPfnV2Recipe {
        num_features: 3,
        hidden_dim: 8,
        num_layers: 1,
        n_heads: 2,
        ffn_dim: 16,
        num_classes: 2,
        norm_eps: 1e-5,
    };
    let gguf_bytes = build_tabpfn_gguf(&recipe);
    let gguf = GgufFile::parse(&gguf_bytes).unwrap();
    assert_eq!(TabPfnV2Recipe::from_gguf(&gguf), recipe);
    assert!(!recipe.requires_kv_cache());

    // SafeTensors with HuggingFace TabPFN names and PyTorch [out_dim, in_dim] matrices
    let base_tensors = make_tabpfn_tensors(&recipe);
    let st_shapes: Vec<(&'static str, Vec<usize>, Vec<f32>)> = base_tensors
        .iter()
        .map(|(n, s, v)| (tabpfn_hf_alias_name(n), to_hf_2d_shape(s), v.clone()))
        .collect();
    let st_refs: Vec<(&str, &[usize], Vec<f32>)> = st_shapes
        .iter()
        .map(|(n, s, v)| (*n, s.as_slice(), v.clone()))
        .collect();
    let st_bytes = build_safetensors_f32(
        &[
            ("general.architecture", "tabpfn-v2"),
            ("nhead", "2"),
            ("num_features", "3"),
        ],
        &st_refs,
    );
    let st = SafeTensorsFile::from_bytes(&st_bytes).unwrap();
    let st_recipe = TabPfnV2Recipe::from_safetensors(&st);
    assert_eq!(st_recipe, recipe);

    let rows_a = [
        0.1_f32, 0.2, 0.3, // ctx 0 (class 0)
        0.9, 0.8, 0.7, // ctx 1 (class 1)
        0.15, 0.25, 0.35, // query 0
        0.85, 0.75, 0.65, // query 1
    ];
    let mut rows_b = rows_a;
    rows_b[9] = 42.0;

    let mut logits_a = [0.0_f32; 4];
    let mut logits_b = [0.0_f32; 4];
    let mut logits_st = [0.0_f32; 4];
    let mut kv = [];
    let mut ctx_a = ExecutionContext::new(&[0_u32, 1], 64, &mut logits_a, &mut kv)
        .unwrap()
        .with_feature_input(&rows_a, 4, 3)
        .unwrap();
    recipe.forward_classify(&mut ctx_a, &gguf).unwrap();

    let mut ctx_b = ExecutionContext::new(&[0_u32, 1], 64, &mut logits_b, &mut kv)
        .unwrap()
        .with_feature_input(&rows_b, 4, 3)
        .unwrap();
    recipe.forward_classify(&mut ctx_b, &gguf).unwrap();

    let mut ctx_st = ExecutionContext::new(&[0_u32, 1], 64, &mut logits_st, &mut kv)
        .unwrap()
        .with_feature_input(&rows_a, 4, 3)
        .unwrap();
    st_recipe.forward_classify(&mut ctx_st, &st).unwrap();

    // Negative fail-closed checks: empty SafeTensors & out-of-bounds context label
    let empty_st_bytes = create_synthetic_safetensors(&[], &[]);
    let empty_st = SafeTensorsFile::from_bytes(&empty_st_bytes).unwrap();
    assert!(matches!(
        recipe.forward_classify(&mut ctx_st, &empty_st),
        Err(K3Error::TensorBoundsError { .. })
    ));

    assert_f32_slice_parity(&logits_a[0..2], &logits_b[0..2], "tabpfn_no_query_leak");
    assert_f32_slice_parity(&logits_a, &logits_st, "tabpfn_gguf_vs_safetensors");

    let mut ctx_bad_lbl = ExecutionContext::new(&[0_u32, 9], 64, &mut logits_st, &mut kv)
        .unwrap()
        .with_feature_input(&rows_a, 4, 3)
        .unwrap();
    assert!(matches!(
        recipe.forward_classify(&mut ctx_bad_lbl, &st),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

#[test]
fn test_decision_transformer_rtg_action_autoregression_and_safetensors_parity() {
    let recipe = DecisionTransformerRecipe {
        state_dim: 3,
        act_dim: 4,
        hidden_dim: 8,
        num_layers: 1,
        n_heads_q: 2,
        n_heads_kv: 2,
        head_dim: 4,
        ffn_dim: 16,
        max_ep_len: 16,
        rms_eps: 1e-6,
        rope_theta: 10_000.0,
    };
    let gguf_bytes = build_dt_gguf(&recipe);
    let gguf = GgufFile::parse(&gguf_bytes).unwrap();
    assert_eq!(DecisionTransformerRecipe::from_gguf(&gguf), recipe);
    assert!(recipe.requires_kv_cache());

    // SafeTensors with HF tensor names & PyTorch [out_dim, in_dim] weights
    let base_tensors = make_dt_tensors(&recipe);
    let st_shapes: Vec<(&'static str, Vec<usize>, Vec<f32>)> = base_tensors
        .iter()
        .map(|(n, s, v)| (dt_hf_alias_name(n), to_hf_2d_shape(s), v.clone()))
        .collect();
    let st_refs: Vec<(&str, &[usize], Vec<f32>)> = st_shapes
        .iter()
        .map(|(n, s, v)| (*n, s.as_slice(), v.clone()))
        .collect();
    let st_bytes = build_safetensors_f32(
        &[
            ("general.architecture", "decision-transformer"),
            ("n_head", "2"),
            ("num_key_value_heads", "2"),
            ("rms_norm_eps", "1e-6"),
            ("rope_theta", "10000.0"),
        ],
        &st_refs,
    );
    let st = SafeTensorsFile::from_bytes(&st_bytes).unwrap();
    let st_recipe = DecisionTransformerRecipe::from_safetensors(&st);
    assert_eq!(st_recipe, recipe);

    let traj = [10.0_f32, 0.1, 0.2, 0.3, 8.0, 0.4, 0.5, 0.6];
    let mut logits_gguf = [0.0_f32; 4];
    let mut logits_st = [0.0_f32; 4];
    let mut kv_gguf = vec![0.0_f32; 64 * 2 * 8 * 2];
    let mut kv_st = vec![0.0_f32; 64 * 2 * 8 * 2];

    let mut ctx_g = ExecutionContext::new(&[1_u32], 64, &mut logits_gguf, &mut kv_gguf)
        .unwrap()
        .with_feature_input(&traj, 2, 4)
        .unwrap();
    recipe.prefill_step(&mut ctx_g, &gguf).expect("dt prefill");
    assert_eq!(ctx_g.position(), 5);

    let mut ctx_s = ExecutionContext::new(&[1_u32], 64, &mut logits_st, &mut kv_st)
        .unwrap()
        .with_feature_input(&traj, 2, 4)
        .unwrap();
    st_recipe
        .prefill_step(&mut ctx_s, &st)
        .expect("dt st prefill");
    assert_eq!(ctx_s.position(), 5);
    assert_f32_slice_parity(
        ctx_g.logits(),
        ctx_s.logits(),
        "dt_prefill_gguf_vs_safetensors",
    );

    ctx_g.prepare_decode(2).unwrap();
    recipe.decode_step(&mut ctx_g, &gguf).expect("dt decode");
    assert_eq!(ctx_g.position(), 6);

    ctx_s.prepare_decode(2).unwrap();
    st_recipe
        .decode_step(&mut ctx_s, &st)
        .expect("dt st decode");
    assert_eq!(ctx_s.position(), 6);
    assert_f32_slice_parity(&logits_gguf, &logits_st, "dt_decode_gguf_vs_safetensors");
}

#[test]
fn test_decision_transformer_rejects_invalid_hyperparams_missing_weights_and_empty_kv() {
    let small_ffn = DecisionTransformerRecipe {
        state_dim: 3,
        act_dim: 4,
        hidden_dim: 8,
        num_layers: 1,
        n_heads_q: 2,
        n_heads_kv: 2,
        head_dim: 4,
        ffn_dim: 3,
        max_ep_len: 16,
        rms_eps: 1e-6,
        rope_theta: 10_000.0,
    };
    let bytes = build_dt_gguf(&small_ffn);
    let gguf = GgufFile::parse(&bytes).unwrap();
    let traj = [10.0_f32, 0.1, 0.2, 0.3];
    let mut logits = [0.0_f32; 4];
    let mut kv_cache = vec![0.0_f32; 64 * 2 * 8 * 2];
    {
        let mut ctx = ExecutionContext::new(&[1_u32], 64, &mut logits, &mut kv_cache)
            .unwrap()
            .with_feature_input(&traj, 1, 4)
            .unwrap();
        assert!(matches!(
            small_ffn.prefill_step(&mut ctx, &gguf),
            Err(K3Error::CapabilityMismatch { .. })
        ));
    }

    let bad_gqa = DecisionTransformerRecipe {
        n_heads_q: 3,
        n_heads_kv: 2,
        ffn_dim: 16,
        ..small_ffn
    };
    let bad_bytes = build_dt_gguf(&bad_gqa);
    let bad_gguf = GgufFile::parse(&bad_bytes).unwrap();
    {
        let mut ctx_gqa = ExecutionContext::new(&[1_u32], 64, &mut logits, &mut kv_cache)
            .unwrap()
            .with_feature_input(&traj, 1, 4)
            .unwrap();
        assert!(matches!(
            bad_gqa.prefill_step(&mut ctx_gqa, &bad_gguf),
            Err(K3Error::CapabilityMismatch { .. })
        ));
    }

    // Empty SafeTensors & empty KV cache must fail closed (ADR-045)
    let valid_dt = DecisionTransformerRecipe {
        ffn_dim: 16,
        ..small_ffn
    };
    let empty_st_bytes = create_synthetic_safetensors(&[], &[]);
    let empty_st = SafeTensorsFile::from_bytes(&empty_st_bytes).unwrap();
    {
        let mut ctx = ExecutionContext::new(&[1_u32], 64, &mut logits, &mut kv_cache)
            .unwrap()
            .with_feature_input(&traj, 1, 4)
            .unwrap();
        assert!(matches!(
            valid_dt.prefill_step(&mut ctx, &empty_st),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    let ok_bytes = build_dt_gguf(&valid_dt);
    let ok_gguf = GgufFile::parse(&ok_bytes).unwrap();
    let mut empty_kv = [];
    let mut ctx_no_kv = ExecutionContext::new(&[1_u32], 64, &mut logits, &mut empty_kv)
        .unwrap()
        .with_feature_input(&traj, 1, 4)
        .unwrap();
    assert!(matches!(
        valid_dt.prefill_step(&mut ctx_no_kv, &ok_gguf),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

#[derive(Debug)]
struct MockTargetLlmRecipe {
    arch: &'static str,
    version: &'static str,
    hidden_dim: usize,
}

impl ArchRecipe for MockTargetLlmRecipe {
    fn architecture_id(&self) -> &'static str {
        self.arch
    }

    fn recipe_version(&self) -> &'static str {
        self.version
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        matches!(quant, QuantType::F32 | QuantType::F16 | QuantType::BF16)
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let emb_view = weights.tensor("token_embd.weight")?;
        let raw = emb_view.as_bytes();
        let mut sum_vec = vec![0.0_f32; self.hidden_dim];
        if let Some(sp) = ctx.soft_prompt_embeds() {
            let (n_sp, sp_dim) = ctx.soft_prompt_dims().unwrap();
            assert_eq!(sp_dim, self.hidden_dim);
            for s in 0..n_sp {
                for d in 0..self.hidden_dim {
                    sum_vec[d] += sp[s * sp_dim + d];
                }
            }
            ctx.advance_position(n_sp)?;
        }
        for &tok in ctx.prompt_tokens() {
            let row = tok as usize;
            for (d, slot) in sum_vec.iter_mut().enumerate() {
                let off = (row * self.hidden_dim + d) * 4;
                *slot += f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
            }
        }
        let n_tok = ctx.prompt_tokens().len();
        ctx.advance_position(n_tok)?;
        for (i, slot) in ctx.logits_mut().iter_mut().enumerate() {
            *slot = sum_vec[i % self.hidden_dim] + (i as f32) * 0.01;
        }
        Ok(())
    }

    fn decode_step(
        &self,
        _ctx: &mut ExecutionContext<'_>,
        _weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        Ok(())
    }
}

#[test]
fn test_mmproj_dimension_mismatch_rejected() {
    let bridge = MmprojLatentBridge::new("qwen2", "1.0.0", 8, 16, 2);
    assert!(matches!(
        bridge.verify_target_compatibility("qwen3", "1.0.0", 16),
        Err(K3Error::CapabilityMismatch { .. })
    ));
    assert!(matches!(
        bridge.verify_target_compatibility("qwen2", "2.0.0", 16),
        Err(K3Error::CapabilityMismatch { .. })
    ));
    assert!(matches!(
        bridge.verify_target_compatibility("qwen2", "1.0.0", 32),
        Err(K3Error::CapabilityMismatch { .. })
    ));

    let target = MockTargetLlmRecipe {
        arch: "qwen2",
        version: "1.0.0",
        hidden_dim: 16,
    };
    let bytes = build_gguf_f32(
        "mmproj",
        &[],
        &[("mmproj.weight", &[8, 2 * 16], vec![0.1_f32; 8 * 32])],
    );
    let gguf = GgufFile::parse(&bytes).unwrap();
    let wrong_meta_bytes = create_synthetic_gguf_v3_with_kvs(
        "qwen2",
        32,
        &[("qwen2.embedding_length", GgufValue::Uint32(32))],
        &[],
    );
    let wrong_gguf = GgufFile::parse(&wrong_meta_bytes).unwrap();
    assert!(matches!(
        bridge.verify_target_recipe(&target, wrong_gguf.metadata()),
        Err(K3Error::CapabilityMismatch { .. })
    ));

    let mut out = vec![0.0_f32; 32];
    assert!(matches!(
        project_machine_latent_to_soft_prompt(&bridge, &[1.0_f32; 4], &gguf, &mut out),
        Err(K3Error::TensorBoundsError { .. })
    ));
    assert!(matches!(
        project_machine_latent_to_soft_prompt(
            &bridge,
            &[1.0_f32, 0.0, f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0],
            &gguf,
            &mut out,
        ),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

#[test]
fn test_mmproj_projects_soft_prompt_onto_decision_transformer_and_safetensors_parity() {
    let bridge = MmprojLatentBridge::new("decision-transformer", "1.0.0", 4, 8, 2);
    let fc1_w: Vec<f32> = (0..24).map(|i| 0.2 + (i as f32) * 0.01).collect();
    let fc1_b = vec![0.05_f32; 6];
    let fc2_w: Vec<f32> = (0..96).map(|i| 0.15 - ((i % 5) as f32) * 0.02).collect();
    let fc2_b = vec![0.02_f32; 16];

    let gguf_bytes = build_gguf_f32(
        "mmproj",
        &[],
        &[
            ("mmproj.fc1.weight", &[4, 6], fc1_w.clone()),
            ("mmproj.fc1.bias", &[6], fc1_b.clone()),
            ("mmproj.fc2.weight", &[6, 16], fc2_w.clone()),
            ("mmproj.fc2.bias", &[16], fc2_b.clone()),
        ],
    );
    let gguf = GgufFile::parse(&gguf_bytes).unwrap();

    // Equivalent SafeTensors with HF aliases and PyTorch [out_dim, in_dim] matrix shapes
    let st_bytes = build_safetensors_f32(
        &[("general.architecture", "mmproj")],
        &[
            ("mmproj.0.weight", &[6, 4], fc1_w),
            ("mmproj.0.bias", &[6], fc1_b),
            ("mmproj.2.weight", &[16, 6], fc2_w),
            ("mmproj.2.bias", &[16], fc2_b),
        ],
    );
    let st = SafeTensorsFile::from_bytes(&st_bytes).unwrap();

    let mut soft_gguf = vec![0.0_f32; 16];
    let mut soft_st = vec![0.0_f32; 16];
    let mut soft_other = vec![0.0_f32; 16];
    project_machine_latent_to_soft_prompt(&bridge, &[0.5, -0.3, 0.8, 0.1], &gguf, &mut soft_gguf)
        .expect("mmproj gguf succeeds");
    project_machine_latent_to_soft_prompt(&bridge, &[0.5, -0.3, 0.8, 0.1], &st, &mut soft_st)
        .expect("mmproj safetensors succeeds");
    project_machine_latent_to_soft_prompt(&bridge, &[-0.9, 0.7, 0.2, 1.4], &st, &mut soft_other)
        .expect("mmproj second latent succeeds");

    assert_ne!(soft_gguf, soft_other);
    assert_f32_slice_parity(&soft_gguf, &soft_st, "mmproj_mlp_gguf_vs_safetensors");

    // Layer-0 soft-prompt injection onto real DecisionTransformerRecipe shifts action logits
    let dt = DecisionTransformerRecipe {
        state_dim: 3,
        act_dim: 4,
        hidden_dim: 8,
        num_layers: 1,
        n_heads_q: 2,
        n_heads_kv: 2,
        head_dim: 4,
        ffn_dim: 16,
        max_ep_len: 16,
        rms_eps: 1e-6,
        rope_theta: 10_000.0,
    };
    let dt_bytes = build_dt_gguf(&dt);
    let dt_gguf = GgufFile::parse(&dt_bytes).unwrap();
    bridge
        .verify_target_recipe(&dt, dt_gguf.metadata())
        .expect("bridge matches DecisionTransformerRecipe");

    let mut logits_plain = [0.0_f32; 4];
    let mut logits_soft = [0.0_f32; 4];
    let mut kv_plain = vec![0.0_f32; 64 * 2 * 8 * 2];
    let mut kv_soft = vec![0.0_f32; 64 * 2 * 8 * 2];

    let mut ctx_plain =
        ExecutionContext::new(&[1_u32, 2], 64, &mut logits_plain, &mut kv_plain).unwrap();
    dt.prefill_step(&mut ctx_plain, &dt_gguf).unwrap();

    let end_pos = forward_with_soft_prompt(
        &dt,
        &[1_u32, 2],
        64,
        &soft_st,
        2,
        8,
        &mut logits_soft,
        &mut kv_soft,
        &dt_gguf,
    )
    .expect("layer-0 soft prompt forward on DecisionTransformer succeeds");
    assert_eq!(end_pos, 4);
    assert!(logits_soft.iter().all(|v| v.is_finite()));
    assert_ne!(logits_plain, logits_soft);
}

#[test]
fn test_mmproj_projects_soft_prompt_onto_mock_target_llm() {
    let qwen_bridge = MmprojLatentBridge::new("qwen2", "1.0.0", 4, 8, 2);
    let soft_st = vec![0.15_f32; 16];
    let qwen_mock = MockTargetLlmRecipe {
        arch: "qwen2",
        version: "1.0.0",
        hidden_dim: 8,
    };
    let qwen_st = build_safetensors_f32(
        &[("general.architecture", "qwen2"), ("hidden_size", "8")],
        &[(
            "token_embd.weight",
            &[8, 8],
            (0..64).map(|i| 0.1 + (i as f32) * 0.02).collect(),
        )],
    );
    let qwen_weights = SafeTensorsFile::from_bytes(&qwen_st).unwrap();
    qwen_bridge
        .verify_target_recipe(&qwen_mock, qwen_weights.metadata())
        .unwrap();
    let mut q_logits_plain = [0.0_f32; 8];
    let mut q_logits_soft = [0.0_f32; 8];
    let mut q_kv = vec![0.0_f32; 64 * 8];
    let mut q_ctx_plain =
        ExecutionContext::new(&[2_u32, 3], 64, &mut q_logits_plain, &mut q_kv).unwrap();
    qwen_mock
        .prefill_step(&mut q_ctx_plain, &qwen_weights)
        .unwrap();
    let q_end = forward_with_soft_prompt(
        &qwen_mock,
        &[2_u32, 3],
        64,
        &soft_st,
        2,
        8,
        &mut q_logits_soft,
        &mut q_kv,
        &qwen_weights,
    )
    .unwrap();
    assert_eq!(q_end, 4);
    assert_ne!(q_logits_plain, q_logits_soft);
}

#[test]
fn test_mmproj_latent_bridge_tuple_check_and_linear_path() {
    let bridge = MmprojLatentBridge::new("qwen2", "1.0.0", 4, 8, 2);
    bridge
        .verify_target_compatibility("qwen2", "1.0.0", 8)
        .expect("matching tuple passes");

    // Fail-closed: absent projector weights MUST error instead of synthesizing identity
    let empty_bytes = create_synthetic_gguf_v3_with_kvs("mmproj", 32, &[], &[]);
    let empty_gguf = GgufFile::parse(&empty_bytes).unwrap();
    let mut out = vec![0.0_f32; 16];
    assert!(matches!(
        project_machine_latent_to_soft_prompt(&bridge, &[1.0_f32; 4], &empty_gguf, &mut out),
        Err(K3Error::TensorBoundsError { .. })
    ));

    let empty_st_bytes = create_synthetic_safetensors(&[], &[]);
    let empty_st = SafeTensorsFile::from_bytes(&empty_st_bytes).unwrap();
    assert!(matches!(
        project_machine_latent_to_soft_prompt(&bridge, &[1.0_f32; 4], &empty_st, &mut out),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // Linear projector path (`mmproj.weight` + `mmproj.norm.weight`) across GGUF and SafeTensors
    let w: Vec<f32> = (0..64).map(|i| 0.1 + (i % 4) as f32 * 0.05).collect();
    let b = vec![0.02_f32; 16];
    let nw = vec![1.0_f32; 8];
    let nb = vec![0.0_f32; 8];
    let w_bytes = build_gguf_f32(
        "mmproj",
        &[],
        &[
            ("mmproj.weight", &[4, 16], w.clone()),
            ("mmproj.bias", &[16], b.clone()),
            ("mmproj.norm.weight", &[8], nw.clone()),
            ("mmproj.norm.bias", &[8], nb.clone()),
        ],
    );
    let w_gguf = GgufFile::parse(&w_bytes).unwrap();
    let w_st_bytes = build_safetensors_f32(
        &[],
        &[
            ("multi_modal_projector.weight", &[16, 4], w),
            ("multi_modal_projector.bias", &[16], b),
            ("multi_modal_projector.norm.weight", &[8], nw),
            ("multi_modal_projector.norm.bias", &[8], nb),
        ],
    );
    let w_st = SafeTensorsFile::from_bytes(&w_st_bytes).unwrap();

    let mut out_st = vec![0.0_f32; 16];
    project_machine_latent_to_soft_prompt(&bridge, &[0.4, 0.8, -0.2, 0.6], &w_gguf, &mut out)
        .expect("linear gguf mmproj path passes");
    project_machine_latent_to_soft_prompt(&bridge, &[0.4, 0.8, -0.2, 0.6], &w_st, &mut out_st)
        .expect("linear safetensors mmproj path passes");
    assert_f32_slice_parity(&out, &out_st, "mmproj_linear_gguf_vs_safetensors");
}

#[test]
fn test_safetensors_f16_and_bf16_non_finite_rejected_fail_closed() {
    // Non-finite F16 (0x7E00 = NaN) inside SafeTensors weight must fail closed
    let f16_nan_bytes: Vec<u8> = [0x00_u16, 0x3C00, 0x7E00, 0x3C00]
        .iter()
        .flat_map(|b| b.to_le_bytes())
        .collect();
    let st_nan = create_synthetic_safetensors(
        &[],
        &[("mmproj.weight", QuantType::F16, &[2, 2], &f16_nan_bytes)],
    );
    let st_parsed = SafeTensorsFile::from_bytes(&st_nan).unwrap();
    let bridge = MmprojLatentBridge::new("qwen2", "1.0.0", 2, 2, 1);
    let mut out = [0.0_f32; 2];
    assert!(matches!(
        project_machine_latent_to_soft_prompt(&bridge, &[1.0, 2.0], &st_parsed, &mut out),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // Valid F16 SafeTensors weight works cleanly
    let f16_ok_bytes: Vec<u8> = [0x3C00_u16, 0x4000, 0x4200, 0x3C00]
        .iter()
        .flat_map(|b| b.to_le_bytes())
        .collect();
    let st_ok = create_synthetic_safetensors(
        &[],
        &[("mmproj.weight", QuantType::F16, &[2, 2], &f16_ok_bytes)],
    );
    let st_ok_parsed = SafeTensorsFile::from_bytes(&st_ok).unwrap();
    project_machine_latent_to_soft_prompt(&bridge, &[1.0, 2.0], &st_ok_parsed, &mut out)
        .expect("valid F16 SafeTensors projector succeeds");
    assert!((out[0] - 5.0).abs() < 1e-4);
    assert!((out[1] - 5.0).abs() < 1e-4);
}
