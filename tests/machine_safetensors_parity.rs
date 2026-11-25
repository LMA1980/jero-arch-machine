use jero_arch_machine::{
    DecisionTransformerRecipe, FtTransformerRecipe, MmprojLatentBridge, TabPfnV2Recipe, TtmRecipe,
    project_machine_latent_to_soft_prompt,
};
use jero_core::safetensors::test_helpers::create_synthetic_safetensors;
use jero_core::{ArchRecipe, ExecutionContext, K3Error, QuantType, SafeTensorsFile};

fn to_f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn build_ttm_st_bytes() -> Vec<u8> {
    let f4 = to_f32_bytes(&[0.25; 4]);
    let f8 = to_f32_bytes(&[0.1; 8]);
    let z4 = to_f32_bytes(&[0.0; 4]);
    let z2 = to_f32_bytes(&[0.0; 2]);
    let ones2 = to_f32_bytes(&[1.0; 2]);
    let patch_w = to_f32_bytes(&[0.5, 0.5]);
    let head_b = to_f32_bytes(&[0.1, -0.1]);

    let mut specs: Vec<(String, QuantType, &[usize], &[u8])> = vec![
        (
            "backbone.patcher.weight".into(),
            QuantType::F32,
            &[2],
            &patch_w,
        ),
        (
            "head.projection.weight".into(),
            QuantType::F32,
            &[2, 2],
            &f4,
        ),
        ("head.projection.bias".into(), QuantType::F32, &[2], &head_b),
    ];
    for mixer in ["patch_mixer", "channel_mixer"] {
        let base = format!("backbone.encoder.layers.0.{mixer}");
        specs.extend([
            (
                format!("{base}.norm.weight"),
                QuantType::F32,
                &[2][..],
                ones2.as_slice(),
            ),
            (
                format!("{base}.norm.bias"),
                QuantType::F32,
                &[2][..],
                z2.as_slice(),
            ),
            (
                format!("{base}.mlp.fc1.weight"),
                QuantType::F32,
                &[4, 2][..],
                f8.as_slice(),
            ),
            (
                format!("{base}.mlp.fc1.bias"),
                QuantType::F32,
                &[4][..],
                z4.as_slice(),
            ),
            (
                format!("{base}.mlp.fc2.weight"),
                QuantType::F32,
                &[2, 4][..],
                f8.as_slice(),
            ),
            (
                format!("{base}.mlp.fc2.bias"),
                QuantType::F32,
                &[2][..],
                z2.as_slice(),
            ),
        ]);
    }
    let refs: Vec<(&str, QuantType, &[usize], &[u8])> = specs
        .iter()
        .map(|(n, q, s, b)| (n.as_str(), *q, *s, *b))
        .collect();
    create_synthetic_safetensors(
        &[
            ("general.architecture", "ttm"),
            ("ttm.context_length", "4"),
            ("ttm.patch_length", "2"),
            ("ttm.forecast_length", "2"),
            ("ttm.num_channels", "2"),
            ("ttm.time_hidden_dim", "4"),
            ("ttm.channel_hidden_dim", "4"),
        ],
        &refs,
    )
}

#[test]
fn integration_safetensors_hydration_and_fail_closed_smoke() {
    let ttm_bytes = build_ttm_st_bytes();
    let ttm_st = SafeTensorsFile::from_bytes(&ttm_bytes).expect("parse ttm safetensors");
    let ttm = TtmRecipe::from_safetensors(&ttm_st);
    ttm.validate_model(ttm_st.metadata(), &ttm_st)
        .expect("validate ttm model");

    let mut ttm_out = [0.0_f32; 4];
    let mut kv = [];
    let mut ctx = ExecutionContext::new_feature_input(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        4,
        2,
        64,
        &mut ttm_out,
        &mut kv,
    )
    .unwrap();
    ttm.forward_classify(&mut ctx, &ttm_st).unwrap();
    assert!(ttm_out.iter().all(|v| v.is_finite()));

    let bridge = MmprojLatentBridge::new("decision-transformer", "1.0.0", 2, 4, 1);
    let empty_st_bytes = create_synthetic_safetensors(&[], &[]);
    let empty_st = SafeTensorsFile::from_bytes(&empty_st_bytes).unwrap();
    let mut sp = [0.0_f32; 4];
    assert!(matches!(
        project_machine_latent_to_soft_prompt(&bridge, &[1.0, -1.0], &empty_st, &mut sp),
        Err(K3Error::TensorBoundsError { .. })
    ));

    let ft = FtTransformerRecipe::new();
    let tabpfn = TabPfnV2Recipe::new();
    let dt = DecisionTransformerRecipe::new();
    assert!(matches!(
        ft.validate_model(ttm_st.metadata(), &ttm_st),
        Err(K3Error::CapabilityMismatch { .. })
    ));
    assert!(matches!(
        tabpfn.validate_model(ttm_st.metadata(), &ttm_st),
        Err(K3Error::CapabilityMismatch { .. })
    ));
    assert!(matches!(
        dt.validate_model(ttm_st.metadata(), &ttm_st),
        Err(K3Error::CapabilityMismatch { .. })
    ));
}
