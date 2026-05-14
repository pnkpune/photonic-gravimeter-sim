from __future__ import annotations

import json

import numpy as np

from gravnav.estimators.error_state_ins import (
    ErrorStateINS,
    ErrorStateINSNominalState,
    ErrorStateINSProcessNoise,
    ErrorStateINSState,
)
from gravnav.estimators.feedback_policy import SequenceFeedbackController, SequenceFeedbackSpec
from gravnav.estimators.gravity_sequence_match import (
    SequenceAmbiguityDiagnostics,
    SequenceCandidateHypothesis,
    SequenceMatchEstimate,
    SequenceMatchUpdateResult,
)
from gravnav.ml.feedback_trust import (
    SEQUENCE_FEEDBACK_FEATURE_NAMES,
    _primary_failure_negative_mask,
    SequenceFeedbackEventCorpus,
    SequenceFeedbackTrustCommittee,
    SequenceFeedbackTrustModel,
    SequenceFeedbackTrustModelSpec,
    cross_validate_sequence_feedback_trust_model,
    extract_sequence_feedback_features,
    fit_sequence_feedback_trust_model,
    load_sequence_feedback_trust_committee,
)


def _make_state(
    *,
    time_s: float,
    north_speed_mps: float = 1.0,
    east_speed_mps: float = 0.0,
) -> ErrorStateINSState:
    nominal = ErrorStateINSNominalState(
        time_s=float(time_s),
        lat_rad=np.deg2rad(63.0),
        lon_rad=np.deg2rad(10.0),
        height_m=0.0,
        v_ned_mps=np.array([north_speed_mps, east_speed_mps, 0.0], dtype=np.float64),
        C_n_b=np.eye(3, dtype=np.float64),
        gyro_bias_radps=np.zeros(3, dtype=np.float64),
        accel_bias_mps2=np.zeros(3, dtype=np.float64),
    )
    P = np.eye(15, dtype=np.float64)
    P[:3, :3] *= 1.0e-10
    return ErrorStateINSState(
        nominal=nominal,
        P=P,
        process_noise=ErrorStateINSProcessNoise.perfect(),
    )


def _make_sequence_update() -> SequenceMatchUpdateResult:
    ambiguity = SequenceAmbiguityDiagnostics(
        posterior_candidate_ess=12.0,
        posterior_candidate_ess_fraction=0.25,
        edge_mass_fraction=0.08,
        support_radius_n_m=22.0,
        support_radius_e_m=18.0,
        horizontal_covariance_eigenvalue_ratio=2.5,
        grid_saturated_north=False,
        grid_saturated_east=False,
        grid_saturated_any=False,
        gravity_predicted_spread_mps2=4.0e-6,
        gravity_information_ratio=2.1,
        bathymetry_predicted_spread_m=5.0,
        bathymetry_information_ratio=1.6,
        magnetic_predicted_spread_nt=12.0,
        magnetic_information_ratio=1.4,
        dominant_failure_mode="informative",
        grid_mode="expanded",
        grid_half_span_m=np.array([60.0, 60.0], dtype=np.float64),
        grid_spacing_m=np.array([10.0, 10.0], dtype=np.float64),
    )
    estimate = SequenceMatchEstimate(
        lat_rad=np.deg2rad(63.0001),
        lon_rad=np.deg2rad(10.0001),
        height_m=0.0,
        covariance_ned_m2=np.diag([36.0, 81.0, 4.0]).astype(np.float64),
        covariance_geodetic=np.diag([1.0e-12, 1.0e-12, 4.0]).astype(np.float64),
        predicted_disturbance_mps2=1.0e-5,
        marginal_peak_probability=0.55,
        predicted_bathymetry_m=100.0,
        predicted_magnetic_total_nt=45000.0,
    )
    return SequenceMatchUpdateResult(
        estimate=estimate,
        global_index=10,
        time_s=50.0,
        window_size_used=7,
        delayed_by_steps=3,
        num_candidates=49,
        posterior_entropy_nats=1.2,
        marginal_peak_probability=0.55,
        predicted_disturbance_mean_mps2=1.0e-5,
        predicted_disturbance_std_mps2=4.0e-6,
        used_gradient=True,
        used_bathymetry=True,
        used_magnetics=True,
        viterbi_log_score=-2.0,
        viterbi_offset_ned_m=np.array([10.0, -8.0, 0.0], dtype=np.float64),
        posterior_mean_offset_ned_m=np.array([8.0, -4.0, 0.0], dtype=np.float64),
        ambiguity_diagnostics=ambiguity,
        candidate_hypotheses=(
            SequenceCandidateHypothesis(
                rank=0,
                candidate_index=11,
                marginal_probability=0.44,
                probability_gap_to_best=0.0,
                lat_rad=np.deg2rad(63.00015),
                lon_rad=np.deg2rad(10.00005),
                height_m=0.0,
                offset_ned_m=np.array([4.0, -2.0, 0.0], dtype=np.float64),
                predicted_disturbance_mps2=1.1e-5,
                predicted_bathymetry_m=101.0,
                predicted_magnetic_total_nt=45010.0,
            ),
            SequenceCandidateHypothesis(
                rank=1,
                candidate_index=7,
                marginal_probability=0.31,
                probability_gap_to_best=0.13,
                lat_rad=np.deg2rad(63.00005),
                lon_rad=np.deg2rad(10.00015),
                height_m=0.0,
                offset_ned_m=np.array([2.0, -1.0, 0.0], dtype=np.float64),
                predicted_disturbance_mps2=9.0e-6,
                predicted_bathymetry_m=99.0,
                predicted_magnetic_total_nt=44990.0,
            ),
        ),
        publishability_probability=0.72,
        support_expansion_probability=0.33,
        learned_covariance_scale=0.9,
    )


def test_extract_sequence_feedback_features_returns_finite_vector() -> None:
    update = _make_sequence_update()
    current_state = _make_state(time_s=53.0, north_speed_mps=1.0, east_speed_mps=0.5)
    previous_state = _make_state(time_s=52.0, north_speed_mps=1.1, east_speed_mps=0.2)

    vector, values = extract_sequence_feedback_features(
        update,
        live_ins_state=current_state,
        current_time_s=53.0,
        previous_live_ins_state=previous_state,
    )

    assert vector.shape == (len(SEQUENCE_FEEDBACK_FEATURE_NAMES),)
    assert np.all(np.isfinite(vector))
    assert values["grid_is_expanded"] == 1.0
    assert values["used_bathymetry"] == 1.0
    assert values["publishability_probability"] == 0.72


def test_sequence_feedback_controller_uses_trust_gate_and_covariance_scale(
    tmp_path,
) -> None:
    model = SequenceFeedbackTrustModel(
        spec=SequenceFeedbackTrustModelSpec(),
        feature_mean=np.zeros(len(SEQUENCE_FEEDBACK_FEATURE_NAMES), dtype=np.float64),
        feature_std=np.ones(len(SEQUENCE_FEEDBACK_FEATURE_NAMES), dtype=np.float64),
        useful_weights=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        useful_bias=np.array([3.0, -3.0], dtype=np.float64),
        error_delta_weights=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        error_delta_bias=np.array([-20.0, 15.0], dtype=np.float64),
        gain_alpha_weights=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        gain_alpha_bias=np.ones(2, dtype=np.float64),
        metadata={},
    )
    model_path = tmp_path / "trust_model.npz"
    model.save_npz(model_path)

    spec = SequenceFeedbackSpec(
        enabled=True,
        mode="lag_replay",
        measurement_geometry="directional_horizontal",
        min_window_size=1,
        min_peak_probability=0.0,
        min_horizontal_eigenvalue_ratio=1.0,
        max_horizontal_std_m=1.0e6,
        max_correction_norm_m=1.0e6,
        covariance_inflation=4.0,
        nis_threshold=None,
        trust_model_export_path=str(model_path),
        trust_gate_source="both",
        min_trust_probability=0.5,
        apply_trust_covariance_scale=True,
        trust_covariance_scale_min=0.5,
        trust_covariance_scale_max=2.0,
    )
    ctrl = SequenceFeedbackController(spec)
    update = _make_sequence_update()
    delayed_state = _make_state(time_s=50.0)
    current_state = _make_state(time_s=53.0)
    previous_state = _make_state(time_s=52.0, north_speed_mps=1.0, east_speed_mps=0.2)

    result = ctrl.evaluate(
        update,
        ErrorStateINS(delayed_state.copy()),
        current_time_s=53.0,
        live_ins_state=current_state,
        previous_live_ins_state=previous_state,
    )

    assert result.diagnostics.trust_gate_source == "both"
    assert result.diagnostics.trust_allowed is True
    assert result.diagnostics.trust_probability is not None
    assert result.diagnostics.trust_probability > 0.5
    assert result.diagnostics.covariance_inflation_applied < spec.covariance_inflation
    assert result.diagnostics.predicted_error_delta_m is not None


def test_sequence_feedback_controller_reranks_topk_candidates(tmp_path) -> None:
    correction_norm_idx = SEQUENCE_FEEDBACK_FEATURE_NAMES.index("correction_norm_m")
    model = SequenceFeedbackTrustModel(
        spec=SequenceFeedbackTrustModelSpec(),
        feature_mean=np.zeros(len(SEQUENCE_FEEDBACK_FEATURE_NAMES), dtype=np.float64),
        feature_std=np.ones(len(SEQUENCE_FEEDBACK_FEATURE_NAMES), dtype=np.float64),
        useful_weights=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        useful_bias=np.array([3.0, -3.0], dtype=np.float64),
        error_delta_weights=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        error_delta_bias=np.array([-2.0, 2.0], dtype=np.float64),
        gain_alpha_weights=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        gain_alpha_bias=np.ones(2, dtype=np.float64),
        metadata={},
    )
    model.useful_weights[0, correction_norm_idx] = -0.5
    model.error_delta_weights[0, correction_norm_idx] = 0.5
    model_path = tmp_path / "trust_model_topk.npz"
    model.save_npz(model_path)

    spec = SequenceFeedbackSpec(
        enabled=True,
        mode="lag_replay",
        measurement_geometry="directional_horizontal",
        min_window_size=1,
        min_peak_probability=0.0,
        min_horizontal_eigenvalue_ratio=1.0,
        max_horizontal_std_m=1.0e6,
        max_correction_norm_m=1.0e6,
        covariance_inflation=4.0,
        nis_threshold=None,
        trust_model_export_path=str(model_path),
        trust_gate_source="both",
        min_trust_probability=0.5,
        max_predicted_error_delta_m=0.0,
        candidate_selection_mode="topk_trust_rerank",
    )
    ctrl = SequenceFeedbackController(spec)
    update = _make_sequence_update()
    delayed_state = _make_state(time_s=50.0)
    current_state = _make_state(time_s=53.0)
    previous_state = _make_state(time_s=52.0, north_speed_mps=1.0, east_speed_mps=0.2)

    result = ctrl.evaluate(
        update,
        ErrorStateINS(delayed_state.copy()),
        current_time_s=53.0,
        live_ins_state=current_state,
        previous_live_ins_state=previous_state,
    )

    assert result.applied is True
    assert result.diagnostics.candidate_selection_mode == "topk_trust_rerank"
    assert result.diagnostics.candidate_selection_selected_source == "candidate_hypothesis"
    assert result.diagnostics.candidate_selection_selected_rank == 1
    assert result.diagnostics.candidate_selection_considered == 3
    assert np.allclose(
        result.diagnostics.horizontal_offset_ned_m,
        np.array([2.0, -1.0], dtype=np.float64),
    )


def test_fit_and_cross_validate_sequence_feedback_trust_model() -> None:
    feature_dim = len(SEQUENCE_FEEDBACK_FEATURE_NAMES)
    features = np.zeros((6, feature_dim), dtype=np.float64)
    features[:, 0] = np.array([-2.0, -1.0, 2.0, 3.0, -3.0, 4.0], dtype=np.float64)
    region_names = ("region_a", "region_b")
    region_index = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    useful = np.array([False, False, True, True, False, True], dtype=bool)
    error_delta = np.array([5.0, 2.0, -4.0, -6.0, 3.0, -8.0], dtype=np.float64)
    corpus = SequenceFeedbackEventCorpus(
        feature_names=SEQUENCE_FEEDBACK_FEATURE_NAMES,
        features=features,
        region_names=region_names,
        region_index=region_index,
        event_region_names=("region_a",) * 3 + ("region_b",) * 3,
        event_seed=np.array([42, 42, 42, 123, 123, 123], dtype=np.int64),
        current_time_s=np.arange(6, dtype=np.float64),
        update_time_s=np.arange(6, dtype=np.float64),
        current_step_index=np.arange(6, dtype=np.int64),
        target_step_index=np.arange(6, dtype=np.int64),
        lag_replay_applied=np.ones(6, dtype=bool),
        lag_replay_improves_error=useful.copy(),
        lag_replay_hmi_safe=np.ones(6, dtype=bool),
        lag_replay_useful_and_safe=useful.copy(),
        lag_replay_error_delta_m=error_delta.copy(),
        lag_replay_best_gain_alpha=np.array(
            [0.0, 0.25, 0.75, 1.0, 0.1, 0.8], dtype=np.float64
        ),
        bias_transfer_applied=np.ones(6, dtype=bool),
        bias_transfer_improves_error=useful.copy(),
        bias_transfer_hmi_safe=np.ones(6, dtype=bool),
        bias_transfer_useful_and_safe=useful.copy(),
        bias_transfer_error_delta_m=error_delta.copy(),
        bias_transfer_best_gain_alpha=np.array(
            [0.0, 0.2, 0.7, 0.9, 0.15, 0.85], dtype=np.float64
        ),
        metadata={},
    )

    model = fit_sequence_feedback_trust_model(corpus)
    prob = model.predict_trust_probability(features, mode="lag_replay")
    assert prob.shape == (6,)
    assert float(prob[2]) > float(prob[1])
    assert "median_applied_alpha" in model.metadata["training_metrics"]["lag_replay"]
    assert "gain_histogram_summary" in model.metadata
    assert "region_a" in model.metadata["gain_histogram_summary"]["lag_replay"]

    cv = cross_validate_sequence_feedback_trust_model(corpus)
    assert cv["num_folds"] == 2
    assert "lag_replay" in cv["aggregate"]
    assert "median_brier_score" in cv["aggregate"]["lag_replay"]
    assert "median_gain_alpha_rmse" in cv["aggregate"]["lag_replay"]
    assert "median_median_applied_alpha" in cv["aggregate"]["lag_replay"]


def test_sequence_feedback_event_corpus_roundtrip_preserves_event_seed(tmp_path) -> None:
    corpus = SequenceFeedbackEventCorpus(
        feature_names=SEQUENCE_FEEDBACK_FEATURE_NAMES,
        features=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        region_names=("region_a",),
        region_index=np.array([0, 0], dtype=np.int64),
        event_region_names=("region_a", "region_a"),
        event_seed=np.array([42, 123], dtype=np.int64),
        current_time_s=np.array([1.0, 2.0], dtype=np.float64),
        update_time_s=np.array([0.5, 1.5], dtype=np.float64),
        current_step_index=np.array([1, 2], dtype=np.int64),
        target_step_index=np.array([0, 1], dtype=np.int64),
        lag_replay_applied=np.array([True, False], dtype=bool),
        lag_replay_improves_error=np.array([True, False], dtype=bool),
        lag_replay_hmi_safe=np.array([True, True], dtype=bool),
        lag_replay_useful_and_safe=np.array([True, False], dtype=bool),
        lag_replay_error_delta_m=np.array([-5.0, 2.0], dtype=np.float64),
        lag_replay_best_gain_alpha=np.array([0.25, 0.0], dtype=np.float64),
        bias_transfer_applied=np.array([False, False], dtype=bool),
        bias_transfer_improves_error=np.array([False, False], dtype=bool),
        bias_transfer_hmi_safe=np.array([True, True], dtype=bool),
        bias_transfer_useful_and_safe=np.array([False, False], dtype=bool),
        bias_transfer_error_delta_m=np.array([1.0, 1.5], dtype=np.float64),
        bias_transfer_best_gain_alpha=np.array([0.0, 0.0], dtype=np.float64),
        metadata={"hello": "world"},
    )
    path = tmp_path / "feedback_corpus.npz"
    corpus.save_npz(path)
    loaded = SequenceFeedbackEventCorpus.from_npz(path)
    assert np.array_equal(loaded.event_seed, np.array([42, 123], dtype=np.int64))


def test_primary_failure_negative_mask_only_marks_flagged_pairs() -> None:
    corpus = SequenceFeedbackEventCorpus(
        feature_names=SEQUENCE_FEEDBACK_FEATURE_NAMES,
        features=np.zeros((4, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        region_names=("norwegian_margin_maritime", "helgeland_offshore"),
        region_index=np.array([0, 0, 1, 1], dtype=np.int64),
        event_region_names=(
            "norwegian_margin_maritime",
            "norwegian_margin_maritime",
            "helgeland_offshore",
            "helgeland_offshore",
        ),
        event_seed=np.array([42, 555, 42, 777], dtype=np.int64),
        current_time_s=np.arange(4, dtype=np.float64),
        update_time_s=np.arange(4, dtype=np.float64),
        current_step_index=np.arange(4, dtype=np.int64),
        target_step_index=np.arange(4, dtype=np.int64),
        lag_replay_applied=np.ones(4, dtype=bool),
        lag_replay_improves_error=np.zeros(4, dtype=bool),
        lag_replay_hmi_safe=np.ones(4, dtype=bool),
        lag_replay_useful_and_safe=np.zeros(4, dtype=bool),
        lag_replay_error_delta_m=np.ones(4, dtype=np.float64),
        lag_replay_best_gain_alpha=np.zeros(4, dtype=np.float64),
        bias_transfer_applied=np.zeros(4, dtype=bool),
        bias_transfer_improves_error=np.zeros(4, dtype=bool),
        bias_transfer_hmi_safe=np.ones(4, dtype=bool),
        bias_transfer_useful_and_safe=np.zeros(4, dtype=bool),
        bias_transfer_error_delta_m=np.ones(4, dtype=np.float64),
        bias_transfer_best_gain_alpha=np.zeros(4, dtype=np.float64),
        metadata={},
    )
    mask = _primary_failure_negative_mask(
        corpus,
        {
            "norwegian_margin_maritime": [42, 123, 777],
            "helgeland_offshore": [42, 123, 777],
        },
    )
    assert np.array_equal(mask, np.array([True, False, True, True], dtype=bool))


def test_load_committee_manifest_and_aggregate_conservative_unanimity(tmp_path) -> None:
    feature_dim = len(SEQUENCE_FEEDBACK_FEATURE_NAMES)
    zeros = np.zeros((2, feature_dim), dtype=np.float64)
    model_specs = [
        ("drop_seed_42", np.array([3.0, -3.0]), np.array([-10.0, 5.0]), np.array([0.24, 0.8])),
        ("drop_seed_123", np.array([2.0, -3.0]), np.array([-5.0, 5.0]), np.array([0.20, 0.7])),
        ("drop_seed_777", np.array([3.0, -3.0]), np.array([1.0, 5.0]), np.array([0.30, 0.9])),
    ]
    members = []
    for idx, (name, useful_bias, error_bias, gain_bias) in enumerate(model_specs):
        model = SequenceFeedbackTrustModel(
            spec=SequenceFeedbackTrustModelSpec(),
            feature_mean=np.zeros(feature_dim, dtype=np.float64),
            feature_std=np.ones(feature_dim, dtype=np.float64),
            useful_weights=zeros.copy(),
            useful_bias=useful_bias.astype(np.float64),
            error_delta_weights=zeros.copy(),
            error_delta_bias=error_bias.astype(np.float64),
            gain_alpha_weights=zeros.copy(),
            gain_alpha_bias=gain_bias.astype(np.float64),
            metadata={},
        )
        model_path = tmp_path / f"{name}.npz"
        model.save_npz(model_path)
        members.append(
            {
                "name": name,
                "drop_seed": [42, 123, 777][idx],
                "model_path": model_path.name,
            }
        )
    manifest_path = tmp_path / "committee_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "aggregator": "conservative_unanimity_v1",
                "members": members,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    committee = load_sequence_feedback_trust_committee(manifest_path)
    assert isinstance(committee, SequenceFeedbackTrustCommittee)
    assert len(committee.members) == 3

    prediction = committee.predict(
        np.zeros((1, feature_dim), dtype=np.float64),
        mode="lag_replay",
        min_trust_probability=0.7,
        max_predicted_error_delta_m=0.0,
        alpha_min=0.0,
        alpha_max=1.0,
        covariance_scale_min=0.75,
        covariance_scale_max=2.0,
    )
    assert prediction.trust_allowed is False
    assert prediction.members_passing == 2
    assert prediction.trust_probability == min(prediction.member_trust_probabilities)
    assert prediction.predicted_error_delta_m == max(
        prediction.member_predicted_error_delta_m
    )
    assert prediction.gain_alpha == min(prediction.member_gain_alpha)
    assert prediction.rejection_reason is not None


def test_sequence_feedback_controller_uses_committee_manifest(tmp_path) -> None:
    feature_dim = len(SEQUENCE_FEEDBACK_FEATURE_NAMES)
    zeros = np.zeros((2, feature_dim), dtype=np.float64)
    members = []
    for idx, (name, trust_bias, error_bias, gain_bias) in enumerate(
        [
            ("drop_seed_42", np.array([3.0, -3.0]), np.array([-10.0, 5.0]), np.array([0.24, 0.8])),
            ("drop_seed_123", np.array([2.5, -3.0]), np.array([-8.0, 5.0]), np.array([0.22, 0.7])),
            ("drop_seed_777", np.array([2.2, -3.0]), np.array([-6.0, 5.0]), np.array([0.20, 0.6])),
        ]
    ):
        model = SequenceFeedbackTrustModel(
            spec=SequenceFeedbackTrustModelSpec(),
            feature_mean=np.zeros(feature_dim, dtype=np.float64),
            feature_std=np.ones(feature_dim, dtype=np.float64),
            useful_weights=zeros.copy(),
            useful_bias=trust_bias.astype(np.float64),
            error_delta_weights=zeros.copy(),
            error_delta_bias=error_bias.astype(np.float64),
            gain_alpha_weights=zeros.copy(),
            gain_alpha_bias=gain_bias.astype(np.float64),
            metadata={},
        )
        model_path = tmp_path / f"{name}.npz"
        model.save_npz(model_path)
        members.append(
            {
                "name": name,
                "drop_seed": [42, 123, 777][idx],
                "model_path": model_path.name,
            }
        )
    manifest_path = tmp_path / "committee_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"aggregator": "conservative_unanimity_v1", "members": members},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    spec = SequenceFeedbackSpec(
        enabled=True,
        mode="lag_replay",
        measurement_geometry="directional_horizontal",
        min_window_size=1,
        min_peak_probability=0.0,
        min_horizontal_eigenvalue_ratio=1.0,
        max_horizontal_std_m=1.0e6,
        max_correction_norm_m=1.0e6,
        covariance_inflation=4.0,
        nis_threshold=None,
        trust_committee_manifest_path=str(manifest_path),
        trust_gate_source="both",
        min_trust_probability=0.5,
        max_predicted_error_delta_m=0.0,
        apply_trust_gain_alpha=True,
        trust_gain_alpha_min=0.1,
        trust_gain_alpha_max=0.9,
    )
    ctrl = SequenceFeedbackController(spec)
    result = ctrl.evaluate(
        _make_sequence_update(),
        ErrorStateINS(_make_state(time_s=50.0).copy()),
        current_time_s=53.0,
        live_ins_state=_make_state(time_s=53.0),
        previous_live_ins_state=_make_state(
            time_s=52.0, north_speed_mps=1.0, east_speed_mps=0.2
        ),
    )

    assert result.applied is True
    assert result.diagnostics.committee_member_count == 3
    assert result.diagnostics.committee_members_passing == 3
    assert len(result.diagnostics.committee_member_trust_probabilities) == 3
    assert len(result.diagnostics.committee_member_predicted_error_delta_m) == 3
    assert len(result.diagnostics.committee_member_gain_alpha) == 3


def test_sequence_feedback_controller_rejects_positive_predicted_error_delta(
    tmp_path,
) -> None:
    model = SequenceFeedbackTrustModel(
        spec=SequenceFeedbackTrustModelSpec(),
        feature_mean=np.zeros(len(SEQUENCE_FEEDBACK_FEATURE_NAMES), dtype=np.float64),
        feature_std=np.ones(len(SEQUENCE_FEEDBACK_FEATURE_NAMES), dtype=np.float64),
        useful_weights=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        useful_bias=np.array([3.0, -3.0], dtype=np.float64),
        error_delta_weights=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        error_delta_bias=np.array([5.0, 5.0], dtype=np.float64),
        gain_alpha_weights=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        gain_alpha_bias=np.ones(2, dtype=np.float64),
        metadata={},
    )
    model_path = tmp_path / "trust_model_positive_delta.npz"
    model.save_npz(model_path)

    spec = SequenceFeedbackSpec(
        enabled=True,
        mode="lag_replay",
        measurement_geometry="directional_horizontal",
        min_window_size=1,
        min_peak_probability=0.0,
        min_horizontal_eigenvalue_ratio=1.0,
        max_horizontal_std_m=1.0e6,
        max_correction_norm_m=1.0e6,
        covariance_inflation=4.0,
        nis_threshold=None,
        trust_model_export_path=str(model_path),
        trust_gate_source="both",
        min_trust_probability=0.5,
        max_predicted_error_delta_m=0.0,
        apply_trust_covariance_scale=False,
    )
    ctrl = SequenceFeedbackController(spec)
    update = _make_sequence_update()
    delayed_state = _make_state(time_s=50.0)
    current_state = _make_state(time_s=53.0)
    previous_state = _make_state(time_s=52.0, north_speed_mps=1.0, east_speed_mps=0.2)

    result = ctrl.evaluate(
        update,
        ErrorStateINS(delayed_state.copy()),
        current_time_s=53.0,
        live_ins_state=current_state,
        previous_live_ins_state=previous_state,
    )

    assert result.applied is False
    assert result.diagnostics.trust_allowed is False
    assert result.diagnostics.predicted_error_delta_m is not None
    assert result.diagnostics.predicted_error_delta_m > 0.0
    assert result.diagnostics.trust_rejection_reason is not None
    assert "predicted_error_delta_m" in result.diagnostics.trust_rejection_reason


def test_sequence_feedback_controller_applies_trust_gain_alpha(tmp_path) -> None:
    model = SequenceFeedbackTrustModel(
        spec=SequenceFeedbackTrustModelSpec(),
        feature_mean=np.zeros(len(SEQUENCE_FEEDBACK_FEATURE_NAMES), dtype=np.float64),
        feature_std=np.ones(len(SEQUENCE_FEEDBACK_FEATURE_NAMES), dtype=np.float64),
        useful_weights=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        useful_bias=np.array([3.0, -3.0], dtype=np.float64),
        error_delta_weights=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        error_delta_bias=np.array([-10.0, 5.0], dtype=np.float64),
        gain_alpha_weights=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        gain_alpha_bias=np.array([0.25, 0.75], dtype=np.float64),
        metadata={},
    )
    model_path = tmp_path / "trust_model_gain_alpha.npz"
    model.save_npz(model_path)

    spec = SequenceFeedbackSpec(
        enabled=True,
        mode="lag_replay",
        measurement_geometry="directional_horizontal",
        min_window_size=1,
        min_peak_probability=0.0,
        min_horizontal_eigenvalue_ratio=1.0,
        max_horizontal_std_m=1.0e6,
        max_correction_norm_m=1.0e6,
        covariance_inflation=4.0,
        nis_threshold=None,
        trust_model_export_path=str(model_path),
        trust_gate_source="both",
        min_trust_probability=0.5,
        apply_trust_gain_alpha=True,
        trust_gain_alpha_min=0.1,
        trust_gain_alpha_max=0.9,
    )
    ctrl = SequenceFeedbackController(spec)
    result = ctrl.evaluate(
        _make_sequence_update(),
        ErrorStateINS(_make_state(time_s=50.0).copy()),
        current_time_s=53.0,
        live_ins_state=_make_state(time_s=53.0),
        previous_live_ins_state=_make_state(
            time_s=52.0, north_speed_mps=1.0, east_speed_mps=0.2
        ),
    )

    assert result.applied is True
    assert result.diagnostics.trust_allowed is True
    assert result.diagnostics.gain_alpha_applied == 0.25


def test_predict_gain_alpha_is_conservative_around_reference() -> None:
    model = SequenceFeedbackTrustModel(
        spec=SequenceFeedbackTrustModelSpec(
            trust_threshold=0.5,
            gain_alpha_reference=0.25,
            gain_alpha_prediction_scale=0.5,
            gain_alpha_safe_max=0.5,
        ),
        feature_mean=np.zeros(len(SEQUENCE_FEEDBACK_FEATURE_NAMES), dtype=np.float64),
        feature_std=np.ones(len(SEQUENCE_FEEDBACK_FEATURE_NAMES), dtype=np.float64),
        useful_weights=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        useful_bias=np.array([3.0, -3.0], dtype=np.float64),
        error_delta_weights=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        error_delta_bias=np.zeros(2, dtype=np.float64),
        gain_alpha_weights=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        gain_alpha_bias=np.array([1.0, 1.0], dtype=np.float64),
        metadata={},
    )
    predicted = model.predict_gain_alpha(
        np.zeros((1, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        mode="lag_replay",
        alpha_min=0.0,
        alpha_max=1.0,
    )
    assert predicted.shape == (1,)
    assert predicted[0] <= 0.5
    assert predicted[0] > 0.25


def test_sequence_feedback_controller_limits_learned_gain_updates(tmp_path) -> None:
    model = SequenceFeedbackTrustModel(
        spec=SequenceFeedbackTrustModelSpec(),
        feature_mean=np.zeros(len(SEQUENCE_FEEDBACK_FEATURE_NAMES), dtype=np.float64),
        feature_std=np.ones(len(SEQUENCE_FEEDBACK_FEATURE_NAMES), dtype=np.float64),
        useful_weights=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        useful_bias=np.array([3.0, -3.0], dtype=np.float64),
        error_delta_weights=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        error_delta_bias=np.array([-10.0, 5.0], dtype=np.float64),
        gain_alpha_weights=np.zeros((2, len(SEQUENCE_FEEDBACK_FEATURE_NAMES)), dtype=np.float64),
        gain_alpha_bias=np.array([0.25, 0.75], dtype=np.float64),
        metadata={},
    )
    model_path = tmp_path / "trust_model_gain_alpha_budget.npz"
    model.save_npz(model_path)

    spec = SequenceFeedbackSpec(
        enabled=True,
        mode="lag_replay",
        measurement_geometry="directional_horizontal",
        min_window_size=1,
        min_peak_probability=0.0,
        min_horizontal_eigenvalue_ratio=1.0,
        max_horizontal_std_m=1.0e6,
        max_correction_norm_m=1.0e6,
        covariance_inflation=4.0,
        nis_threshold=None,
        trust_model_export_path=str(model_path),
        trust_gate_source="both",
        min_trust_probability=0.5,
        apply_trust_gain_alpha=True,
        trust_gain_alpha_min=0.1,
        trust_gain_alpha_max=0.9,
        learned_gain_max_applied_updates=1,
    )
    ctrl = SequenceFeedbackController(spec)
    first = ctrl.evaluate(
        _make_sequence_update(),
        ErrorStateINS(_make_state(time_s=50.0).copy()),
        current_time_s=53.0,
        live_ins_state=_make_state(time_s=53.0),
        previous_live_ins_state=_make_state(
            time_s=52.0, north_speed_mps=1.0, east_speed_mps=0.2
        ),
    )
    second = ctrl.evaluate(
        _make_sequence_update(),
        ErrorStateINS(_make_state(time_s=57.0).copy()),
        current_time_s=60.0,
        live_ins_state=_make_state(time_s=60.0),
        previous_live_ins_state=_make_state(
            time_s=59.0, north_speed_mps=1.0, east_speed_mps=0.2
        ),
    )

    assert first.applied is True
    assert first.diagnostics.runtime_budget_active is True
    assert first.diagnostics.runtime_budget_allowed is True
    assert first.diagnostics.applied_update_count_before == 0

    assert second.applied is False
    assert second.diagnostics.runtime_budget_active is True
    assert second.diagnostics.runtime_budget_allowed is False
    assert (
        second.diagnostics.runtime_budget_rejection_reason
        == "learned_gain_max_applied_updates_reached"
    )
    assert second.diagnostics.applied_update_count_before == 1
    assert second.diagnostics.rejection_reason == (
        "learned_gain_max_applied_updates_reached"
    )


def test_sequence_feedback_controller_rejects_overconfident_projected_std() -> None:
    update = _make_sequence_update()
    update = SequenceMatchUpdateResult(
        **{
            **update.__dict__,
            "estimate": SequenceMatchEstimate(
                **{
                    **update.estimate.__dict__,
                    "covariance_ned_m2": np.diag([1.0e-12, 81.0, 4.0]).astype(np.float64),
                }
            ),
        }
    )
    spec = SequenceFeedbackSpec(
        enabled=True,
        mode="lag_replay",
        measurement_geometry="directional_horizontal",
        min_window_size=1,
        min_peak_probability=0.0,
        min_horizontal_eigenvalue_ratio=1.0,
        min_projected_std_m=1.0,
        max_horizontal_std_m=1.0e6,
        max_correction_norm_m=1.0e6,
        covariance_inflation=4.0,
        nis_threshold=None,
        trust_gate_source="heuristic",
    )
    ctrl = SequenceFeedbackController(spec)
    result = ctrl.evaluate(
        update,
        ErrorStateINS(_make_state(time_s=50.0).copy()),
        current_time_s=53.0,
        live_ins_state=_make_state(time_s=53.0),
        previous_live_ins_state=_make_state(
            time_s=52.0, north_speed_mps=1.0, east_speed_mps=0.2
        ),
    )

    assert result.applied is False
    assert result.diagnostics.heuristic_allowed is False
    assert result.diagnostics.heuristic_rejection_reason is not None
    assert "projected_std" in result.diagnostics.heuristic_rejection_reason
