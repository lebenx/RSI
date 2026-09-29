# Numeric controlled-grid report

This report uses 3,600 independent scenario-seed cells (30 seeds × 2 physical states × 2 hypothesis directions × 5 priors × 3 likelihood strengths × 2 utility families), 5 information conditions, 5 multiplicities and 7 methods. Confidence intervals are seed-clustered 95% Student-t intervals. The analytical methods use oracle provenance IDs; they are mechanism diagnostics, not evidence that an LLM internally applies the stated estimator.

## Wrong root, pure duplication

| multiplicity | method | root_confidence_mean | root_confidence_ci95_low | root_confidence_ci95_high | action_flip_mean | reward_mean | expected_reward_mean | value_rmse_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | belief_mixing | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 1 | exact_dedup | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 1 | flat | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 1 | provenance | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 1 | semantic_dedup | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 1 | source_average | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 1 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.000 | 0.310 | 0.440 | 0.906 |
| 2 | belief_mixing | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 2 | exact_dedup | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 2 | flat | 0.563 | 0.557 | 0.569 | 0.098 | 0.150 | 0.687 | 0.826 |
| 2 | provenance | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 2 | semantic_dedup | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 2 | source_average | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 2 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.000 | 0.310 | 0.440 | 0.906 |
| 4 | belief_mixing | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 4 | exact_dedup | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 4 | flat | 0.764 | 0.761 | 0.768 | 0.268 | -0.272 | 0.437 | 1.125 |
| 4 | provenance | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 4 | semantic_dedup | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 4 | source_average | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 4 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.000 | 0.310 | 0.440 | 0.906 |
| 8 | belief_mixing | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 8 | exact_dedup | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 8 | flat | 0.905 | 0.904 | 0.906 | 0.361 | -0.505 | 0.291 | 1.381 |
| 8 | provenance | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 8 | semantic_dedup | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 8 | source_average | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 8 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.000 | 0.310 | 0.440 | 0.906 |
| 16 | belief_mixing | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 16 | exact_dedup | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 16 | flat | 0.975 | 0.974 | 0.975 | 0.412 | -0.639 | 0.200 | 1.520 |
| 16 | provenance | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 16 | semantic_dedup | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 16 | source_average | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 16 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.000 | 0.310 | 0.440 | 0.906 |

## Correct root, independent rollout

| multiplicity | method | root_confidence_mean | root_confidence_ci95_low | root_confidence_ci95_high | action_flip_mean | reward_mean | expected_reward_mean | value_rmse_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | belief_mixing | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 1 | exact_dedup | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 1 | flat | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 1 | provenance | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 1 | semantic_dedup | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 1 | source_average | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 1 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.000 | 0.652 | 0.465 | 0.921 |
| 2 | belief_mixing | 0.555 | 0.548 | 0.563 | 0.122 | 0.730 | 0.850 | 0.541 |
| 2 | exact_dedup | 0.555 | 0.548 | 0.563 | 0.122 | 0.730 | 0.850 | 0.541 |
| 2 | flat | 0.662 | 0.656 | 0.669 | 0.161 | 0.908 | 0.807 | 0.618 |
| 2 | provenance | 0.555 | 0.548 | 0.563 | 0.122 | 0.730 | 0.850 | 0.541 |
| 2 | semantic_dedup | 0.555 | 0.548 | 0.563 | 0.122 | 0.730 | 0.850 | 0.541 |
| 2 | source_average | 0.555 | 0.548 | 0.563 | 0.122 | 0.730 | 0.850 | 0.541 |
| 2 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.177 | 0.640 | 0.489 | 0.787 |
| 4 | belief_mixing | 0.555 | 0.548 | 0.563 | 0.143 | 0.728 | 0.892 | 0.390 |
| 4 | exact_dedup | 0.555 | 0.548 | 0.563 | 0.143 | 0.728 | 0.892 | 0.390 |
| 4 | flat | 0.802 | 0.799 | 0.806 | 0.258 | 1.164 | 0.677 | 0.726 |
| 4 | provenance | 0.555 | 0.548 | 0.563 | 0.143 | 0.728 | 0.892 | 0.390 |
| 4 | semantic_dedup | 0.555 | 0.548 | 0.563 | 0.143 | 0.728 | 0.892 | 0.390 |
| 4 | source_average | 0.555 | 0.548 | 0.563 | 0.143 | 0.728 | 0.892 | 0.390 |
| 4 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.236 | 0.676 | 0.528 | 0.700 |
| 8 | belief_mixing | 0.555 | 0.548 | 0.563 | 0.161 | 0.731 | 0.911 | 0.274 |
| 8 | exact_dedup | 0.555 | 0.548 | 0.563 | 0.161 | 0.731 | 0.911 | 0.274 |
| 8 | flat | 0.910 | 0.909 | 0.911 | 0.313 | 1.251 | 0.606 | 0.858 |
| 8 | provenance | 0.555 | 0.548 | 0.563 | 0.161 | 0.731 | 0.911 | 0.274 |
| 8 | semantic_dedup | 0.555 | 0.548 | 0.563 | 0.161 | 0.731 | 0.911 | 0.274 |
| 8 | source_average | 0.555 | 0.548 | 0.563 | 0.161 | 0.731 | 0.911 | 0.274 |
| 8 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.249 | 0.630 | 0.538 | 0.650 |
| 16 | belief_mixing | 0.555 | 0.548 | 0.563 | 0.164 | 0.736 | 0.918 | 0.193 |
| 16 | exact_dedup | 0.555 | 0.548 | 0.563 | 0.164 | 0.736 | 0.918 | 0.193 |
| 16 | flat | 0.976 | 0.975 | 0.976 | 0.353 | 1.356 | 0.519 | 0.960 |
| 16 | provenance | 0.555 | 0.548 | 0.563 | 0.164 | 0.736 | 0.918 | 0.193 |
| 16 | semantic_dedup | 0.555 | 0.548 | 0.563 | 0.164 | 0.736 | 0.918 | 0.193 |
| 16 | source_average | 0.555 | 0.548 | 0.563 | 0.164 | 0.736 | 0.918 | 0.193 |
| 16 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.257 | 0.682 | 0.566 | 0.624 |

## Wrong root, real new evidence

| multiplicity | method | root_confidence_mean | root_confidence_ci95_low | root_confidence_ci95_high | action_flip_mean | reward_mean | expected_reward_mean | value_rmse_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | belief_mixing | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 1 | exact_dedup | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 1 | flat | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 1 | provenance | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 1 | semantic_dedup | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 1 | source_average | 0.442 | 0.437 | 0.448 | 0.000 | 0.395 | 0.758 | 0.771 |
| 1 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.000 | 0.310 | 0.440 | 0.906 |
| 2 | belief_mixing | 0.401 | 0.393 | 0.408 | 0.086 | 0.477 | 0.793 | 0.783 |
| 2 | exact_dedup | 0.401 | 0.393 | 0.408 | 0.086 | 0.477 | 0.793 | 0.783 |
| 2 | flat | 0.401 | 0.393 | 0.408 | 0.086 | 0.477 | 0.793 | 0.783 |
| 2 | provenance | 0.401 | 0.393 | 0.408 | 0.086 | 0.477 | 0.793 | 0.783 |
| 2 | semantic_dedup | 0.401 | 0.393 | 0.408 | 0.086 | 0.477 | 0.793 | 0.783 |
| 2 | source_average | 0.401 | 0.393 | 0.408 | 0.086 | 0.477 | 0.793 | 0.783 |
| 2 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.000 | 0.310 | 0.434 | 0.936 |
| 4 | belief_mixing | 0.331 | 0.322 | 0.339 | 0.122 | 0.590 | 0.827 | 0.805 |
| 4 | exact_dedup | 0.331 | 0.322 | 0.339 | 0.122 | 0.590 | 0.827 | 0.805 |
| 4 | flat | 0.331 | 0.322 | 0.339 | 0.122 | 0.590 | 0.827 | 0.805 |
| 4 | provenance | 0.331 | 0.322 | 0.339 | 0.122 | 0.590 | 0.827 | 0.805 |
| 4 | semantic_dedup | 0.331 | 0.322 | 0.339 | 0.122 | 0.590 | 0.827 | 0.805 |
| 4 | source_average | 0.331 | 0.322 | 0.339 | 0.122 | 0.590 | 0.827 | 0.805 |
| 4 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.000 | 0.310 | 0.423 | 0.974 |
| 8 | belief_mixing | 0.262 | 0.253 | 0.271 | 0.150 | 0.654 | 0.858 | 0.836 |
| 8 | exact_dedup | 0.262 | 0.253 | 0.271 | 0.150 | 0.654 | 0.858 | 0.836 |
| 8 | flat | 0.262 | 0.253 | 0.271 | 0.150 | 0.654 | 0.858 | 0.836 |
| 8 | provenance | 0.262 | 0.253 | 0.271 | 0.150 | 0.654 | 0.858 | 0.836 |
| 8 | semantic_dedup | 0.262 | 0.253 | 0.271 | 0.150 | 0.654 | 0.858 | 0.836 |
| 8 | source_average | 0.262 | 0.253 | 0.271 | 0.150 | 0.654 | 0.858 | 0.836 |
| 8 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.000 | 0.310 | 0.410 | 1.029 |
| 16 | belief_mixing | 0.200 | 0.191 | 0.209 | 0.183 | 0.751 | 0.892 | 0.860 |
| 16 | exact_dedup | 0.200 | 0.191 | 0.209 | 0.183 | 0.751 | 0.892 | 0.860 |
| 16 | flat | 0.200 | 0.191 | 0.209 | 0.183 | 0.751 | 0.892 | 0.860 |
| 16 | provenance | 0.200 | 0.191 | 0.209 | 0.183 | 0.751 | 0.892 | 0.860 |
| 16 | semantic_dedup | 0.200 | 0.191 | 0.209 | 0.183 | 0.751 | 0.892 | 0.860 |
| 16 | source_average | 0.200 | 0.191 | 0.209 | 0.183 | 0.751 | 0.892 | 0.860 |
| 16 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.000 | 0.310 | 0.378 | 1.080 |

## Correct root, paraphrase duplication

| multiplicity | method | root_confidence_mean | root_confidence_ci95_low | root_confidence_ci95_high | action_flip_mean | reward_mean | expected_reward_mean | value_rmse_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | belief_mixing | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 1 | exact_dedup | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 1 | flat | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 1 | provenance | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 1 | semantic_dedup | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 1 | source_average | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 1 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.000 | 0.652 | 0.465 | 0.921 |
| 2 | belief_mixing | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 2 | exact_dedup | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 2 | flat | 0.662 | 0.656 | 0.669 | 0.081 | 0.871 | 0.738 | 0.837 |
| 2 | provenance | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 2 | semantic_dedup | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 2 | source_average | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 2 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.000 | 0.652 | 0.465 | 0.921 |
| 4 | belief_mixing | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 4 | exact_dedup | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 4 | flat | 0.802 | 0.799 | 0.806 | 0.178 | 1.049 | 0.614 | 1.034 |
| 4 | provenance | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 4 | semantic_dedup | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 4 | source_average | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 4 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.000 | 0.652 | 0.465 | 0.921 |
| 8 | belief_mixing | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 8 | exact_dedup | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 8 | flat | 0.910 | 0.909 | 0.911 | 0.253 | 1.181 | 0.500 | 1.223 |
| 8 | provenance | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 8 | semantic_dedup | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 8 | source_average | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 8 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.000 | 0.652 | 0.465 | 0.921 |
| 16 | belief_mixing | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 16 | exact_dedup | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 16 | flat | 0.976 | 0.975 | 0.976 | 0.297 | 1.260 | 0.430 | 1.351 |
| 16 | provenance | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 16 | semantic_dedup | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 16 | source_average | 0.555 | 0.548 | 0.563 | 0.000 | 0.708 | 0.788 | 0.774 |
| 16 | trajectory_average | 0.500 | 0.500 | 0.500 | 0.000 | 0.652 | 0.465 | 0.921 |

## Wrong-root flat effect stratified by prior and likelihood strength (m=16)

| prior | strength | family | root_confidence_mean | reward_mean |
| --- | --- | --- | --- | --- |
| 0.100 | 1.250 | 0 | 0.746 | -1.386 |
| 0.100 | 1.250 | 1 | 0.763 | 0.103 |
| 0.100 | 4.000 | 0 | 1.000 | -1.380 |
| 0.100 | 4.000 | 1 | 1.000 | -0.050 |
| 0.500 | 1.250 | 0 | 0.964 | -1.596 |
| 0.500 | 1.250 | 1 | 0.963 | -0.213 |
| 0.500 | 4.000 | 0 | 1.000 | -1.298 |
| 0.500 | 4.000 | 1 | 1.000 | 0.348 |
| 0.900 | 1.250 | 0 | 0.996 | -1.500 |
| 0.900 | 1.250 | 1 | 0.996 | -0.063 |
| 0.900 | 4.000 | 0 | 1.000 | -1.380 |
| 0.900 | 4.000 | 1 | 1.000 | 0.186 |
| 0.100 | 1.250 | 0 | 0.763 | -0.934 |
| 0.100 | 1.250 | 1 | 0.752 | 0.302 |
| 0.100 | 4.000 | 0 | 1.000 | -1.532 |
| 0.100 | 4.000 | 1 | 1.000 | 0.059 |
| 0.500 | 1.250 | 0 | 0.965 | -1.361 |
| 0.500 | 1.250 | 1 | 0.964 | 0.272 |
| 0.500 | 4.000 | 0 | 1.000 | -1.387 |
| 0.500 | 4.000 | 1 | 1.000 | -0.056 |
| 0.900 | 1.250 | 0 | 0.996 | -1.347 |
| 0.900 | 1.250 | 1 | 0.996 | 0.090 |
| 0.900 | 4.000 | 0 | 1.000 | -1.455 |
| 0.900 | 4.000 | 1 | 1.000 | 0.355 |

## Exact audits

- `belief_mixing` and `provenance` differ by at most 0.0 in the numeric grid.
- Pure-duplication provenance root confidence has zero spread across m within every scenario.
- Independent rollout leaves q unchanged for mixing/provenance but lowers conditional value RMSE and regret.
- New real evidence changes q; rollout duplication alone does not.
- `numeric_raw.csv` includes action flips, harmful/helpful flips, reward, expected reward, Brier, NLL, value RMSE and paired m=16−m=1 deltas.

Plots: [root confidence](numeric_root_confidence.png), [reward](numeric_reward.png).

Limitations: synthetic binary environment, oracle provenance for analytic methods, and no natural-task or cross-model claim. The separate DeepSeek audit is reported under `llm_subset/` once complete.
