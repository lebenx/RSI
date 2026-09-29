# Qualitative controlled examples

These examples are from the held-out `scaling_benchmark` test split. Reward and optimal-action labels are used only after the decision.

## Wrong premise, harmful flip

Task `spb-hid-h-20261915` has a hidden false premise. At duplication 1, flat aggregation selects `take_safe_route`, reward `+1.041`, and is optimal. Repeating the same descendants twice raises the root confidence from `0.754` to `0.821`; the action flips to `take_short_route`, whose true reward is `-1.940`. At duplication 64, confidence reaches `1.000` and the harmful action remains. Provenance-preserving aggregation keeps confidence `0.754`, the safe action, and reward `+1.041` at every budget.

| duplication | flat confidence | flat action | flat reward | provenance confidence | provenance action | provenance reward |
|---:|---:|---|---:|---:|---|---:|
| 1 | 0.754 | take_safe_route | 1.041 | 0.754 | take_safe_route | 1.041 |
| 2 | 0.821 | take_short_route | -1.940 | 0.754 | take_safe_route | 1.041 |
| 4 | 0.912 | take_short_route | -1.940 | 0.754 | take_safe_route | 1.041 |
| 8 | 0.981 | take_short_route | -1.940 | 0.754 | take_safe_route | 1.041 |
| 64 | 1.000 | take_short_route | -1.940 | 0.754 | take_safe_route | 1.041 |

The result is a mechanism example, not a claim about the local or API model's default behavior.
