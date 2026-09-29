from submission.alfworld_interactive_runner import flat_value, grouped_value


def _candidate():
    return {
        "p_true": 0.5,
        "actions": [{"id": i, "action": f"a{i}"} for i in range(4)],
    }


def _bank():
    rows = []
    for action in range(4):
        for hypothesis in (False, True):
            score = float(hypothesis) if action == 0 else 0.55
            rows.append({"action_id": action, "hypothesis_value": hypothesis,
                         "predicted_final_score": score,
                         "success_probability": score,
                         "sample_id": f"s-a{action}-h{int(hypothesis)}"})
    return rows


def test_flat_value_duplicate_can_flip_but_grouped_value_is_invariant():
    candidate = _candidate()
    bank = _bank()
    grouped = grouped_value(candidate, bank)
    flat_m1 = flat_value(candidate, bank, 1)
    flat_m4 = flat_value(candidate, bank + [r for r in bank if r["hypothesis_value"]], 4)
    assert grouped["action_id"] == 1
    assert flat_m1["action_id"] == 1
    assert flat_m4["action_id"] == 0
    assert grouped["p_true"] == 0.5
    assert flat_m4["p_true"] > flat_m1["p_true"]
