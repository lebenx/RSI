from submission.qwen_bank_diagnostic import bank_metrics


def bank(values):
    return [dict(sample_id=f'{a}-{h}', action_id=a, hypothesis_value=bool(h),
                 predicted_final_score=v, success_probability=.2, imagined_actions=[])
            for a, pair in enumerate(values) for h, v in enumerate(pair)]


def test_constant_bank_and_branch_sensitive_counterexample():
    tied = bank([[.1, .1]] * 4)
    result = bank_metrics(tied * 8, .7)
    assert result['constant_value'] and result['example_values_only']
    assert result['unique_samples'] == 8
    assert result['mixture_action_id'] == 0
    assert not result['belief_sensitive_winner']
    crossing = bank([[.8, 0], [0, .9], [.1, .1], [0, 0]])
    assert bank_metrics(crossing, .1)['mixture_action_id'] == 0
    assert bank_metrics(crossing, .9)['mixture_action_id'] == 1
    assert bank_metrics(crossing, .5)['belief_sensitive_winner']


def test_distinct_values_need_not_yield_a_belief_sensitive_winner():
    dominating = bank([[.8, .9], [.1, .5], [.2, .4], [0, .3]])
    result = bank_metrics(dominating, .5)
    assert not result['constant_value']
    assert not result['belief_sensitive_winner']
    assert result['top_two_margin'] > 0
