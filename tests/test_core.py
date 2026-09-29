import numpy as np
from experiments.core import *

def test_information_boundaries_and_method_equivalence():
    s=scenario(7, physical_state=0, hypothesis_direction=1, prior=.3, strength=2., family=0)
    # Same q/evidence and same conditional bank across all interventions.
    q1, d1=intervention(s,'pure_duplication',1)
    q8, d8=intervention(s,'pure_duplication',8)
    assert q1==q8
    p1,v1=predict(q1,d1,'provenance')
    p8,v8=predict(q8,d8,'provenance')
    assert abs(p1-p8)<1e-12 and np.max(abs(v1-v8))<1e-12
    b1,w1=predict(q1,d1,'belief_mixing')
    assert abs(p1-b1)<1e-12 and np.max(abs(v1-w1))<1e-12
    # Independent samples refine conditional values but do not update q.
    qi,di1=intervention(s,'independent_rollout',1)
    _,di8=intervention(s,'independent_rollout',8)
    assert qi==q1
    _,vi1=predict(qi,di1,'provenance')
    _,vi8=predict(qi,di8,'provenance')
    assert np.max(abs(vi1-vi8))>1e-6
    # Real evidence is the only intervention that changes q.
    qreal1,_=intervention(s,'new_real_evidence',1)
    qreal8,_=intervention(s,'new_real_evidence',8)
    assert qreal1 != qreal8

def test_dedup_variants_and_equal_length_semantics():
    s=scenario(2, physical_state=1, hypothesis_direction=0, prior=.7, strength=1.25, family=1)
    q,plain=intervention(s,'pure_duplication',4)
    _,para=intervention(s,'paraphrase_duplication',4)
    assert len({r.sid for r in plain}) < len(plain)
    assert len({r.text for r in para}) > len({r.text for r in plain})
    assert predict(q,para,'semantic_dedup')[0] == predict(q,plain,'semantic_dedup')[0]
    assert len({len(r.text) for r in para}) > 0

def test_selective_duplication_is_not_silently_invariant_for_flat():
    s=scenario(3, physical_state=1, hypothesis_direction=0, prior=.3, strength=2., family=0)
    q1,r1=intervention(s,'selective_duplication',1)
    q8,r8=intervention(s,'selective_duplication',8)
    p1,_=predict(q1,r1,'flat'); p8,_=predict(q8,r8,'flat')
    assert p1 != p8
