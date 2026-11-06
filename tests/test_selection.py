import pandas as pd
from tpsaudit import selection as S

def fake(df_lens):
    return pd.DataFrame(df_lens, columns=["n_prompt","n_chosen","n_rejected"])

def test_steps_roundtrip():
    assert S.expected_steps(234)==59 and S.expected_steps(693)==174
    assert S.steps_to_examples(59)==(233,236)

def test_calibrate_flags_matching_rule():
    d=fake([(900,50,50),(1100,10,10),(1000,30,10),(500,600,10)])
    r=S.calibrate(d,1024,target=1).set_index("rule")
    assert r.loc["prompt < L","survivors"]==3
    assert r.loc["prompt + chosen <= L","survivors"]==1 and r.loc["prompt + chosen <= L","matches_log"]

def test_truncation_profile():
    d=fake([(900,50,200),(1100,10,10)])
    p=S.truncation_profile(d,d.n_prompt<1024)
    assert p["n"]==1 and p["rejected_cut"]==1 and p["chosen_cut"]==0
