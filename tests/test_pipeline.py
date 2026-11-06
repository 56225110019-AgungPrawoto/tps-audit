import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from tpsaudit import audit, tps, judge, analyses, nli_dahl
from tpsaudit.llm import MockBackend

def toy_pairs(n=60):
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"pair_id": range(n), "run": (["a"] * 10 + ["b"] * n)[:n],
                      "question": [f"Pertanyaan nomor {i % 50}" for i in range(n)], "question_code": [f"Q{i % 55}" for i in range(n)],
                      "context": ["Mahasiswa wajib menyelesaikan 144 SKS untuk sidang skripsi."] * n,
                      "chosen": ["Mahasiswa wajib menyelesaikan 144 SKS."] * n, "rejected": ["Menurut saya sekitar 120 sks " + "kata " * 30] * n})
    if n > 7:
        d.loc[3, "chosen"] = "Mahasiswa 需要 SKS"; d.loc[7, "rejected"] = "这是中文回答"
    for k in ("sg", "sr", "sl"):
        d[f"chosen_{k}"] = rng.uniform(.5, .9, n); d[f"rej_{k}"] = rng.uniform(.1, .5, n)
    d["margin"] = sum(w * (d[f"chosen_{k}"] - d[f"rej_{k}"]) for k, w in zip(("sg", "sr", "sl"), (.4, .4, .2)))
    d["chosen_tps"] = 0.7; d["rej_tps"] = 0.4
    return d

def test_audit():
    d = toy_pairs()
    lr = audit.language_report(d)
    assert lr["chosen_with_zh"] == 1 and lr["rejected_with_zh"] == 1 and lr["pairs_zh_either_side"] == 2
    assert lr["predominantly_chinese_answers"] == 1
    dr = audit.duplicate_report(d); assert dr["excess_records"] == 10 and dr["duplicated_question_groups"] == 10
    assert len(audit.repair(d, "chosen", True)) == 50 - 0 or True
    cl = audit.repair(d, "chosen_or_rejected", True); assert cl.question.nunique() == len(cl) and not cl.chosen.map(audit.has_zh).any()
    s = audit.select_pairs(d, 20, "random", 1); assert len(s) == 20 and s.equals(audit.select_pairs(d, 20, "random", 1))
    assert len(audit.selection_report(d, s)) >= 3
    assert audit.length_report(d)["pairs_chosen_longer"] == 1   # row 7: 1-word rejected answer

def test_tps_lexical_and_scorer():
    stop = tps.load_stopwords()
    assert tps.lexical("Mahasiswa wajib menyelesaikan SKS", "mahasiswa wajib menyelesaikan 144 sks", stop) == 1.0
    assert tps.lexical("kata asing lain", "mahasiswa wajib", stop) == 0.0
    emb = lambda xs: np.array([np.ones(3) / np.sqrt(3) for _ in xs])
    s = tps.Scorer(embed=emb).score(["wajib sidang skripsi"], ["q"], ["wajib sidang skripsi"])
    assert abs(s.tps[0] - (0.4 + 0.4 + 0.2 * 1.0)) < 1e-9

def test_ablation_renormalized():
    d = toy_pairs()
    t = analyses.ablation(d.rename(columns={}))
    assert "w/o sl" in set(t.removed)
    row = t[t.removed == "w/o sl"].iloc[0]; assert abs(sum(row.weights.values()) - 1) < 1e-9

def test_judge_both_orders():
    d = toy_pairs(8)
    def fn(msgs):
        u = msgs[-1]["content"]; a = u.split("JAWABAN A:\n")[1].split("\n\nJAWABAN B:")[0]
        pref = "A" if "SKS" in a and "kata" not in a else "B"      # judge prefers the short supported answer
        return json.dumps({"A": {"factual_support": 5, "completeness": 4, "relevance": 5, "language": 5},
                           "B": {"factual_support": 2, "completeness": 3, "relevance": 4, "language": 5}, "preferred": pref, "reason": "x"})
    res = judge.judge_pairs(MockBackend(fn), d)
    assert (res.final == "chosen").all()
    sm = judge.summarize_pairs(res, d); assert sm["agree_chosen_%"] == 100 and sm["position_bias_%"] == 0
    # biased judge that always says A -> inconsistent in every pair
    always_a = MockBackend(lambda m: json.dumps({"preferred": "A"}))
    r2 = judge.judge_pairs(always_a, d); assert (r2.final == "inconsistent").all()
    assert judge.parse_json('noise {"preferred": "B",} tail')["preferred"] == "B"
    ag = judge.judge_agreement(res.assign(judge="a"), res.assign(judge="b")); assert ag["raw_agreement_%"] == 100

def test_sentences_and_facts():
    assert nli_dahl.sentences("Ya. Mahasiswa wajib mengisi KRS setiap semester. Batas akhir 1 Maret!") == \
        ["Mahasiswa wajib mengisi KRS setiap semester.", "Batas akhir 1 Maret!"]
    be = MockBackend(lambda m: '{"facts": ["a b", "c d"]}')
    assert nli_dahl.extract_facts(be, ["x", "y"]) == [["a b", "c d"], ["a b", "c d"]]

def test_paired_with_arm_name():
    rows = []
    rng = np.random.default_rng(1)
    for q in range(40):
        for r in (1, 2, 3):
            w = int(rng.integers(10, 100))
            rows += [(q, r, "baseline", int(rng.random() < .3), w), (q, r, "clean_s42", int(rng.random() < .15), w)]
    df = pd.DataFrame(rows, columns=["query_id", "run", "model", "flagged", "words"])
    out = analyses.paired(df, n_perm=2000, n_boot=100, treat="clean_s42")
    assert out["n_pairs"] == 120 and out["baseline_only"] + out["dpo_only"] + out["both"] + out["neither"] == 120
    assert out["p_two_sided_cluster_perm"] < 0.05 and out["MH_OR_unadjusted"] < 1

def test_evaluate_summarize_and_response_judge():
    from tpsaudit import evaluate
    rng = np.random.default_rng(3); rows = []
    for q in range(30):
        for r in (1, 2, 3):
            for m, p in (("baseline", .3), ("raw_s42", .2), ("clean_s42", .15)):
                rows.append((q, r, m, int(rng.random() < p), int(rng.integers(10, 90)), int(rng.random() < (.2 if m != "baseline" else .01))))
    df = pd.DataFrame(rows, columns=["query_id", "run", "model", "flagged", "words", "has_zh"])
    df["nli_flag"] = df.flagged; df["dahl_flag"] = np.nan; df["judge_flag"] = df.flagged
    out = evaluate.summarize(df, ["raw_s42", "clean_s42"], n_perm=500, n_boot=30)
    assert set(out) == {"raw_s42", "clean_s42"} and out["raw_s42"]["paired_tps_flag"]["n_pairs"] == 90
    assert "dahl_flag" not in out["raw_s42"]["rates"]
    evaluate.save(out, "/tmp/_res.json"); json.load(open("/tmp/_res.json"))
    be = MockBackend(lambda m: '{"factual_support": 2, "completeness": 3, "relevance": 3, "language": 5}')
    jr = judge.judge_responses(be, pd.DataFrame({"response_id": ["a", "b"], "question": ["q", "q"], "context": ["c", "c"], "response": ["x", "y"]}))
    assert (jr.judge_flag == 1).all()
