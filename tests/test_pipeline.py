import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from tpsaudit import audit, tps, judge, analyses, nli_dahl, data_io, train_dpo, evaluate
from tpsaudit.llm import MockBackend

def stage2_csv(tmp_path, n=40):
    """A tiny CSV with the ORIGINAL Stage 2 column names."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rows.append({"kode_pertanyaan": f"Q{i % 30:03d}", "topik": "t", "pertanyaan": f"Pertanyaan nomor {i % 35}", "sumber_file": "a.pdf",
                     "context_retrieved": "Mahasiswa wajib menyelesaikan 144 SKS.", "sources_summary": "x", "chosen": "Mahasiswa wajib menyelesaikan 144 SKS.",
                     "score_chosen": 0.7, "sem_grounding_chosen": 0.6, "sem_relevance_chosen": 0.8, "lexical_chosen": 0.65, "temp_chosen": 0.4, "label_chosen": "chosen_conservative",
                     "rejected": "Menurut saya sekitar 120 sks " + "kata " * 30, "score_rejected": 0.45, "sem_grounding_rejected": 0.44, "sem_relevance_rejected": 0.65,
                     "lexical_rejected": 0.15, "temp_rejected": 1.8, "score_margin": 0.25, "timestamp": f"2026-10-0{1 + i // 20}T10:{i % 60:02d}:00"})
    rows[3]["chosen"] = "Mahasiswa 需要 SKS"; rows[7]["rejected"] = "这是中文回答"
    p = tmp_path / "dpo.csv"; pd.DataFrame(rows).to_csv(p, index=False); return p

def test_load_stage2_pairs_and_audit(tmp_path):
    d = data_io.load_stage2_pairs(stage2_csv(tmp_path), initial_run_n_pairs=10)
    assert {"question", "context", "chosen_sg", "rej_tps", "margin", "pair_id", "run", "question_code"} <= set(d.columns)
    assert d.pair_id.is_unique and (d.run[:10] == "initial").all() and (d.run[10:] == "extended").all()
    assert d.margin.dtype.kind == "f" and data_io.run_boundary_report(d)["initial"][0].startswith("2026-10-01")
    lr = audit.language_report(d); assert lr["chosen_with_zh"] == 1 and lr["rejected_with_zh"] == 1 and lr["predominantly_chinese_answers"] == 1
    dr = audit.duplicate_report(d); assert dr["excess_records"] == 5 and dr["codes_shared_by_both_runs"] > 0
    assert audit.length_report(d)["pairs_chosen_longer"] == 1

def test_selection_and_minimal_clean(tmp_path):
    d = data_io.load_stage2_pairs(stage2_csv(tmp_path), 10)
    sub = audit.select_pairs(d, 20, "first_n"); assert list(sub.pair_id) == list(d.pair_id[:20])
    assert audit.select_pairs(d, 20, "random", 1).equals(audit.select_pairs(d, 20, "random", 1))
    clean, removed = audit.minimal_clean(sub)
    assert len(clean) + len(removed) == 20 and set(removed.reason) == {"chinese", "duplicate_question"} or set(removed.reason) == {"chinese"}
    assert not clean.chosen.map(audit.has_zh).any() and not clean.rejected.map(audit.has_zh).any() and clean.question.nunique() == len(clean)
    assert len(audit.selection_report(d, sub)) >= 3

def test_training_prep_matches_stage3(tmp_path):
    import yaml
    prompts = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "prompts", "generation_prompts.yaml"), encoding="utf8"))
    d = data_io.load_stage2_pairs(stage2_csv(tmp_path), 10); d.loc[1, "chosen"] = "   "
    f = train_dpo.prepare_training_frame(d); assert len(f) == len(d) - 1
    rec = train_dpo.to_trl_conversational(f.iloc[0].to_dict(), prompts)
    assert rec["prompt"][0]["content"] == "Jawab singkat & padat (min 3 kalimat) sesuai konteks yang diberikan"
    assert rec["prompt"][1]["content"].startswith("Konteks:\nMahasiswa") and rec["prompt"][1]["content"].endswith("Pertanyaan: Pertanyaan nomor 0")
    assert rec["chosen"][0]["role"] == "assistant"
    assert train_dpo.expected_steps(234, 1, 4) == 59                         # the 59 optimizer steps reported in the paper

def test_ablation_variants():
    rng = np.random.default_rng(0); n = 100
    d = pd.DataFrame({f"{p}_{k}": rng.uniform(a, b, n) for p, (a, b) in (("chosen", (.5, .9)), ("rej", (.1, .5))) for k in ("sg", "sr", "sl")})
    t = analyses.ablation(d)
    assert set(t.variant) == {"full", "proportional", "equal_split", "unadjusted"} and len(t) == 10
    assert abs(sum(t[(t.removed == "w/o sl") & (t.variant == "proportional")].weights.iloc[0].values()) - 1) < 1e-2
    assert t[(t.removed == "w/o sl") & (t.variant == "equal_split")].weights.iloc[0] == {"sg": 0.5, "sr": 0.5}

def test_judge_both_orders():
    d = pd.DataFrame({"pair_id": range(8), "run": "a", "margin": .2, "question": "q", "context": "c", "chosen": "Mahasiswa wajib 144 SKS.", "rejected": "kata " * 30})
    def fn(msgs):
        u = msgs[-1]["content"]; a = u.split("JAWABAN A:\n")[1].split("\n\nJAWABAN B:")[0]
        return json.dumps({"A": {"factual_support": 5}, "B": {"factual_support": 2}, "preferred": "A" if "SKS" in a else "B", "reason": "x"})
    res = judge.judge_pairs(MockBackend(fn), d); assert (res.final == "chosen").all()
    sm = judge.summarize_pairs(res, d); assert sm["agree_chosen_%"] == 100 and sm["position_bias_%"] == 0
    assert (judge.judge_pairs(MockBackend(lambda m: '{"preferred": "A"}'), d).final == "inconsistent").all()
    assert judge.parse_json('noise {"preferred": "B",} tail')["preferred"] == "B"

def test_nli_flags_both_modes():
    seen = []
    def pipe(batch):
        seen.extend(batch); out = []
        for b in batch:
            txt = b if isinstance(b, str) else b["text_pair"]
            hi = "didukung" in txt
            out.append([{"label": "entailment", "score": .9 if hi else .1}, {"label": "neutral", "score": .05}, {"label": "contradiction", "score": .05 if hi else .85}])
        return out
    items = [("konteks panjang " * 50, "jawaban didukung konteks " * 3), ("konteks", "jawaban lain sekali ini tidak ada dukungan"), ("", "abc def ghi jkl"), ("konteks", "")]
    o = nli_dahl.nli_flags(pipe, items, mode="original"); assert [x["nli_hallucination"] for x in o] == [False, True, True, True]
    assert o[2]["nli_label"] == "UNKNOWN" and isinstance(seen[0], str) and "</s>" in seen[0] and len(seen[0].split(" </s> ")[0]) <= 512
    seen.clear(); p = nli_dahl.nli_flags(pipe, items, mode="pair"); assert isinstance(seen[0], dict) and [x["nli_hallucination"] for x in p] == [False, True, True, True]
    assert len(seen[0]["text"]) <= 512 and len(seen[0]["text_pair"]) <= 256

def test_atomic_helpers_and_flag():
    assert nli_dahl.is_noncommittal("Maaf, saya tidak tahu") and not nli_dahl.is_noncommittal("Batas akhir 1 Maret")
    u = ["Mahasiswa wajib mengisi KRS setiap semester", "Mahasiswa wajib mengisi KRS setiap semester ya", "Batas cuti akademik dua semester"]
    assert nli_dahl.dedup_atomic_units(u, 0.85) == [u[0], u[2]]
    assert nli_dahl.dahl_flag("x", None) and nli_dahl.dahl_flag("", 1.0) and nli_dahl.dahl_flag("x", .49) and not nli_dahl.dahl_flag("x", .5)

def test_existing_runs_roundtrip_and_flag_consistency(tmp_path):
    rows = []
    for e in (1, 2):
        for lab in ("Baseline (pre-DPO)", "DPO Model (post-DPO)"):
            for i in range(5):
                sc = [.7, .4, .6, .3, .55][i]; d1 = [.8, .5, .3, .6, .7][i]; d2 = [.5, .4, .5, .1, .6][i]
                f, _ = tps.hallucination_flag("jawaban ok", sc, {"dim1_relevance": d1, "dim2_grounding": d2}, ["f.png"])
                rows.append({"kode": f"Q{i}", "model_label": lab, "response": "jawaban ok", "filenames": "f.png", "dahl_score": sc, "dim1_relevance": d1,
                             "dim2_grounding": d2, "dim3_lexical": .3, "hallucination_flag": f})
    for e in (1, 2):
        pd.DataFrame([r for r in rows[(e - 1) * 10:(e - 1) * 10 + 10]]).to_csv(tmp_path / f"evaluation_per_sample_epoch{e}.csv", index=False)
    ex = data_io.load_existing_runs(str(tmp_path), epochs=(1, 2)); assert set(ex.model) == {"baseline", "dpo_orig"} and set(ex.run) == {1, 2}
    st = evaluate.standardize_existing(ex); assert st.groupby(["model", "run"]).query_id.apply(lambda s: list(s) == list(range(5))).all()
    assert evaluate.recheck_flags(st)["mismatches"] == 0
    fin = evaluate.finish(st); assert {"flagged", "words", "has_zh", "flag_score_lt_050"} <= set(fin.columns)

def test_paired_and_summarize():
    rng = np.random.default_rng(3); rows = []
    for q in range(30):
        for r in (1, 2, 3):
            for m, p in (("baseline", .3), ("dpo_orig", .2), ("clean_minimal", .15)):
                rows.append((q, r, m, int(rng.integers(10, 90)), int(rng.random() < p), int(rng.random() < (.2 if m != "baseline" else .01))))
    df = pd.DataFrame(rows, columns=["query_id", "run", "model", "words", "flagged", "has_zh"]); df["zh_share"] = df.has_zh * .6
    df["flag_score_lt_050"] = df.flagged
    out = evaluate.summarize(df, ["dpo_orig", "clean_minimal"], n_perm=500, n_boot=30)
    assert out["dpo_orig"]["paired_flag"]["n_pairs"] == 90 and out["clean_minimal"]["language"]["arm_zh"] == int(df[df.model == "clean_minimal"].has_zh.sum())
    evaluate.save(out, "/tmp/_res.json"); json.load(open("/tmp/_res.json"))
