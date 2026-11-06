"""Parity test: the rewritten scoring / selection / flag logic must equal the ORIGINAL notebook functions.
The original code is executed straight from original_notebooks/*.ipynb (sanitised copies), with a deterministic stub embedder."""
import sys, os, re, json, random, hashlib, logging
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tpsaudit import tps

NB = os.path.join(os.path.dirname(__file__), "..", "original_notebooks")

def cells_text(name):
    nb = json.load(open(os.path.join(NB, name), encoding="utf8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")

def between(src, a, b):
    i = src.index(a); j = src.index(b, i + len(a)); return src[i:j]

def test_parity_with_original_notebooks():
    s2 = cells_text("stage2_dataset_generator.ipynb"); s4 = cells_text("stage4_evaluation.ipynb")
    # deterministic bag-of-words embedder (similar texts -> similar vectors), L2-normalised
    def vec(text, d=64):
        v = np.zeros(d)
        for w in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            r = np.random.default_rng(h % (2**32)); v += r.normal(size=d)
        n = np.linalg.norm(v); return v / n if n else v
    def embed(xs): return np.array([vec(x) for x in xs])

    class Sc:  # stub 'tensor' result of util.cos_sim
        def __init__(s, x): s.x = x
        def item(s): return float(s.x)
    class StubST:
        def encode(self, x, convert_to_tensor=False, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
            return embed(x) if isinstance(x, list) else vec(x)
    class util:
        @staticmethod
        def cos_sim(a, b): return Sc(np.dot(a, b))

    # ---- original Stage 2 functions
    g2 = {'re': re, 'np': np, 'log': logging.getLogger('p'), 'MARGIN_THRESHOLD': 0.15,
          'W_SEM_GROUNDING': 0.4, 'W_SEM_RELEVANCE': 0.4, 'W_LEXICAL': 0.2, 'sentence_model': StubST(), 'load_sentence_model': lambda: None}
    exec(between(s2, 'STOP_WORDS_ID: set = {', '# [Cell 13] System prompt'), g2)
    exec(between(s2, 'def select_dpo_pair', '# [Cell 17] Pipeline utama'), g2)
    # ---- original Stage 4 functions
    g4 = {'re': re, 'np': np, 'embed_model': StubST(), 'util': util, 'DAHL_W1': .4, 'DAHL_W2': .4, 'DAHL_W3': .2, 'HALLUCINATION_THRESHOLD': 0.5,
          'MIN_TOKEN_LEN': 4}
    exec(between(s4, 'STOP_WORDS_ID: set = {', 'PAGE_PATTERN'), g4)
    exec(between(s4, 'def tokenize_for_lexical', 'def parse_page_number'), g4)
    exec(between(s4, 'def calculate_DAHL_score', '# CELL 9B'), g4)
    exec(between(s4, 'def hallucination_audit', '# =========================================================\n# CELL 13-18'), g4)

    assert g2['STOP_WORDS_ID'] == tps.STOP_WORDS_ID == g4['STOP_WORDS_ID'], 'stop-word sets differ'
    rng = random.Random(1)
    vocab = list(tps.STOP_WORDS_ID) + ["mahasiswa", "skripsi", "sidang", "magang", "kredit", "semester", "dosen", "pembimbing", "syarat", "nilai",
              "kampus", "laporan", "seminar", "wajib", "tiga", "SKS", "144", "Maret", "cuti", "akademik", "pendaftaran", "tenggat", "registrasi"]
    def sent(n): return ' '.join(rng.choice(vocab) + rng.choice(['', '.', ',']) for _ in range(n))


    g2 = {"re": re, "np": np, "log": logging.getLogger("parity"), "MARGIN_THRESHOLD": 0.15, "W_SEM_GROUNDING": 0.4, "W_SEM_RELEVANCE": 0.4,
          "W_LEXICAL": 0.2, "sentence_model": StubST(), "load_sentence_model": lambda: None}
    logging.getLogger("parity").setLevel(logging.CRITICAL)
    exec(between(s2, "STOP_WORDS_ID: set = {", "# [Cell 13] System prompt"), g2)
    exec(between(s2, "def select_dpo_pair", "# [Cell 17] Pipeline utama"), g2)
    g4 = {"re": re, "np": np, "embed_model": StubST(), "util": util, "DAHL_W1": .4, "DAHL_W2": .4, "DAHL_W3": .2, "HALLUCINATION_THRESHOLD": 0.5, "MIN_TOKEN_LEN": 4}
    exec(between(s4, "STOP_WORDS_ID: set = {", "PAGE_PATTERN"), g4)
    exec(between(s4, "def tokenize_for_lexical", "def parse_page_number"), g4)
    exec(between(s4, "def calculate_DAHL_score", "# CELL 9B"), g4)
    exec(between(s4, "def hallucination_audit", "# =========================================================\n# CELL 13-18"), g4)
    assert g2["STOP_WORDS_ID"] == tps.STOP_WORDS_ID == g4["STOP_WORDS_ID"] and len(tps.STOP_WORDS_ID) == 89
    for k in range(400):
        resp, ctx, q = sent(rng.randint(3, 40)), sent(rng.randint(10, 120)), sent(rng.randint(3, 12))
        # tokenizer + lexical
        assert g2['tokenize_for_lexical'](resp) == tps.tokenize_for_lexical(resp) == g4['tokenize_for_lexical'](resp)
        # Stage 2 scoring
        a = g2['compute_sfinal'](resp, ctx, q); b = tps.compute_sfinal(resp, ctx, q, embed)
        assert a == b, (a, b)
        # Stage 2 selection
        cands = [{'label': 'c', 'text': sent(rng.randint(3, 40)), 'temperature': .4, 'with_context': True},
                 {'label': 'k', 'text': sent(rng.randint(3, 40)), 'temperature': 1.2, 'with_context': True},
                 {'label': 'r', 'text': sent(rng.randint(3, 60)), 'temperature': 1.8, 'with_context': False}]
        p1 = g2['select_dpo_pair'](cands, ctx, q); p2 = tps.select_dpo_pair(cands, ctx, q, embed)
        assert (p1 is None) == (p2 is None)
        if p1:
            assert p1['chosen'] == p2['chosen']['text'] and p1['rejected'] == p2['rejected']['text'] and p1['score_margin'] == p2['margin']
        # Stage 4 scoring + flag
        contexts = [{'filename': f'doc_p000{i}.png', 'text': sent(rng.randint(0, 60))} for i in range(rng.randint(0, 3))]
        sa, da = g4['calculate_DAHL_score'](resp, q, contexts); sb, db = tps.eval_score(resp, q, contexts, embed)
        assert sa == sb and da == db, (sa, sb, da, db)
        row = {'response': resp, 'dahl_score': sa, 'dahl_details': da, 'filenames': [c['filename'] for c in contexts]}
        f1 = g4['hallucination_audit']([row])[0]['hallucination_flag']
        f2, _ = tps.hallucination_flag(resp, sa, da, row['filenames'])
        assert f1 == f2

    # edge cases
    for r in ["", "abc", "    "]:
        row = {'response': r, 'dahl_score': 0.0, 'dahl_details': {}, 'filenames': ['x']}
        assert g4['hallucination_audit']([row])[0]['hallucination_flag'] == tps.hallucination_flag(r, 0.0, {}, ['x'])[0]
        assert g4['calculate_DAHL_score'](r, 'q', []) == tps.eval_score(r, 'q', [], embed)
    # dataset context builder
    pages = [{'pdf_name': 'A', 'page_number': 0, 'score': 12.3456, 'image_path': None}, {'pdf_name': 'B', 'page_number': None, 'score': 1.0}, {'pdf_name': 'C', 'page_number': 4, 'score': 0.5}]
    texts = {'A': 'x' * 3000, 'B': '', 'C': 'halo dunia'}
    g2['get_page_text_with_ocr_fallback'] = lambda pdf, pn, ip: texts[pdf]
    g2['MAX_CONTEXT_CHARS_PER_SOURCE'] = 1200
    exec(between(s2, 'def build_context', '# [Cell 11] Stop-words'), g2)
    pages_o = [{**p, 'image_path': None} for p in pages]
    assert g2['build_context'](pages_o) == tps.build_dataset_context(pages, lambda p: texts[p['pdf_name']])

    # exhaustive grid around every flag threshold (random inputs rarely land near 0.50 / 0.35 / 0.20)
    import itertools
    for sc, d1, d2, fn, rs in itertools.product([None, 0.0, 0.4999, 0.5, 0.5001], [0.3499, 0.35, 0.3501, 0.30], [0.1999, 0.2, 0.2001, 0.1],
                                                [[], ["x"]], ["", "jawaban"]):
        det = {"dim1_relevance": d1, "dim2_grounding": d2}
        row = {"response": rs, "dahl_score": sc, "dahl_details": det, "filenames": fn}
        assert g4["hallucination_audit"]([row])[0]["hallucination_flag"] == tps.hallucination_flag(rs, sc, det, fn)[0], (sc, d1, d2, fn, rs)
    # missing detail keys default to "not flagged" in the original (details.get(..., 1.0))
    for sc in (0.6, 0.4):
        row = {"response": "x y z", "dahl_score": sc, "dahl_details": {}, "filenames": ["f"]}
        assert g4["hallucination_audit"]([row])[0]["hallucination_flag"] == tps.hallucination_flag("x y z", sc, {}, ["f"])[0]
