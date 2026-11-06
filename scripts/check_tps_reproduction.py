#!/usr/bin/env python3
"""Re-score the stored pairs and compare with the TPS values stored in data/pairs.csv.
Run this FIRST. If the maximum absolute difference is not ~0, adapt src/tpsaudit/tps.py (e.g. how the context is
embedded: concatenated vs per source, truncation) until it reproduces your original scores.
Columns needed: question, context, chosen, rejected, chosen_tps, rej_tps."""
import sys, pandas as pd
sys.path.insert(0, "src")
from tpsaudit.tps import Scorer
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/pairs.csv")
sc = Scorer()
c = sc.score(df.chosen, df.question, df.context); r = sc.score(df.rejected, df.question, df.context)
dc, dr = (c.tps - df.chosen_tps).abs(), (r.tps - df.rej_tps).abs()
print(f"chosen: max|diff|={dc.max():.4f} mean={dc.mean():.4f}   rejected: max|diff|={dr.max():.4f} mean={dr.mean():.4f}")
print("margin kept >= 0.15 under re-scoring: %d of %d" % (((c.tps - r.tps) >= 0.15).sum(), len(df)))
