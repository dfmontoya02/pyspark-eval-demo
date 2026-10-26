# PySpark Model Evaluation Demo

A small, self-contained PySpark script that mirrors the shape of a real
anomaly-detection model evaluation workflow: synthetic "officer referral"
records (airport, contraband type, model confidence score, ground-truth
outcome) are scored across several confidence thresholds to show the
precision/recall tradeoff, then broken down per airport using Spark SQL.

## Why this exists

I evaluate computer-vision anomaly-detection models in my day job, building
ground-truth datasets from a production PostgreSQL database and scoring
model output against it in Python. That work has always been single-node.
This is a quick hands-on exercise translating the same evaluation logic —
confusion-matrix metrics, threshold tuning, per-segment breakdowns — into
PySpark, to get comfortable with how the same ideas look once the data
(and the compute) is distributed across a cluster instead of one machine.

## What it does

1. Generates a synthetic dataset of 5,000 referral records (contraband is
   the rare class, ~12%, with realistic score overlap near the decision
   boundary).
2. Sweeps six confidence thresholds (0.3–0.8) and reports precision,
   recall, and F1 at each.
3. Picks an operating threshold that favors **recall over raw accuracy** —
   deliberately mirroring the real-world judgment call that a missed
   detection is costlier than a false alarm.
4. Breaks recall down per airport using a Spark SQL query against a
   temp view, the same kind of slice used to catch a model quietly
   underperforming at one location while looking fine in aggregate.

## Running it

```bash
python3 -m venv venv
./venv/bin/pip install pyspark
./venv/bin/python eval_pipeline.py
```

No cluster or cloud setup needed — runs in Spark's local mode.
