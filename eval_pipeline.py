"""
PySpark model-evaluation demo.

Mirrors the shape of a real anomaly-detection evaluation workflow:
synthetic "officer referral" records (airport, contraband type, model
confidence score, ground-truth outcome) are scored at several
confidence thresholds to show the precision/recall tradeoff, then
broken down per airport -- the same kind of slicing used to sanity
check a model against operational conditions rather than a single
global accuracy number.
"""

import random

from pyspark.sql import SparkSession, functions as F

random.seed(7)

AIRPORTS = ["JFK", "LAX", "ORD", "MIA", "ATL"]
CONTRABAND_TYPES = ["narcotics", "currency", "counterfeit_goods", "none"]


def make_synthetic_referrals(n=5000):
    rows = []
    for i in range(n):
        is_contraband = random.random() < 0.12  # contraband is the rare class
        contraband_type = (
            random.choice(CONTRABAND_TYPES[:-1]) if is_contraband else "none"
        )
        # true positives tend to score higher; negatives are spread lower,
        # with some overlap near the decision boundary (realistic noise)
        confidence = (
            min(1.0, max(0.0, random.gauss(0.78, 0.15)))
            if is_contraband
            else min(1.0, max(0.0, random.gauss(0.30, 0.18)))
        )
        rows.append(
            {
                "referral_id": i,
                "airport": random.choice(AIRPORTS),
                "contraband_type": contraband_type,
                "confidence_score": round(confidence, 4),
                "actual_label": int(is_contraband),
            }
        )
    return rows


def confusion_counts(df, threshold):
    scored = df.withColumn(
        "predicted_label",
        (F.col("confidence_score") >= F.lit(threshold)).cast("int"),
    )
    agg = scored.agg(
        F.sum(
            F.when(
                (F.col("predicted_label") == 1) & (F.col("actual_label") == 1), 1
            ).otherwise(0)
        ).alias("tp"),
        F.sum(
            F.when(
                (F.col("predicted_label") == 1) & (F.col("actual_label") == 0), 1
            ).otherwise(0)
        ).alias("fp"),
        F.sum(
            F.when(
                (F.col("predicted_label") == 0) & (F.col("actual_label") == 0), 1
            ).otherwise(0)
        ).alias("tn"),
        F.sum(
            F.when(
                (F.col("predicted_label") == 0) & (F.col("actual_label") == 1), 1
            ).otherwise(0)
        ).alias("fn"),
    ).first()
    return agg["tp"], agg["fp"], agg["tn"], agg["fn"]


def precision_recall_f1(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def main():
    spark = SparkSession.builder.appName("referral-eval-demo").master(
        "local[*]"
    ).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.createDataFrame(make_synthetic_referrals())
    df.createOrReplaceTempView("referrals")

    print("\n=== Sample rows ===")
    df.show(5, truncate=False)

    print("=== Class balance (actual_label) ===")
    df.groupBy("actual_label").count().orderBy("actual_label").show()

    # Threshold sweep -- same idea as tuning a detection threshold against
    # how cautious the model should be, instead of optimizing raw accuracy.
    print("=== Threshold sweep ===")
    print(f"{'threshold':>9} | {'precision':>9} | {'recall':>7} | {'f1':>6}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        tp, fp, tn, fn = confusion_counts(df, t)
        p, r, f1 = precision_recall_f1(tp, fp, tn, fn)
        print(f"{t:>9.2f} | {p:>9.3f} | {r:>7.3f} | {f1:>6.3f}")

    # Recall-favoring threshold: officers err on the side of caution, so a
    # lower threshold that minimizes missed detections (false negatives)
    # mirrors that judgment better than the threshold that maximizes raw
    # accuracy would.
    chosen_threshold = 0.4
    tp, fp, tn, fn = confusion_counts(df, chosen_threshold)
    p, r, f1 = precision_recall_f1(tp, fp, tn, fn)
    print(
        f"\nChosen operating threshold: {chosen_threshold} "
        f"(precision={p:.3f}, recall={r:.3f}, f1={f1:.3f}) "
        f"-- favors recall to limit missed contraband (fn={fn})"
    )

    # Per-airport breakdown at the chosen threshold -- same kind of slice
    # ("what airport") used to check the model isn't quietly underperforming
    # at one location while looking fine in aggregate.
    print("\n=== Per-airport recall at chosen threshold (via Spark SQL) ===")
    spark.sql(
        f"""
        SELECT
            airport,
            SUM(CASE WHEN actual_label = 1 THEN 1 ELSE 0 END) AS actual_positives,
            SUM(CASE WHEN confidence_score >= {chosen_threshold}
                      AND actual_label = 1 THEN 1 ELSE 0 END) AS true_positives,
            ROUND(
                SUM(CASE WHEN confidence_score >= {chosen_threshold}
                          AND actual_label = 1 THEN 1 ELSE 0 END)
                / SUM(CASE WHEN actual_label = 1 THEN 1 ELSE 0 END), 3
            ) AS recall
        FROM referrals
        GROUP BY airport
        ORDER BY airport
        """
    ).show()

    spark.stop()


if __name__ == "__main__":
    main()
