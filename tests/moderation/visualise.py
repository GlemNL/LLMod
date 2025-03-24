#!/usr/bin/env python3
"""
Script to visualize the moderation test results
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns


def visualize_results(results_file="tests/data/moderation_test_results.csv"):
    """Visualize the results of the moderation tests"""
    if not Path(results_file).exists():
        print(f"Results file {results_file} not found.")
        return

    # Load results
    df = pl.read_csv(results_file)

    # Replace null values for better visualization - handling each column separately to avoid type issues
    if "predicted_flagged" in df.columns:
        df = df.with_columns(pl.col("predicted_flagged").fill_null(False))

    # Set up the plots
    plt.figure(figsize=(15, 10))

    # Ensure boolean columns are actually boolean type
    if "ground_truth_flagged" in df.columns:
        df = df.with_columns(pl.col("ground_truth_flagged").cast(pl.Boolean))

    if "predicted_flagged" in df.columns:
        df = df.with_columns(pl.col("predicted_flagged").cast(pl.Boolean))

    # 1. Confusion matrix heatmap
    plt.subplot(2, 2, 1)
    # Create confusion matrix manually
    truth_pred_counts = df.group_by(
        ["ground_truth_flagged", "predicted_flagged"]
    ).count()

    # Convert to a format suitable for the heatmap
    matrix_data = {
        (row["ground_truth_flagged"], row["predicted_flagged"]): row["count"]
        for row in truth_pred_counts.to_dicts()
    }

    # Create 2x2 matrix with zeros as default
    conf_matrix = np.zeros((2, 2))

    # Fill in the counts
    for (truth, pred), count in matrix_data.items():
        truth_idx = 1 if truth else 0
        pred_idx = 1 if pred else 0
        conf_matrix[truth_idx, pred_idx] = count

    # Create proper DataFrame for the heatmap
    conf_matrix_df = np.array(conf_matrix)
    sns.heatmap(
        conf_matrix_df,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=["False", "True"],
        yticklabels=["False", "True"],
    )
    plt.title("Confusion Matrix")

    # 2. Harassment score distribution by prediction status
    plt.subplot(2, 2, 2)

    # Create categories for the plot
    df = df.with_columns(pl.lit("Other").alias("prediction_status"))

    # Update prediction status for each category
    df = df.with_columns(
        [
            pl.when(
                (pl.col("ground_truth_flagged") == True)
                & (pl.col("predicted_flagged") == True)
            )
            .then(pl.lit("True Positive"))
            .otherwise(pl.col("prediction_status"))
            .alias("prediction_status")
        ]
    )

    df = df.with_columns(
        [
            pl.when(
                (pl.col("ground_truth_flagged") == False)
                & (pl.col("predicted_flagged") == True)
            )
            .then(pl.lit("False Positive"))
            .otherwise(pl.col("prediction_status"))
            .alias("prediction_status")
        ]
    )

    df = df.with_columns(
        [
            pl.when(
                (pl.col("ground_truth_flagged") == True)
                & (pl.col("predicted_flagged") == False)
            )
            .then(pl.lit("False Negative"))
            .otherwise(pl.col("prediction_status"))
            .alias("prediction_status")
        ]
    )

    df = df.with_columns(
        [
            pl.when(
                (pl.col("ground_truth_flagged") == False)
                & (pl.col("predicted_flagged") == False)
            )
            .then(pl.lit("True Negative"))
            .otherwise(pl.col("prediction_status"))
            .alias("prediction_status")
        ]
    )

    # Convert to pandas for the boxplot
    boxplot_data = df.select(["prediction_status", "harassment_score"]).to_pandas()
    sns.boxplot(x="prediction_status", y="harassment_score", data=boxplot_data)
    plt.title("Harassment Score Distribution by Prediction Status")
    plt.xticks(rotation=45)

    # 3. ROC curve (approximation)
    plt.subplot(2, 2, 3)

    # Calculate TPR and FPR at different thresholds
    total_positives = df.filter(pl.col("ground_truth_flagged") == True).height
    total_negatives = df.height - total_positives

    if total_positives > 0 and total_negatives > 0:
        # Get all unique harassment scores as thresholds
        thresholds = (
            df.select(pl.col("harassment_score").unique().sort(descending=True))
            .to_series()
            .to_list()
        )
        thresholds.append(0)  # Add zero as the final threshold

        tpr = []
        fpr = []

        # For each threshold, calculate TPR and FPR
        for threshold in thresholds:
            # Mark instances with score >= threshold as flagged
            predicted_positives = df.filter(pl.col("harassment_score") >= threshold)

            # Calculate TP and FP
            tp = predicted_positives.filter(
                pl.col("ground_truth_flagged") == True
            ).height
            fp = predicted_positives.filter(
                pl.col("ground_truth_flagged") == False
            ).height

            # Calculate TPR and FPR
            tpr.append(tp / total_positives if total_positives > 0 else 0)
            fpr.append(fp / total_negatives if total_negatives > 0 else 0)

        # Plot ROC curve
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], "k--")  # Random classifier line
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (by harassment score)")

        # Calculate AUC
        auc = np.trapz(tpr, fpr)
        plt.text(0.6, 0.2, f"AUC = {auc:.3f}")
    else:
        plt.text(0.5, 0.5, "Insufficient data for ROC curve", ha="center")

    # 4. Histogram of harassment scores colored by prediction
    plt.subplot(2, 2, 4)

    # Check if 'correct' column exists in the dataset
    if "correct" in df.columns:
        # Convert to lists for seaborn histplot
        correctly_classified = df.filter(pl.col("correct") == True)[
            "harassment_score"
        ].to_list()
        incorrectly_classified = df.filter(pl.col("correct") == False)[
            "harassment_score"
        ].to_list()

        # Plot for correctly classified
        sns.histplot(
            correctly_classified, color="green", alpha=0.5, label="Correctly Classified"
        )

        # Plot for incorrectly classified
        sns.histplot(
            incorrectly_classified,
            color="red",
            alpha=0.5,
            label="Incorrectly Classified",
        )
    else:
        # Create the 'correct' column if it doesn't exist
        df = df.with_columns(
            (
                (pl.col("predicted_flagged") & pl.col("ground_truth_flagged"))
                | (~pl.col("predicted_flagged") & ~pl.col("ground_truth_flagged"))
            ).alias("correct")
        )

        # Now plot with the newly created column
        correctly_classified = df.filter(pl.col("correct") == True)[
            "harassment_score"
        ].to_list()
        incorrectly_classified = df.filter(pl.col("correct") == False)[
            "harassment_score"
        ].to_list()

        # Plot for correctly classified
        sns.histplot(
            correctly_classified, color="green", alpha=0.5, label="Correctly Classified"
        )

        # Plot for incorrectly classified
        sns.histplot(
            incorrectly_classified,
            color="red",
            alpha=0.5,
            label="Incorrectly Classified",
        )

    plt.xlabel("Harassment Score")
    plt.ylabel("Count")
    plt.title("Harassment Score Distribution by Classification Correctness")
    plt.legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("tests/data/moderation_test_results.png")
    print(f"Visualization saved to moderation_test_results.png")

    # Generate summary statistics
    print("\n===== Summary Statistics =====")

    # Get average harassment score by prediction category
    print("\nAverage Harassment Score by Category:")
    for category in [
        "True Positive",
        "False Positive",
        "True Negative",
        "False Negative",
    ]:
        subset = df.filter(pl.col("prediction_status") == category)
        if subset.height > 0:
            mean_score = subset.select(pl.col("harassment_score").mean()).item()
            print(f"{category}: {mean_score:.4f} (n={subset.height})")

    # Overall accuracy
    accuracy = (
        (df.filter(pl.col("correct") == True).height / df.height)
        if df.height > 0
        else 0
    )
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    # Find examples of false positives and false negatives
    print("\nExample False Positives (Non-harassment predicted as harassment):")
    false_positives = df.filter(
        (pl.col("ground_truth_flagged") == False)
        & (pl.col("predicted_flagged") == True)
    )

    for i in range(min(3, false_positives.height)):
        row = false_positives.row(i, named=True)
        print(f"ID: {row['id']}, Score: {row['harassment_score']:.4f}")
        print(f"Text: {row['text']}")
        print(f"Reason: {row['reason']}")
        print("---")

    print("\nExample False Negatives (Harassment predicted as non-harassment):")
    false_negatives = df.filter(
        (pl.col("ground_truth_flagged") == True)
        & (pl.col("predicted_flagged") == False)
    )

    for i in range(min(3, false_negatives.height)):
        row = false_negatives.row(i, named=True)
        print(f"ID: {row['id']}, Score: {row['harassment_score']:.4f}")
        print(f"Text: {row['text']}")
        print(f"Reason: {row['reason']}")
        print("---")


if __name__ == "__main__":
    visualize_results()
