#!/usr/bin/env python3
"""
Script to visualize the moderation test results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def visualize_results(results_file="tests/data/moderation_test_results.csv"):
    """Visualize the results of the moderation tests"""
    if not Path(results_file).exists():
        print(f"Results file {results_file} not found.")
        return
        
    # Load results
    df = pd.read_csv(results_file)
    
    # Replace NaN values for better visualization
    df = df.fillna({"predicted_flagged": False})
    
    # Set up the plots
    plt.figure(figsize=(15, 10))
    
    # 1. Confusion matrix heatmap
    plt.subplot(2, 2, 1)
    conf_matrix = pd.crosstab(df['ground_truth_flagged'], df['predicted_flagged'], 
                            rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    
    # 2. Harassment score distribution by prediction status
    plt.subplot(2, 2, 2)
    
    # Create categories for the plot
    df['prediction_status'] = 'Other'
    df.loc[(df['ground_truth_flagged'] == True) & (df['predicted_flagged'] == True), 'prediction_status'] = 'True Positive'
    df.loc[(df['ground_truth_flagged'] == False) & (df['predicted_flagged'] == True), 'prediction_status'] = 'False Positive'
    df.loc[(df['ground_truth_flagged'] == True) & (df['predicted_flagged'] == False), 'prediction_status'] = 'False Negative'
    df.loc[(df['ground_truth_flagged'] == False) & (df['predicted_flagged'] == False), 'prediction_status'] = 'True Negative'
    
    # Box plot
    sns.boxplot(x='prediction_status', y='harassment_score', data=df)
    plt.title('Harassment Score Distribution by Prediction Status')
    plt.xticks(rotation=45)
    
    # 3. ROC curve (approximation)
    plt.subplot(2, 2, 3)
    # Sort by harassment score
    df_sorted = df.sort_values(by='harassment_score', ascending=False).reset_index(drop=True)
    
    # Calculate TPR and FPR at different thresholds
    total_positives = df['ground_truth_flagged'].sum()
    total_negatives = len(df) - total_positives
    
    if total_positives > 0 and total_negatives > 0:
        tpr = []
        fpr = []
        thresholds = []
        
        for i in range(len(df_sorted) + 1):
            # Consider top i messages as 'flagged'
            flagged = set(df_sorted.iloc[:i].index) if i > 0 else set()
            actual_positives = set(df[df['ground_truth_flagged'] == True].index)
            actual_negatives = set(df[df['ground_truth_flagged'] == False].index)
            
            # Calculate True Positives and False Positives
            tp = len(flagged.intersection(actual_positives))
            fp = len(flagged.intersection(actual_negatives))
            
            # Calculate TPR and FPR
            tpr.append(tp / total_positives if total_positives > 0 else 0)
            fpr.append(fp / total_negatives if total_negatives > 0 else 0)
            
            if i < len(df_sorted):
                thresholds.append(df_sorted.iloc[i]['harassment_score'])
            else:
                thresholds.append(0)
        
        # Plot ROC curve
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')  # Random classifier line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (by harassment score)')
        
        # Calculate AUC
        auc = np.trapz(tpr, fpr)
        plt.text(0.6, 0.2, f'AUC = {auc:.3f}')
    else:
        plt.text(0.5, 0.5, 'Insufficient data for ROC curve', ha='center')
    
    # 4. Histogram of harassment scores colored by prediction
    plt.subplot(2, 2, 4)
    
    # Plot for correctly classified
    sns.histplot(df[df['correct'] == True]['harassment_score'], 
                color='green', alpha=0.5, label='Correctly Classified')
    
    # Plot for incorrectly classified
    sns.histplot(df[df['correct'] == False]['harassment_score'], 
                color='red', alpha=0.5, label='Incorrectly Classified')
    
    plt.xlabel('Harassment Score')
    plt.ylabel('Count')
    plt.title('Harassment Score Distribution by Classification Correctness')
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('tests/data/moderation_test_results.png')
    print(f"Visualization saved to moderation_test_results.png")
    
    # Generate summary statistics
    print("\n===== Summary Statistics =====")
    
    # Get average harassment score by prediction category
    print("\nAverage Harassment Score by Category:")
    for category in ['True Positive', 'False Positive', 'True Negative', 'False Negative']:
        subset = df[df['prediction_status'] == category]
        if not subset.empty:
            print(f"{category}: {subset['harassment_score'].mean():.4f} (n={len(subset)})")
    
    # Overall accuracy
    accuracy = (df['correct'].sum() / len(df)) if len(df) > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Find examples of false positives and false negatives
    print("\nExample False Positives (Non-harassment predicted as harassment):")
    false_positives = df[(df['ground_truth_flagged'] == False) & (df['predicted_flagged'] == True)]
    for i, row in false_positives.head(3).iterrows():
        print(f"ID: {row['id']}, Score: {row['harassment_score']:.4f}")
        print(f"Text: {row['text']}")
        print(f"Reason: {row['reason']}")
        print("---")
    
    print("\nExample False Negatives (Harassment predicted as non-harassment):")
    false_negatives = df[(df['ground_truth_flagged'] == True) & (df['predicted_flagged'] == False)]
    for i, row in false_negatives.head(3).iterrows():
        print(f"ID: {row['id']}, Score: {row['harassment_score']:.4f}")
        print(f"Text: {row['text']}")
        print(f"Reason: {row['reason']}")
        print("---")

if __name__ == "__main__":
    visualize_results()