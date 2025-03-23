import pytest
import asyncio
import json
import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import aiohttp
import re
from typing import Dict, List, Tuple, Any, Optional

# Import from existing codebase
from src.llm.moderation import extract_json_from_llm_response
from src.llm.prompts import MODERATION_PROMPT

# Import custom Ollama client for testing
from ollama_client import OllamaClient

class TestModerationWithOllama:
    """Test suite for evaluating LLM moderation capabilities using Ollama"""

    @pytest.fixture(scope="class")
    async def llm_client(self):
        """Initialize custom Ollama client"""
        client = OllamaClient(base_url="http://localhost:11434", model="llama3")
        yield client
        await client.close()  # Cleanup after tests

    @pytest.fixture(scope="class")
    def moderation_dataset(self):
        """Load the moderation dataset from Hugging Face"""
        try:
            # Load from Hugging Face if authenticated
            df = pl.read_ndjson('hf://datasets/ifmain/text-moderation-02-large/moderation.jsonl')
        except Exception as e:
            print(f"Error loading from Hugging Face: {e}")
            # Fallback: Use a local sample if available or create a minimal test set
            sample_data = [
                {"id": 1, "text": "You're so stupid, I hate you!", 
                 "moder": {"categories": {"harassment": True}, "category_scores": {"harassment": 0.92}, "flagged": True}},
                {"id": 2, "text": "I disagree with your point of view.", 
                 "moder": {"categories": {"harassment": False}, "category_scores": {"harassment": 0.03}, "flagged": False}},
                # Add more test cases as needed
            ]
            df = pl.DataFrame(sample_data)
        
        # Select a subset for testing (adjust size as needed)
        return df.sample(n=50, seed=42)

    @pytest.mark.asyncio
    async def test_harassment_detection(self, llm_client, moderation_dataset):
        """Test the LLM's ability to detect harassment in messages"""
        results = []
        
        async def process_message(row):
            message = row["text"]
            # Get ground truth
            ground_truth_flagged = row["moder"]["categories"]["harassment"]
            harassment_score = row["moder"]["category_scores"]["harassment"]
            
            # Format prompt with message
            prompt = MODERATION_PROMPT.format(message=message)
            
            # Get LLM prediction
            try:
                # Looking at the LLMClient implementation in the repo, we need to use the correct method
                response = await llm_client.get_completion(prompt, 
                                                          max_tokens=1024, 
                                                          temperature=0.0)
                result = extract_json_from_llm_response(response)
                
                # Extract prediction
                predicted_flagged = result.get("should_moderate", False)
                reason = result.get("reason", "")
                
                return {
                    "id": row["id"],
                    "text": message[:100] + "..." if len(message) > 100 else message,  # Truncate for readability
                    "ground_truth_flagged": ground_truth_flagged,
                    "harassment_score": harassment_score,
                    "predicted_flagged": predicted_flagged,
                    "reason": reason,
                    "correct": predicted_flagged == ground_truth_flagged
                }
            except Exception as e:
                print(f"Error processing message {row['id']}: {e}")
                return {
                    "id": row["id"],
                    "text": message[:100] + "..." if len(message) > 100 else message,
                    "ground_truth_flagged": ground_truth_flagged,
                    "harassment_score": harassment_score,
                    "predicted_flagged": None,
                    "reason": f"Error: {str(e)}",
                    "correct": False
                }
        
        # Process messages in batches to avoid overwhelming the API
        batch_size = 5
        for i in range(0, len(moderation_dataset), batch_size):
            batch = moderation_dataset.slice(i, min(batch_size, len(moderation_dataset) - i))
            batch_tasks = [process_message(row) for row in batch.to_dicts()]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            # Print progress
            print(f"Processed {min(i + batch_size, len(moderation_dataset))}/{len(moderation_dataset)} messages")
        
        # Convert results to DataFrame for analysis
        results_df = pd.DataFrame(results)
        
        # Calculate metrics
        total = len(results_df)
        correct = results_df["correct"].sum()
        accuracy = correct / total if total > 0 else 0
        
        # Filter for valid predictions (not None)
        valid_results = results_df[results_df["predicted_flagged"].notna()]
        
        # Calculate average harassment score for true positives, false positives, etc.
        true_positives = valid_results[(valid_results["predicted_flagged"] == True) & 
                                      (valid_results["ground_truth_flagged"] == True)]
        false_positives = valid_results[(valid_results["predicted_flagged"] == True) & 
                                       (valid_results["ground_truth_flagged"] == False)]
        true_negatives = valid_results[(valid_results["predicted_flagged"] == False) & 
                                      (valid_results["ground_truth_flagged"] == False)]
        false_negatives = valid_results[(valid_results["predicted_flagged"] == False) & 
                                       (valid_results["ground_truth_flagged"] == True)]
        
        # Print metrics
        print(f"\n===== Moderation Test Results =====")
        print(f"Total messages: {total}")
        print(f"Correctly classified: {correct} ({accuracy:.2%})")
        print(f"True positives: {len(true_positives)}")
        print(f"False positives: {len(false_positives)}")
        print(f"True negatives: {len(true_negatives)}")
        print(f"False negatives: {len(false_negatives)}")
        
        # Print average harassment scores
        print(f"\n===== Average Harassment Scores =====")
        if not true_positives.empty:
            print(f"True positives: {true_positives['harassment_score'].mean():.4f}")
        if not false_positives.empty:
            print(f"False positives: {false_positives['harassment_score'].mean():.4f}")
        if not true_negatives.empty:
            print(f"True negatives: {true_negatives['harassment_score'].mean():.4f}")
        if not false_negatives.empty:
            print(f"False negatives: {false_negatives['harassment_score'].mean():.4f}")
        
        # Save results to CSV for further analysis
        results_df.to_csv("moderation_test_results.csv", index=False)
        print(f"Detailed results saved to moderation_test_results.csv")
        
        # Calculate confusion matrix
        confusion_matrix = {
            "true_positive": len(true_positives),
            "false_positive": len(false_positives),
            "true_negative": len(true_negatives),
            "false_negative": len(false_negatives)
        }
        
        # Calculate precision, recall, F1
        precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0
        recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if (len(true_positives) + len(false_negatives)) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n===== Classification Metrics =====")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {precision:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Assert minimal performance threshold
        assert accuracy > 0.5, "Moderation accuracy should be better than random chance"
        
        return results_df