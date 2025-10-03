"""
This module contains the CodeExplanationEnvironment, which is responsible for running code explanation lessons.
"""
from __future__ import annotations

from app.parent.coder import Coder
from rouge_score import rouge_scorer

class CodeExplanationEnvironment:
    """An environment for running code explanation lessons."""

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def step(self, code_snippet: str, ground_truth_explanation: str) -> float:
        """
        Generates an explanation for the given code snippet and compares it to the ground truth explanation.

        Returns:
            A ROUGE-L F-measure score from 0.0 to 1.0 indicating the similarity.
        """
        try:
            llm = Coder(role='coder')
        except Exception as e:
            print(f"  - ERROR: Could not get LLM provider: {e}")
            return 0.0

        system_prompt = (
            "You are an expert Python programmer. Your task is to generate a concise, accurate explanation of what the given code snippet does. "
            "The explanation should be no more than two sentences."
        )
        user_prompt = f"Explain the following code snippet:\n\n```python\n{code_snippet}\n```"

        try:
            generated_explanation = llm.complete(prompt=user_prompt, system=system_prompt)
        except Exception as e:
            print(f"  - ERROR: Failed to request explanation from LLM: {e}")
            return 0.0

        if not generated_explanation:
            print("  - WARNING: Code explanation generation LLM returned an empty response.")
            return 0.0

        scores = self.scorer.score(ground_truth_explanation, generated_explanation)
        # We use the F-measure of the ROUGE-L score as the similarity metric.
        similarity_score = scores['rougeL'].fmeasure

        return similarity_score