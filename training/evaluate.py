"""
Evaluation Metrics & Benchmark Module for LipNet VSR.
Computes Character Error Rate (CER) and Word Error Rate (WER) using Levenshtein distance.
"""

from typing import List, Tuple, Dict
import numpy as np


def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """Computes minimum edit operations (insertions, deletions, substitutions)."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )
    return dp[m][n]


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Computes Character Error Rate (CER).
    CER = (Substitutions + Deletions + Insertions) / Total_Reference_Characters
    """
    ref_chars = list(reference.strip().lower())
    hyp_chars = list(hypothesis.strip().lower())

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    edit_dist = levenshtein_distance(ref_chars, hyp_chars)
    return float(edit_dist / len(ref_chars))


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Computes Word Error Rate (WER).
    WER = (Substitutions + Deletions + Insertions) / Total_Reference_Words
    """
    ref_words = reference.strip().lower().split()
    hyp_words = hypothesis.strip().lower().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    edit_dist = levenshtein_distance(ref_words, hyp_words)
    return float(edit_dist / len(ref_words))


def evaluate_batch(
    references: List[str],
    hypotheses: List[str]
) -> Dict[str, float]:
    """
    Computes corpus-level average CER and WER.
    """
    cer_list = [calculate_cer(r, h) for r, h in zip(references, hypotheses)]
    wer_list = [calculate_wer(r, h) for r, h in zip(references, hypotheses)]

    return {
        "mean_cer": float(np.mean(cer_list)) if cer_list else 0.0,
        "mean_wer": float(np.mean(wer_list)) if wer_list else 0.0,
        "cer_percent": f"{np.mean(cer_list) * 100:.2f}%" if cer_list else "0.00%",
        "wer_percent": f"{np.mean(wer_list) * 100:.2f}%" if wer_list else "0.00%",
        "sample_count": len(references)
    }


if __name__ == "__main__":
    ref = "bin blue at f two now"
    hyp = "bin blue at f two now"
    print("Exact match CER:", calculate_cer(ref, hyp), "WER:", calculate_wer(ref, hyp))

    hyp_err = "bin red at s two now"
    print("Error match CER:", calculate_cer(ref, hyp_err), "WER:", calculate_wer(ref, hyp_err))
