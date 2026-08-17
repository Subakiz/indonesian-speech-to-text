"""Evaluation Metrics for Indonesian Speech-to-Text (ASR)."""

from typing import Dict, List, Tuple
import evaluate
from data.indonesian_normalizer import IndonesianTextNormalizer


class IndonesianASRMetrics:
    """Calculates Word Error Rate (WER) and Character Error Rate (CER) with Indonesian-aware normalization."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.wer_metric = evaluate.load("wer")
        self.cer_metric = evaluate.load("cer")
        self.eval_normalizer = IndonesianTextNormalizer(remove_punctuation=True, to_lower=True)

    def compute_metrics(self, pred_ids: List[List[int]], label_ids: List[List[int]]) -> Dict[str, float]:
        """Compute WER and CER between predicted token IDs and target label IDs."""
        # Replace -100 with tokenizer pad token id
        clean_label_ids = []
        for row in label_ids:
            clean_row = [t if t != -100 else self.tokenizer.pad_token_id for t in row]
            clean_label_ids.append(clean_row)

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(clean_label_ids, skip_special_tokens=True)

        return self.compute_metrics_from_text(pred_str, label_str)

    def compute_metrics_from_text(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute Raw and Normalized WER and CER from strings."""
        # Raw evaluation
        raw_wer = self.wer_metric.compute(predictions=predictions, references=references) * 100
        raw_cer = self.cer_metric.compute(predictions=predictions, references=references) * 100

        # Indonesian-normalized evaluation
        norm_preds = [self.eval_normalizer(p) for p in predictions]
        norm_refs = [self.eval_normalizer(r) for r in references]

        # Filter out empty references if any to avoid jiwer ZeroDivisionError
        filtered_pairs = [(p, r) for p, r in zip(norm_preds, norm_refs) if len(r.strip()) > 0]
        if filtered_pairs:
            f_preds, f_refs = zip(*filtered_pairs)
            norm_wer = self.wer_metric.compute(predictions=list(f_preds), references=list(f_refs)) * 100
            norm_cer = self.cer_metric.compute(predictions=list(f_preds), references=list(f_refs)) * 100
        else:
            norm_wer = 0.0
            norm_cer = 0.0

        return {
            "raw_wer": round(raw_wer, 2),
            "raw_cer": round(raw_cer, 2),
            "normalized_wer": round(norm_wer, 2),
            "normalized_cer": round(norm_cer, 2),
        }
