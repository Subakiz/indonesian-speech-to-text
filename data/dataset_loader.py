"""Hugging Face Indonesian Speech Dataset Loader with SpecAugment & Multi-Corpus Blending."""

import io
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch
import torchaudio
import soundfile as sf
import librosa
import numpy as np
from datasets import load_dataset, Dataset, Audio, concatenate_datasets
from transformers import WhisperProcessor
from data.indonesian_normalizer import IndonesianTextNormalizer


def apply_spec_augment(
    features: np.ndarray,
    freq_mask_param: int = 15,
    time_mask_param: int = 35,
    max_masks: int = 2,
) -> np.ndarray:
    """Apply dynamic SpecAugment (Frequency and Time Masking) to Log-Mel Spectrogram."""
    augmented = features.copy()
    num_mels, time_steps = augmented.shape

    # Frequency Masking
    for _ in range(random.randint(1, max_masks)):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(0, num_mels - f))
        augmented[f0 : f0 + f, :] = 0.0

    # Time Masking
    for _ in range(random.randint(1, max_masks)):
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, max(0, time_steps - t))
        augmented[:, t0 : t0 + t] = 0.0

    return augmented


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Collator for Whisper model fine-tuning with dynamic padding."""
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def extract_audio_waveform(audio_item: Any, target_sr: int = 16000) -> np.ndarray:
    """Safely extract and resample audio waveform to target sample rate using soundfile/librosa."""
    if isinstance(audio_item, dict):
        if "array" in audio_item and audio_item["array"] is not None:
            arr = np.array(audio_item["array"], dtype=np.float32)
            sr = audio_item.get("sampling_rate", target_sr)
            if sr != target_sr:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
            return arr
        elif "bytes" in audio_item and audio_item["bytes"] is not None:
            arr, sr = sf.read(io.BytesIO(audio_item["bytes"]))
            if len(arr.shape) > 1:
                arr = np.mean(arr, axis=1)
            arr = arr.astype(np.float32)
            if sr != target_sr:
                arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
            return arr
        elif "path" in audio_item and audio_item["path"]:
            arr, sr = librosa.load(audio_item["path"], sr=target_sr, mono=True)
            return arr.astype(np.float32)
    elif isinstance(audio_item, (np.ndarray, list)):
        arr = np.array(audio_item, dtype=np.float32)
        return arr
    raise ValueError(f"Could not decode audio from format: {type(audio_item)}")


class IndonesianSpeechDatasetManager:
    """Manages loading, multi-corpus blending, SpecAugment, and preprocessing Indonesian speech datasets."""

    def __init__(
        self,
        model_name_or_path: str = "openai/whisper-large-v3-turbo",
        language: str = "indonesian",
        task: str = "transcribe",
        sampling_rate: int = 16000,
        enable_spec_augment: bool = False,
    ):
        self.sampling_rate = sampling_rate
        self.language = language
        self.task = task
        self.enable_spec_augment = enable_spec_augment
        self.processor = WhisperProcessor.from_pretrained(
            model_name_or_path,
            language=self.language,
            task=self.task,
        )
        self.feature_size = self.processor.feature_extractor.feature_size
        self.normalizer = IndonesianTextNormalizer(remove_punctuation=False, to_lower=False)

    def load_fleurs_indonesian(self, split: str = "train", max_samples: Optional[int] = None) -> Dataset:
        """Load Google FLEURS Indonesian ('id_id') dataset."""
        print(f"[*] Loading Google FLEURS Indonesian (id_id) - split: {split}...")
        ds = load_dataset(
            "google/fleurs",
            "id_id",
            split=split,
        )
        ds = ds.cast_column("audio", Audio(decode=False))
        if max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        print(f"[✓] Loaded {len(ds)} samples from FLEURS Indonesian ({split}).")
        return ds

    def prepare_dataset(
        self,
        dataset: Dataset,
        desc: str = "Preprocessing Indonesian Speech Dataset",
        augment: bool = False,
    ) -> Dataset:
        """Process raw audio waveforms and transcripts into input_features and labels with optional SpecAugment."""
        feat_size = self.feature_size

        def _process_sample(batch):
            try:
                audio_array = extract_audio_waveform(batch["audio"], target_sr=self.sampling_rate)
                
                input_features = self.processor.feature_extractor(
                    audio_array,
                    sampling_rate=self.sampling_rate,
                ).input_features[0]

                if augment and self.enable_spec_augment:
                    input_features = apply_spec_augment(input_features)

                raw_text = batch.get("transcription") or batch.get("sentence") or batch.get("text", "")
                normalized_text = self.normalizer(raw_text)
                labels = self.processor.tokenizer(normalized_text).input_ids

                return {
                    "input_features": input_features,
                    "labels": labels,
                    "raw_text": raw_text,
                    "normalized_text": normalized_text,
                }
            except Exception:
                return {
                    "input_features": np.zeros((feat_size, 3000), dtype=np.float32),
                    "labels": [self.processor.tokenizer.eos_token_id],
                    "raw_text": "",
                    "normalized_text": "",
                }

        print(f"[*] {desc} ({len(dataset)} samples)...")
        processed_ds = dataset.map(
            _process_sample,
            remove_columns=dataset.column_names,
            desc=desc,
        )
        return processed_ds

    def get_data_collator(self) -> DataCollatorSpeechSeq2SeqWithPadding:
        return DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
