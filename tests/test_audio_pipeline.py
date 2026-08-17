"""Unit tests for Indonesian Audio Processing Pipeline."""

import numpy as np
import pytest
from transformers import WhisperProcessor
from data.dataset_loader import IndonesianSpeechDatasetManager


def test_dataset_manager_initialization():
    mgr = IndonesianSpeechDatasetManager(model_name_or_path="openai/whisper-tiny")
    assert mgr.sampling_rate == 16000
    assert mgr.language == "indonesian"
    assert mgr.task == "transcribe"
    assert mgr.processor is not None


def test_feature_extraction_dimensions():
    mgr = IndonesianSpeechDatasetManager(model_name_or_path="openai/whisper-tiny")
    # Generate 5 seconds of 16kHz audio
    audio_5s = np.zeros(16000 * 5, dtype=np.float32)
    features = mgr.processor.feature_extractor(
        audio_5s,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features

    # Whisper expects shape [batch_size, n_mels=80, seq_len=3000]
    assert features.shape == (1, 80, 3000)
