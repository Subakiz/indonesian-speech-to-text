"""Unit tests for Apple Neural Engine (ANE) Inference Engine."""

import os
import numpy as np
import pytest
from ane.inference_ane import IndonesianANEInferenceEngine


def test_ane_inference_execution():
    encoder_path = "./checkpoints/WhisperEncoder_ANE.mlpackage"
    if not os.path.exists(encoder_path):
        pytest.skip("CoreML ANE model not yet exported.")

    engine = IndonesianANEInferenceEngine(
        coreml_encoder_path=encoder_path,
        hf_model_path="openai/whisper-tiny",
    )

    # 2 seconds synthetic speech waveform
    audio_test = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 16000 * 2)).astype(np.float32)
    result = engine.transcribe(audio_test, sampling_rate=16000)

    assert "transcription" in result
    assert "ane_encoder_latency_ms" in result
    assert result["ane_encoder_latency_ms"] > 0
    assert result["real_time_factor_rtf"] < 1.0  # Must be faster than real-time
