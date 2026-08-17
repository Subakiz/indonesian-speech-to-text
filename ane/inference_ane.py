"""Apple Neural Engine (ANE) Accelerated Speech-to-Text Inference Engine with Beam Search."""

import time
import os
from typing import Dict, Optional, Union, Tuple, List
import numpy as np
import torch
import soundfile as sf
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from data.indonesian_normalizer import IndonesianTextNormalizer

try:
    import coremltools as ct
    HAS_COREML = True
except ImportError:
    HAS_COREML = False


class IndonesianANEInferenceEngine:
    """End-to-End Indonesian ASR Inference Engine running audio encoding on Apple Neural Engine (ANE) with Beam Search."""

    def __init__(
        self,
        coreml_encoder_path: str = "./checkpoints/WhisperEncoder_Indonesian_Full_ANE.mlpackage",
        hf_model_path: str = "./checkpoints/indonesian_whisper_full",
        compute_unit = None,
        language: str = "indonesian",
        task: str = "transcribe",
        num_beams: int = 5,
    ):
        self.language = language
        self.task = task
        self.num_beams = num_beams
        self.coreml_encoder = None

        if HAS_COREML and os.path.exists(coreml_encoder_path):
            cu = compute_unit if compute_unit is not None else ct.ComputeUnit.ALL
            print(f"[*] Initializing Core ML Engine on Apple Neural Engine...")
            t0 = time.time()
            self.coreml_encoder = ct.models.MLModel(
                coreml_encoder_path,
                compute_units=cu,
            )
            print(f"[✓] Core ML model loaded on ANE in {time.time() - t0:.3f}s")
        else:
            print("[*] Running in standard PyTorch mode (ANE Core ML model not loaded).")

        print(f"[*] Loading Whisper Decoder & Processor from {hf_model_path}...")
        self.processor = WhisperProcessor.from_pretrained(hf_model_path)
        self.decoder_model = WhisperForConditionalGeneration.from_pretrained(hf_model_path)
        self.decoder_model.eval()

        self.normalizer = IndonesianTextNormalizer(remove_punctuation=False, to_lower=False)
        self.eval_normalizer = IndonesianTextNormalizer(remove_punctuation=True, to_lower=True)

    def load_and_preprocess_audio(self, audio_input: Union[str, np.ndarray], sampling_rate: int = 16000) -> Tuple[np.ndarray, float]:
        """Load, resample to 16kHz mono, and calculate audio duration."""
        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                raise FileNotFoundError(f"Audio file not found: {audio_input}")
            audio_array, sr = librosa.load(audio_input, sr=sampling_rate, mono=True)
        elif isinstance(audio_input, np.ndarray):
            audio_array = audio_input
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

        audio_array = audio_array.astype(np.float32)
        duration_sec = len(audio_array) / sampling_rate
        return audio_array, duration_sec

    def transcribe(
        self,
        audio_input: Union[str, np.ndarray],
        sampling_rate: int = 16000,
        apply_normalization: bool = True,
        num_beams: Optional[int] = None,
    ) -> Dict[str, Union[str, float]]:
        """Run speech-to-text inference with ANE / GPU acceleration and beam search."""
        t_start = time.perf_counter()
        beams = num_beams if num_beams is not None else self.num_beams

        audio_array, duration_sec = self.load_and_preprocess_audio(audio_input, sampling_rate=sampling_rate)

        input_features = self.processor.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="np",
        ).input_features

        t_ane_start = time.perf_counter()
        if self.coreml_encoder is not None:
            ane_outputs = self.coreml_encoder.predict({"input_features": input_features})
            ane_time_ms = (time.perf_counter() - t_ane_start) * 1000.0
            output_key = list(ane_outputs.keys())[0]
            encoder_hidden_states = torch.tensor(ane_outputs[output_key], dtype=torch.float32)
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        else:
            pt_features = torch.tensor(input_features, dtype=torch.float32)
            with torch.no_grad():
                encoder_outputs = self.decoder_model.model.encoder(pt_features)
            ane_time_ms = (time.perf_counter() - t_ane_start) * 1000.0

        t_dec_start = time.perf_counter()
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language,
            task=self.task,
        )

        with torch.no_grad():
            generated_ids = self.decoder_model.generate(
                encoder_outputs=encoder_outputs,
                forced_decoder_ids=forced_decoder_ids,
                num_beams=beams,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                max_new_tokens=128,
            )

        decoding_time_ms = (time.perf_counter() - t_dec_start) * 1000.0
        total_time_ms = (time.perf_counter() - t_start) * 1000.0

        transcription_raw = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        transcription_normalized = self.normalizer(transcription_raw) if apply_normalization else transcription_raw
        rtf = (total_time_ms / 1000.0) / max(duration_sec, 0.001)

        return {
            "transcription": transcription_normalized,
            "transcription_raw": transcription_raw,
            "duration_sec": round(duration_sec, 2),
            "num_beams": beams,
            "ane_encoder_latency_ms": round(ane_time_ms, 2),
            "decoder_latency_ms": round(decoding_time_ms, 2),
            "total_latency_ms": round(total_time_ms, 2),
            "real_time_factor_rtf": round(rtf, 4),
            "device": "Apple Neural Engine (ANE)" if self.coreml_encoder else "PyTorch CUDA/CPU",
        }
