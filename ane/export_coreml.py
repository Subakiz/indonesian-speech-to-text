"""Apple Neural Engine (ANE) Core ML Exporter for Whisper ASR (supports Turbo, Large-v3, Small, Tiny)."""

import os
import torch
import torch.nn as nn
import coremltools as ct
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class WhisperEncoderWrapper(nn.Module):
    """Wraps Whisper's audio encoder into a clean PyTorch module for Core ML / ANE tracing."""

    def __init__(self, whisper_encoder):
        super().__init__()
        self.encoder = whisper_encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        # input_features: [1, num_mel_bins, 3000]
        encoder_outputs = self.encoder(input_features)
        # Returns last_hidden_state: [1, 1500, hidden_dim]
        return encoder_outputs.last_hidden_state


def export_whisper_encoder_to_ane(
    model_path: str,
    output_mlpackage_path: str = "./checkpoints/WhisperEncoder_Large_Turbo_ANE.mlpackage",
    compute_precision: ct.precision = ct.precision.FLOAT16,
) -> str:
    """Convert Whisper Encoder (Large-v3-Turbo / Small / Tiny) to Apple Core ML (.mlpackage) optimized for ANE."""
    print(f"[*] Loading model from {model_path} for ANE export...")
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    num_mel_bins = getattr(model.config, "num_mel_bins", 80)
    print(f"[*] Detected mel bins: {num_mel_bins} (128 for Large-v3/Turbo, 80 for Small/Tiny)")

    encoder_wrapper = WhisperEncoderWrapper(model.model.encoder)
    encoder_wrapper.eval()

    # Dummy input: 1 audio chunk of 30 seconds at 16kHz -> [1, num_mel_bins, 3000]
    dummy_input = torch.randn(1, num_mel_bins, 3000, dtype=torch.float32)

    print("[*] Tracing PyTorch Whisper Encoder module...")
    with torch.no_grad():
        traced_model = torch.jit.trace(encoder_wrapper, dummy_input)

    print("[*] Converting to Apple Core ML with ANE-targeted FP16 compute precision...")
    input_shape = (1, num_mel_bins, 3000)
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_features", shape=input_shape, dtype=float)],
        outputs=[ct.TensorType(name="last_hidden_state", dtype=float)],
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.macOS13,
    )

    coreml_model.author = "Antigravity Indonesian ASR"
    coreml_model.short_description = f"Whisper Encoder ({model_path}) for Apple Neural Engine (ANE)"
    coreml_model.input_description["input_features"] = f"Log-Mel Spectrogram of 16kHz audio [1, {num_mel_bins}, 3000]"
    coreml_model.output_description["last_hidden_state"] = "Encoded audio embeddings [1, 1500, hidden_dim]"

    os.makedirs(os.path.dirname(os.path.abspath(output_mlpackage_path)), exist_ok=True)
    coreml_model.save(output_mlpackage_path)
    print(f"[✓] Successfully exported ANE Core ML package to: {output_mlpackage_path}")

    return output_mlpackage_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="openai/whisper-large-v3-turbo")
    parser.add_argument("--output_path", type=str, default="./checkpoints/WhisperEncoder_Large_Turbo_ANE.mlpackage")
    args = parser.parse_args()

    export_whisper_encoder_to_ane(args.model_path, args.output_path)
