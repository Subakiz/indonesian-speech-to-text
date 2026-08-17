"""Full Training and ANE Core ML Export Pipeline using Whisper-Large-v3-Turbo for Bahasa Indonesia."""

import os
import time
import argparse
from training.train_mps import IndonesianWhisperTrainer
from ane.export_coreml import export_whisper_encoder_to_ane
from ane.inference_ane import IndonesianANEInferenceEngine
from data.indonesian_normalizer import IndonesianTextNormalizer
from training.metrics import IndonesianASRMetrics
from data.dataset_loader import extract_audio_waveform
from datasets import load_dataset, Audio


def main():
    parser = argparse.ArgumentParser(description="Indonesian Speech-to-Text with Whisper-Large-v3-Turbo")
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3-turbo", help="Base Whisper model")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Micro batch size")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/indonesian_whisper_turbo_sota", help="Checkpoint dir")
    parser.add_argument("--ane_output_path", type=str, default="./checkpoints/WhisperEncoder_Indonesian_Turbo_ANE.mlpackage", help="ANE package")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Train samples limit")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Val samples limit")
    args = parser.parse_args()

    print("=" * 80, flush=True)
    print("   STATE-OF-THE-ART INDONESIAN SPEECH-TO-TEXT (ASR)", flush=True)
    print(f"   Backbone: {args.model} | Epochs: {args.epochs} | LoRA Rank: {args.lora_r}", flush=True)
    print("=" * 80, flush=True)

    # 1. Full Training on GPU (MPS) with Whisper-Large-v3-Turbo
    t0 = time.time()
    trainer = IndonesianWhisperTrainer(
        model_name=args.model,
        output_dir=args.checkpoint_dir,
        use_lora=True,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_epochs=args.epochs,
        enable_spec_augment=True,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    saved_model_path = trainer.train()

    # 2. Export Whisper-Large-v3-Turbo to Apple Core ML for ANE
    print("\n" + "=" * 80, flush=True)
    print("[*] Exporting Whisper-Large-v3-Turbo to Apple Neural Engine (ANE)...", flush=True)
    print("=" * 80, flush=True)
    export_whisper_encoder_to_ane(
        model_path=saved_model_path,
        output_mlpackage_path=args.ane_output_path,
    )

    # 3. Benchmark on Indonesian Test Split using ANE Engine with Beam Search
    print("\n" + "=" * 80, flush=True)
    print("[*] Benchmarking Fine-Tuned Whisper-Large-v3-Turbo on Apple Neural Engine (ANE)...", flush=True)
    print("=" * 80, flush=True)
    ane_engine = IndonesianANEInferenceEngine(
        coreml_encoder_path=args.ane_output_path,
        hf_model_path=saved_model_path,
        num_beams=5,
    )

    test_ds = load_dataset("google/fleurs", "id_id", split="test")
    test_ds = test_ds.cast_column("audio", Audio(decode=False))
    metrics_calc = IndonesianASRMetrics(tokenizer=ane_engine.processor.tokenizer)

    num_eval = min(5, len(test_ds))
    predictions = []
    references = []
    latencies = []

    print(f"[*] Running ANE Beam Search (k=5) evaluation across {num_eval} test samples...\n", flush=True)
    for i in range(num_eval):
        sample = test_ds[i]
        audio_array = extract_audio_waveform(sample["audio"], target_sr=16000)
        ref = sample["transcription"]

        res = ane_engine.transcribe(audio_array, sampling_rate=16000, num_beams=5)
        predictions.append(res["transcription"])
        references.append(ref)
        latencies.append(res["ane_encoder_latency_ms"])

        print(f"[{i+1}/{num_eval}] Audio: {res['duration_sec']}s | ANE Encoder: {res['ane_encoder_latency_ms']:.1f}ms | Total: {res['total_latency_ms']:.1f}ms | RTF: {res['real_time_factor_rtf']:.4f}", flush=True)
        print(f"  Reference : {ref}", flush=True)
        print(f"  Predicted : {res['transcription']}\n", flush=True)

    metrics = metrics_calc.compute_metrics_from_text(predictions, references)
    avg_ane_latency = sum(latencies) / len(latencies)

    print("=" * 80, flush=True)
    print("                 FINAL SOTA BENCHMARK RESULTS", flush=True)
    print("=" * 80, flush=True)
    print(f"  Model Architecture       : {args.model}", flush=True)
    print(f"  Model Saved To           : {saved_model_path}", flush=True)
    print(f"  ANE Package Saved To     : {args.ane_output_path}", flush=True)
    print(f"  Average ANE Latency      : {avg_ane_latency:.2f} ms", flush=True)
    print(f"  Normalized WER           : {metrics['normalized_wer']}%", flush=True)
    print(f"  Normalized CER           : {metrics['normalized_cer']}%", flush=True)
    print(f"  Raw WER                  : {metrics['raw_wer']}%", flush=True)
    print(f"  Total Pipeline Time      : {time.time() - t0:.1f}s", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
