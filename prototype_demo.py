"""End-to-End Prototype Demo: Bahasa Indonesia Speech-to-Text.

Pipeline:
1. Load real Indonesian speech samples from Hugging Face (`google/fleurs` `id_id`).
2. Fine-tune on Apple Silicon GPU (`mps`) using PyTorch + PEFT LoRA.
3. Export fine-tuned model to Core ML `.mlpackage` for Apple Neural Engine (ANE).
4. Run real-time transcription on Apple Neural Engine (ANE).
5. Evaluate Ground Truth vs Prediction with Indonesian WER/CER & Latency profiling.
"""

import os
import time
import torch
import numpy as np
from datasets import load_dataset, Audio
from data.dataset_loader import IndonesianSpeechDatasetManager, extract_audio_waveform
from data.indonesian_normalizer import IndonesianTextNormalizer
from training.train_mps import IndonesianWhisperTrainer
from training.metrics import IndonesianASRMetrics
from ane.export_coreml import export_whisper_encoder_to_ane
from ane.inference_ane import IndonesianANEInferenceEngine


def run_prototype_pipeline():
    print("=" * 80)
    print("    PROTOTYPE: INDONESIAN SPEECH-TO-TEXT (ASR)")
    print("    Training: Apple Silicon GPU (MPS) | Inference: Apple Neural Engine (ANE)")
    print("=" * 80)

    checkpoint_dir = "./checkpoints/indonesian_whisper_prototype"
    coreml_path = "./checkpoints/WhisperEncoder_Indonesian_ANE.mlpackage"

    # -------------------------------------------------------------
    # Step 1: Train/Fine-Tune on GPU (MPS) with Hugging Face Indonesian Data
    # -------------------------------------------------------------
    print("\n[PHASE 1] Fine-Tuning Whisper on GPU (MPS) with Hugging Face Indonesian Data...")
    trainer = IndonesianWhisperTrainer(
        model_name="openai/whisper-tiny",
        output_dir=checkpoint_dir,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        learning_rate=2e-4,
        batch_size=4,
        gradient_accumulation_steps=1,
        num_epochs=1,
        max_train_samples=8,
        max_val_samples=4,
    )
    trained_model_dir = trainer.train()

    # -------------------------------------------------------------
    # Step 2: Export Fine-Tuned Model to Core ML for Apple Neural Engine (ANE)
    # -------------------------------------------------------------
    print("\n[PHASE 2] Exporting Fine-Tuned Model to Apple Neural Engine (ANE) Format...")
    export_whisper_encoder_to_ane(
        model_path=trained_model_dir,
        output_mlpackage_path=coreml_path,
    )

    # -------------------------------------------------------------
    # Step 3: Run Inference on Apple Neural Engine (ANE) with Real Audio
    # -------------------------------------------------------------
    print("\n[PHASE 3] Running Real-Time Indonesian Inference on Apple Neural Engine (ANE)...")
    ane_engine = IndonesianANEInferenceEngine(
        coreml_encoder_path=coreml_path,
        hf_model_path=trained_model_dir,
    )

    # Load test Indonesian speech samples from FLEURS
    print("[*] Fetching real Indonesian test samples from Hugging Face (google/fleurs id_id)...")
    test_ds = load_dataset("google/fleurs", "id_id", split="test")
    test_ds = test_ds.cast_column("audio", Audio(decode=False))
    
    normalizer = IndonesianTextNormalizer(remove_punctuation=True, to_lower=True)
    metrics_calc = IndonesianASRMetrics(tokenizer=ane_engine.processor.tokenizer)

    results_table = []
    predictions = []
    references = []

    print("\n" + "-" * 80)
    print(f"{'IDX':<4} | {'AUDIO (s)':<10} | {'ANE TIME':<10} | {'TOTAL TIME':<10} | {'RTF':<8}")
    print("-" * 80)

    num_eval_samples = min(3, len(test_ds))
    for i in range(num_eval_samples):
        sample = test_ds[i]
        audio_array = extract_audio_waveform(sample["audio"], target_sr=16000)
        raw_reference = sample["transcription"]

        # Run ANE transcription
        result = ane_engine.transcribe(audio_array, sampling_rate=16000)

        predictions.append(result["transcription"])
        references.append(raw_reference)

        print(f"{i+1:<4} | {result['duration_sec']:<10} | {result['ane_encoder_latency_ms']:<7.1f} ms | {result['total_latency_ms']:<7.1f} ms | {result['real_time_factor_rtf']:<8.4f}")

        results_table.append({
            "index": i + 1,
            "duration": result["duration_sec"],
            "ane_time": result["ane_encoder_latency_ms"],
            "total_time": result["total_latency_ms"],
            "rtf": result["real_time_factor_rtf"],
            "reference": raw_reference,
            "prediction": result["transcription"],
        })

    # -------------------------------------------------------------
    # Step 4: Quality & Performance Evaluation
    # -------------------------------------------------------------
    metrics = metrics_calc.compute_metrics_from_text(predictions, references)

    print("\n" + "=" * 80)
    print("                     EVALUATION & TRANSCRIPTION RESULTS")
    print("=" * 80)
    for res in results_table:
        print(f"\n[Sample #{res['index']}] Audio Duration: {res['duration']}s | ANE Latency: {res['ane_time']:.1f}ms | RTF: {res['rtf']:.4f}")
        print(f"  Reference (Ground Truth) : {res['reference']}")
        print(f"  ANE Prediction           : {res['prediction']}")

    print("\n" + "-" * 80)
    print(f"  Aggregate Normalized WER : {metrics['normalized_wer']}%")
    print(f"  Aggregate Normalized CER : {metrics['normalized_cer']}%")
    print(f"  Raw WER                 : {metrics['raw_wer']}%")
    print("=" * 80)
    print("\n[✓] Prototype pipeline successfully executed on Apple Silicon GPU & Neural Engine (ANE)!")


if __name__ == "__main__":
    run_prototype_pipeline()
