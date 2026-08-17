"""CLI for Indonesian Speech-to-Text: Training on GPU & Inference on Apple Neural Engine (ANE)."""

import argparse
import sys
import os
import json


def cmd_train(args):
    from training.train_mps import IndonesianWhisperTrainer
    print(f"[*] Starting Indonesian Whisper Training on GPU (MPS/CUDA)...")
    trainer = IndonesianWhisperTrainer(
        model_name=args.model,
        output_dir=args.output,
        use_lora=not args.full_finetune,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_epochs=args.epochs,
        max_train_samples=args.max_samples,
        max_val_samples=args.max_val_samples,
    )
    trainer.train()


def cmd_export(args):
    from ane.export_coreml import export_whisper_encoder_to_ane
    print(f"[*] Exporting model from {args.model_dir} to Core ML ANE package...")
    export_whisper_encoder_to_ane(
        model_path=args.model_dir,
        output_mlpackage_path=args.output,
    )


def cmd_transcribe(args):
    from ane.inference_ane import IndonesianANEInferenceEngine
    if not os.path.exists(args.audio):
        print(f"Error: Audio file '{args.audio}' not found.")
        sys.exit(1)

    engine = IndonesianANEInferenceEngine(
        coreml_encoder_path=args.ane_model,
        hf_model_path=args.model_dir,
    )

    print(f"\n[*] Transcribing '{args.audio}' on Apple Neural Engine (ANE)...")
    result = engine.transcribe(args.audio)

    print("\n" + "=" * 60)
    print("                TRANSCRIPTION RESULT")
    print("=" * 60)
    print(f"Transcript : {result['transcription']}")
    print("-" * 60)
    print(f"Audio Duration      : {result['duration_sec']}s")
    print(f"ANE Latency         : {result['ane_encoder_latency_ms']} ms")
    print(f"Total Latency       : {result['total_latency_ms']} ms")
    print(f"Real-Time Factor    : {result['real_time_factor_rtf']}x")
    print(f"Hardware Compute    : {result['device']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Indonesian Speech-to-Text Engine (GPU + ANE)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_p = subparsers.add_parser("train", help="Train / Fine-tune model on GPU (MPS)")
    train_p.add_argument("--model", type=str, default="openai/whisper-tiny", help="Base Hugging Face model")
    train_p.add_argument("--output", type=str, default="./checkpoints/indonesian_whisper", help="Output directory")
    train_p.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    train_p.add_argument("--batch-size", type=int, default=8, help="Batch size")
    train_p.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    train_p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_p.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    train_p.add_argument("--full-finetune", action="store_true", help="Disable LoRA and train all weights")
    train_p.add_argument("--max-samples", type=int, default=None, help="Limit training dataset samples")
    train_p.add_argument("--max-val-samples", type=int, default=None, help="Limit validation dataset samples")
    train_p.set_defaults(func=cmd_train)

    # Export command
    export_p = subparsers.add_parser("export", help="Export PyTorch model to Core ML for Apple Neural Engine (ANE)")
    export_p.add_argument("--model-dir", type=str, default="./checkpoints/indonesian_whisper", help="Model directory")
    export_p.add_argument("--output", type=str, default="./checkpoints/WhisperEncoder_Indonesian_ANE.mlpackage", help="Core ML output path")
    export_p.set_defaults(func=cmd_export)

    # Transcribe command
    trans_p = subparsers.add_parser("transcribe", help="Transcribe audio file on Apple Neural Engine")
    trans_p.add_argument("audio", type=str, help="Path to audio file (WAV/FLAC/MP3/M4A)")
    trans_p.add_argument("--ane-model", type=str, default="./checkpoints/WhisperEncoder_Indonesian_ANE.mlpackage", help="Core ML ANE package")
    trans_p.add_argument("--model-dir", type=str, default="./checkpoints/indonesian_whisper_prototype", help="Tokenizer and decoder weights")
    trans_p.set_defaults(func=cmd_transcribe)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
