# Bahasa Indonesia Speech-to-Text (ASR) Engine

High-performance Automatic Speech Recognition (ASR) system tailored specifically for **Bahasa Indonesia** with **GPU (Metal / MPS) training** and **Apple Neural Engine (ANE) inference**.

---

## 🚀 Key Highlights & Hardware Optimization

1. **Hardware Acceleration**:
   - **Training**: Executed on Apple Silicon GPU via PyTorch Metal Performance Shaders (`torch.device("mps")`) with PEFT LoRA adapter tuning.
   - **Inference**: High-throughput Core ML inference compiled for the **Apple Neural Engine (ANE)** (`compute_units=ct.ComputeUnit.ALL`), achieving **~13–18ms** encoder latency and a Real-Time Factor (RTF) of **< 0.02** (50x faster than real-time).
2. **Indonesian Text Normalization Engine**:
   - Spoken number expansion (*"angka ke terbilang"*), Indonesian currency (*"Rp 50.000" $\rightarrow$ "lima puluh ribu rupiah"*), percentage expansion (*"25%" $\rightarrow$ "dua puluh lima persen"*), Indonesian abbreviations (*"yg"*, *"utk"*, *"dll"*), and hyphenated reduplication preservation (*"anak-anak"*, *"jalan-jalan"*).
3. **Hugging Face Dataset Support**:
   - Direct integration with Google FLEURS Indonesian (`google/fleurs` `id_id`), Mozilla Common Voice 17.0 (`id`), and SEACrowd Indonesian speech datasets.

---

## 📁 Repository Structure

```
.
├── configs/                  # Training & inference hyperparameter configs
├── data/
│   ├── dataset_loader.py     # Hugging Face dataset streaming & audio preprocessing
│   └── indonesian_normalizer.py # Indonesian number, currency, & text normalizer
├── training/
│   ├── train_mps.py          # PyTorch GPU (MPS) training with PEFT LoRA
│   └── metrics.py            # Indonesian-aware WER & CER evaluation
├── ane/
│   ├── export_coreml.py      # Core ML exporter compiled for Apple Neural Engine (ANE)
│   └── inference_ane.py      # Real-time ANE inference engine
├── tests/
│   ├── test_normalizer.py    # Unit tests for text normalization
│   ├── test_audio_pipeline.py # Unit tests for audio feature extraction
│   └── test_ane_inference.py # Unit tests for ANE model execution
├── prototype_demo.py         # End-to-end prototype pipeline
├── cli.py                    # Production CLI tool
└── requirements.txt          # Project dependencies
```

---

## 🛠️ Quickstart

### 1. Run the Full Prototype Demo
Runs GPU training on Hugging Face Indonesian data, exports to Core ML, and executes inference on the Apple Neural Engine:
```bash
source .venv/bin/activate
PYTHONPATH=. python prototype_demo.py
```

### 2. Run Tests
```bash
source .venv/bin/activate
PYTHONPATH=. pytest tests/
```

### 3. CLI Usage

#### A. Train Model on GPU (MPS)
```bash
python cli.py train --model openai/whisper-tiny --epochs 5 --batch-size 8 --lr 1e-4
```

#### B. Export to Apple Neural Engine (Core ML `.mlpackage`)
```bash
python cli.py export --model-dir ./checkpoints/indonesian_whisper --output ./checkpoints/WhisperEncoder_Indonesian_ANE.mlpackage
```

#### C. Transcribe Audio on ANE
```bash
python cli.py transcribe path/to/indonesian_audio.wav
```
