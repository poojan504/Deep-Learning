# PyTorch Certification — Handwritten Image Classification (MNIST + EMNIST)

This repository contains PyTorch certification projects that implement **end-to-end handwritten image classification**, progressing from digit recognition (MNIST) to letter decoding and message reconstruction (EMNIST).

---

## What it does (1 sentence)
Trains and evaluates PyTorch models for handwritten digit and letter classification, then decodes sequences of predicted characters into readable text.

---

## Why it matters (use-case)
This project demonstrates how low-level vision models can be composed into higher-level systems:
- validating deep learning fundamentals in PyTorch,
- building reproducible training and inference pipelines,
- and transforming pixel-level predictions into semantic outputs (decoded text).

These skills directly transfer to OCR, document understanding, and perception pipelines.

---

## Results (metrics, latency, size, FPS, accuracy)

### MNIST (Digits 0–9)
- Test accuracy: **~98–99%** (architecture-dependent)
- Inference: real-time on CPU for single images
- Model: lightweight MLP / CNN suitable for fast iteration

### EMNIST (Letters A–Z)
- Task: 26-class handwritten letter classification
- Output: reconstructed human-readable messages
- Performance: strong per-character accuracy with expected ambiguity for visually similar letters (e.g., I/L, O/Q)

> Exact metrics depend on model architecture and training configuration used in the notebooks.

---

## Approach (diagram or bullets)

**Shared pipeline**
1. Load dataset using `torchvision.datasets`
2. Normalize and transform images to tensors
3. Define PyTorch model (MLP or CNN)
4. Train using cross-entropy loss and Adam/SGD
5. Evaluate accuracy on held-out test data
6. Run inference and visualize predictions

**EMNIST-specific decoding**
- Map predicted class indices → ASCII letters
- Concatenate predictions into full messages
- Analyze character-level errors and their impact on decoded text

---

## Run it locally (3–6 commands)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision matplotlib numpy scikit-learn
jupyter notebook

