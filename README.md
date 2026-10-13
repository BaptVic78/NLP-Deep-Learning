# RNN / LSTM text-classification lab

Visualization of two existing exercises:

- **Fake-news dataset classification**: Embedding → BiLSTM(64) → BiLSTM(16) → Dense layers.
- **Emotion classification**: Embedding → BiLSTM(64) → BiLSTM(32) → six-class softmax.

These LSTM architectures are already implemented in the original notebooks, which remain unchanged. The added Python scripts make their steps explicit and save models for Streamlit. No stock-price exercise is included in the current project.

## Run locally on Ubuntu

```bash
cd /home/victor/NLP/lab3-RNN
bash install.sh  # first installation only
bash run.sh
```

Open http://localhost:8504. Stop the interface with Ctrl+C. Prepared models are loaded without retraining on each click.

## Train

```bash
venv/bin/python train_news.py
venv/bin/python train_emotions.py
```

The scripts follow: load → clean → split → tokenizer → padding → notebook model → train → evaluate → save. Defaults use ten epochs. EPOCHS and SAMPLES can limit a quick CPU demonstration; the interface shows the actual sizes and epochs. The prepared runs use three epochs, 6000 news samples and 8000 emotion training samples.

## Visualization

Enter English text, inspect its predicted dataset class or emotion, and view class scores. Evaluation includes test accuracy, macro F1, confusion matrices and actual training/validation curves. The method tab explains the existing architecture.

News labels describe this dataset; the interface is not a factual verification service. Probabilities are model scores, not calibrated guarantees. The news notebook's linear output is a logit: sigmoid is applied before thresholding at 0.5. Metrics use true labels first, correcting reversed arguments in the notebook's evaluation cells.

Normalization and duplicate/conflict removal occur before splitting; the tokenizer is fitted only on train. Emotion uses the official validation/test splits, with normalized duplicates and overlaps removed.

The original notebooks are the reference for your lab work. The saved demo models are newly trained from their architectures, not recovered from runtime memory. The TSVs contain embeddings only and cannot replace a complete model.

The course PDF and generated embedding TSVs are excluded from Git. CSV inputs and prepared models are included to reproduce the visualization locally.
