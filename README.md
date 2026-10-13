# RNN / LSTM lab

Three exercises, with their original notebooks and a local Streamlit interface:

1. Google opening-price prediction from a window of 60 observed prices, using four LSTM layers.
2. Fake-news dataset classification using Embedding and bidirectional LSTM layers.
3. Six-class emotion classification using Embedding and bidirectional LSTM layers.

## Run on Ubuntu

```bash
bash install.sh
bash run.sh
```

Open http://localhost:8504. The prepared models are loaded without training on each click.

## Train

The scripts follow the notebooks in a straightforward order: load, clean/scale, split, build the model, train, evaluate and save.

```bash
venv/bin/python train_stock.py
venv/bin/python train_news.py
venv/bin/python train_emotions.py
```

Training duration can be changed with `EPOCHS`. News and emotion training can be limited with `SAMPLES` for a quick demonstration; the interface displays the sample sizes and actual training history. Defaults follow the notebook epoch counts. Prepared demonstration runs may use fewer epochs and are identified in their reports.

The notebooks remain unchanged. The news notebook has a linear output and a loss configured with `from_logits=True`; inference applies sigmoid and thresholds the resulting probability at 0.5. The saved report computes metrics with true labels first, correcting the notebook's reversed metric arguments.

Stock predictions are rolling one-step predictions using the previous observed prices, not a forecast of all future days from one initial window. A last-observation baseline is displayed. Historical prices are provided for this educational exercise; this is not a trading recommendation.

The fake-news labels describe the supplied dataset. A model score is not a factual verification of an arbitrary article. All text input is in English.

The course PDF and generated embedding TSVs are excluded from version control. The original CSV inputs and notebooks are preserved.
