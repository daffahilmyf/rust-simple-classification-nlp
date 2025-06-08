# Rust NLP Sentiment Classifier

A simple, fast, and production-grade sentiment analysis project built with **Rust** using the [IMDb Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

> "The reason I tried this: I got bored with Python ML frameworks and wanted something fast, static, and refreshing."

---

## Features

* 📦 Written in Rust, powered by `tch-rs` (PyTorch bindings)
* 🧠 Trainable binary sentiment classifier (Positive/Negative)
* 📊 Outputs confusion matrix, precision, recall, and F1 score
* ⚙️ Configurable training parameters via `.env`
* 📦 CI/CD-ready with GitHub Actions (build, test, lint, release)
* 📁 Clean, modular project structure

---

## Dataset

* IMDb Large Movie Review Dataset: [https://ai.stanford.edu/\~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)
* After extraction, expected structure:

```
data/
  imbd/
    train/
      pos/
      neg/
    test/
      pos/
      neg/
```

---

## .env Example

```
DATASET_PATH=data/imbd
TESTSET_PATH=data/imbd/test
MODEL_PATH=output/model.ot
VOCAB_PATH=output/vocab.json
EPOCHS=50
BATCH_SIZE=32
MAX_VOCAB=5000
HIDDEN_DIM=64
```

---

## Usage

### Train

```bash
cargo run --bin train
```

### Predict

```bash
cargo run --bin predict "I really enjoyed the movie."
# Prediction: Positive
```

---

## Project Structure

```
.
├── src/
│   ├── bin/              # Entry points: train.rs, predict.rs
│   ├── config.rs         # .env loader
│   ├── io/               # Dataset loading
│   ├── model/            # Classifier & training
│   ├── postprocessing/   # Evaluation metrics
│   └── preprocessing/    # Tokenizer, vectorizer
├── output/               # Model & vocab outputs
├── .env                  # Training config
├── Cargo.toml
└── README.md
```

---

## License

MIT — free to use, modify, and share.

---

Made with 🦀 because Python got boring.
