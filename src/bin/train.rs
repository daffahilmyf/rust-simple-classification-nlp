use rand::seq::SliceRandom;
use rand::thread_rng;

use simple_nlp_rust::{
    config::Config,
    io::dataset::load_dataset,
    model::classifier::build_model,
    model::train_utils::train,
    postprocessing::metrics::{confusion_matrix, precision_recall_f1, print_confusion_matrix},
    preprocessing::vocab,
};
use std::env;
use tch::nn::VarStore;
use tch::{Kind, Tensor, nn::Module};

fn main() -> anyhow::Result<()> {
    let config = Config::load();

    println!("Load dataset");
    // Load and shuffle training data
    let mut train_reviews = load_dataset(&format!("{}/train", config.dataset_path))?;
    train_reviews.shuffle(&mut thread_rng());

    let train_texts: Vec<_> = train_reviews.iter().map(|r| r.text.clone()).collect();
    let train_labels: Vec<i64> = train_reviews.iter().map(|r| r.label).collect();

    // Load and shuffle test data
    let mut test_reviews = load_dataset(&format!("{}/test", config.dataset_path))?;
    test_reviews.shuffle(&mut thread_rng());

    let test_texts: Vec<_> = test_reviews.iter().map(|r| r.text.clone()).collect();
    let test_labels: Vec<i64> = test_reviews.iter().map(|r| r.label).collect();

    println!("Load hyperparams");
    // Hyperparameters from env or defaults
    let epochs: i64 = env::var("EPOCHS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let hidden_dim: i64 = env::var("HIDDEN_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let max_vocab: usize = env::var("MAX_VOCAB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5000);

    // Build vocab
    let vocab_vec = vocab::build_vocab(&train_texts, max_vocab);
    vocab::save(&vocab_vec, &config.vocab_path)?;

    // Convert to tensors (train)
    let xs_train: Vec<_> = train_texts
        .iter()
        .map(|t| {
            Tensor::f_from_slice(&vocab::vectorize(t, &vocab_vec))
                .expect("Failed to convert train tensor")
                .unsqueeze(0)
        })
        .collect();
    let xs_train = Tensor::cat(&xs_train, 0);
    let ys_train = Tensor::f_from_slice(&train_labels)?.to_kind(Kind::Int64);

    // Convert to tensors (test)
    let xs_test: Vec<_> = test_texts
        .iter()
        .map(|t| {
            Tensor::f_from_slice(&vocab::vectorize(t, &vocab_vec))
                .expect("Failed to convert test tensor")
                .unsqueeze(0)
        })
        .collect();
    let xs_test = Tensor::cat(&xs_test, 0);
    let ys_test: Tensor = Tensor::f_from_slice(&test_labels)?.to_kind(Kind::Int64);

    // Initialize model
    let vs = VarStore::new(tch::Device::Cpu);
    let root = vs.root();
    let model = build_model(&root, xs_train.size()[1], hidden_dim, 2);

    train(&model, &vs, &xs_train, &ys_train, epochs);
    vs.save(&config.model_path)?;

    // Prediction
    let predictions: Vec<i64> = xs_test
        .unbind(0)
        .into_iter()
        .map(|x| {
            let out = model.forward(&x);
            out.argmax(-1, false).int64_value(&[])
        })
        .collect();

    // Convert true labels from Tensor to Vec<i64>
    let true_labels: Vec<i64> = ys_test.iter::<i64>()?.collect();

    // Evaluation
    let matrix = confusion_matrix(&true_labels, &predictions);
    print_confusion_matrix(&matrix);

    let (precision, recall, f1) = precision_recall_f1(&true_labels, &predictions);
    println!(
        "Precision: {:.4}, Recall: {:.4}, F1 Score: {:.4}",
        precision, recall, f1
    );

    Ok(())
}
