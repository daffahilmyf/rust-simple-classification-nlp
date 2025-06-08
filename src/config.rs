use std::env;

pub struct Config {
    pub dataset_path: String,
    pub testset_path: String,
    pub model_path: String,
    pub vocab_path: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub max_vocab: usize,
    pub hidden_dim: i64,
}

impl Config {
    pub fn load() -> Self {
        dotenv::dotenv().ok();

        Self {
            dataset_path: env::var("DATASET_PATH").unwrap_or_else(|_| "data/imbd".to_string()),
            testset_path: env::var("TESTSET_PATH").unwrap_or_else(|_| "data/imbd/test".to_string()),
            model_path: env::var("MODEL_PATH").unwrap_or_else(|_| "output/model.ot".to_string()),
            vocab_path: env::var("VOCAB_PATH").unwrap_or_else(|_| "output/vocab.json".to_string()),
            epochs: env::var("EPOCHS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(50),
            batch_size: env::var("BATCH_SIZE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32),
            max_vocab: env::var("MAX_VOCAB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5000),
            hidden_dim: env::var("HIDDEN_DIM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64),
        }
    }
}
