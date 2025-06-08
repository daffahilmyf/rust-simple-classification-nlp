use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

pub fn build_vocab(corpus: &[String], max_vocab: usize) -> Vec<String> {
    let mut freq = HashMap::new();
    for text in corpus {
        for word in super::tokenize::tokenize(text) {
            *freq.entry(word).or_insert(0) += 1;
        }
    }
    let mut sorted: Vec<_> = freq.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    sorted.into_iter().take(max_vocab).map(|(w, _)| w).collect()
}

pub fn vectorize(text: &str, vocab: &[String]) -> Vec<f32> {
    let tokens = super::tokenize::tokenize(text);
    let mut vec = vec![0.0; vocab.len()];
    for tok in tokens {
        if let Some(pos) = vocab.iter().position(|w| w == &tok) {
            vec[pos] += 1.0;
        }
    }
    vec
}

pub fn save(vocab: &[String], path: &str) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    let json = serde_json::to_string(vocab)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}
