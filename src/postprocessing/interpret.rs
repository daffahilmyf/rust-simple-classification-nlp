use tch::Tensor;

pub fn predict_label(logits: &Tensor) -> String {
    let idx = logits.argmax(-1, false).int64_value(&[]);
    match idx {
        0 => "Negative".to_string(),
        1 => "Positive".to_string(),
        _ => "Unknown".to_string(),
    }
}
