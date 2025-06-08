use std::collections::HashMap;

pub fn confusion_matrix(true_labels: &[i64], pred_labels: &[i64]) -> HashMap<(i64, i64), usize> {
    let mut matrix = HashMap::new();
    for (&true_label, &pred_label) in true_labels.iter().zip(pred_labels.iter()) {
        *matrix.entry((true_label, pred_label)).or_insert(0) += 1;
    }
    matrix
}

pub fn precision_recall_f1(y_true: &[i64], y_pred: &[i64]) -> (f64, f64, f64) {
    let mut tp = 0;
    let mut fp = 0;
    let mut fn_ = 0;

    for (&true_label, &pred_label) in y_true.iter().zip(y_pred.iter()) {
        if pred_label == 1 {
            if true_label == 1 {
                tp += 1;
            } else {
                fp += 1;
            }
        } else if true_label == 1 {
            fn_ += 1;
        }
    }

    let precision = tp as f64 / (tp + fp).max(1) as f64;
    let recall = tp as f64 / (tp + fn_).max(1) as f64;
    let f1 = 2.0 * precision * recall / (precision + recall).max(1e-9);

    (precision, recall, f1)
}

pub fn print_confusion_matrix(matrix: &HashMap<(i64, i64), usize>) {
    println!("\nConfusion Matrix:");
    println!("{:>12} {:>12} {:>8}", "Actual", "Predicted", "Count");
    for ((actual, predicted), count) in matrix.iter() {
        println!("{:>12} {:>12} {:>8}", actual, predicted, count);
    }
}
