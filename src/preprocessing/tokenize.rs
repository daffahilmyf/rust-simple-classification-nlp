pub fn clean_text(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect()
}

pub fn tokenize(text: &str) -> Vec<String> {
    clean_text(text)
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}
