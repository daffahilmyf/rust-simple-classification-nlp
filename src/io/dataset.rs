use anyhow::Result;
use std::fs;
use std::io::Read;
use std::path::Path;

#[derive(Debug)]
pub struct Review {
    pub text: String,
    pub label: i64,
}

pub fn load_dataset(dir: &str) -> Result<Vec<Review>> {
    let mut dataset = Vec::new();
    for (folder, label) in [("pos", 1), ("neg", 0)] {
        let path = Path::new(dir).join(folder);
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let mut file = fs::File::open(entry.path())?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)?;
            dataset.push(Review {
                text: contents,
                label,
            });
        }
    }
    Ok(dataset)
}
