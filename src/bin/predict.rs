use simple_nlp_rust::{
    model::classifier::build_model, postprocessing::interpret::predict_label, preprocessing::vocab,
};
use std::{env, fs::File, io::Read};
use tch::{Tensor, nn::Module};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: predict \"Your input sentence here\"");
        std::process::exit(1);
    }

    let input_text = &args[1];

    let mut file = File::open("output/vocab.json")?;
    let mut json = String::new();
    file.read_to_string(&mut json)?;
    let vocab_vec: Vec<String> = serde_json::from_str(&json)?;

    let input_tensor =
        Tensor::f_from_slice(&vocab::vectorize(input_text, &vocab_vec))?.unsqueeze(0);

    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    vs.load("output/model.ot")?;
    let root = vs.root();
    let model = build_model(&root, vocab_vec.len() as i64, 64, 2);

    let output = model.forward(&input_tensor);
    let label = predict_label(&output);

    println!("Prediction: {}", label);
    Ok(())
}
