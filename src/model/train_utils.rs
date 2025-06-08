use tch::nn::OptimizerConfig;
use tch::{Tensor, nn}; // Required for .build()

pub fn train(model: &impl nn::Module, vs: &nn::VarStore, xs: &Tensor, ys: &Tensor, epochs: i64) {
    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
    println!("Start fine-tuning");
    for epoch in 1..=epochs {
        let logits = model.forward(xs);
        let loss = logits.cross_entropy_for_logits(ys);

        opt.backward_step(&loss);

        let loss_value = loss.double_value(&[]); // Extract scalar
        println!("Epoch {:>3}/{:<3} | Loss: {:.6}", epoch, epochs, loss_value);
    }
}
