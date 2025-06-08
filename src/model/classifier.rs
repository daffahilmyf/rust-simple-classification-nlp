use tch::{nn, nn::Module};

pub fn build_model(vs: &nn::Path, input_dim: i64, hidden: i64, output: i64) -> impl Module {
    nn::seq()
        .add(nn::linear(
            vs / "fc1",
            input_dim,
            hidden,
            Default::default(),
        ))
        .add_fn(|x| x.relu())
        .add(nn::linear(vs / "fc2", hidden, output, Default::default()))
}
