use deep_neural_network::{
    dnn::{DeepNeuralNetwork},
    utils::*,
};
use polars::prelude::Float32Type;
fn main() {
    let layer_dims: Vec<usize> = [12288, 3, 4, 1].to_vec();
    let learning_rate: f32 = 0.2;
    let network = DeepNeuralNetwork {
        layer_dims: layer_dims,
        learning_rate,
    };
    let parameters = network.initialize_parameters();

    let data = load_data("datasets/training_set.csv").unwrap();
    let  x_train_data = data.drop("y").unwrap();
    let _y_train_data = data.column("y").unwrap();

    let feed_forward = network.l_model_forward(x_train_data.to_ndarray::<Float32Type>().unwrap().reversed_axes(),parameters);
    println!("{:?}", feed_forward.1["1"])
}
