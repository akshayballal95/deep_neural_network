use deep_neural_network::{dnn::DeepNeuralNetwork, utils::*};

use polars::{prelude::Float32Type,};
fn main() {
    let layer_dims: Vec<usize> = [12288, 20, 7, 5, 1].to_vec();
    let learning_rate: f32 = 0.2;
    let network = DeepNeuralNetwork {
        layer_dims,
        learning_rate,
    };
    let parameters = network.initialize_parameters();

    let data = load_data("datasets/training_set.csv").unwrap();
    let x_train_data = data.drop("y").unwrap();
    let y_train_data = data.select(["y"]).unwrap();

    let x_train_data_array = x_train_data
        .to_ndarray::<Float32Type>()
        .unwrap()
        .reversed_axes();

    let y_train_data_array = y_train_data
        .to_ndarray::<Float32Type>()
        .unwrap()
        .reversed_axes();

    let (al, caches) = network.l_model_forward(
        x_train_data_array,
        parameters,
    );
    let cost = network.cost(&al, &y_train_data_array);

    println!("{:?}", cost);
}
