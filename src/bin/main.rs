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

    let (x_train_data, y_train_data) = load_data_as_dataframe("datasets/training_set.csv");


    let x_train_data_array = array_from_dataframe(&x_train_data);
    let y_train_data_array = array_from_dataframe(&y_train_data);

    let (al, caches) = network.l_model_forward(
        x_train_data_array,
        parameters,
    );
    let cost = network.cost(&al, &y_train_data_array);

    let grads = network.l_model_backward(&al, &y_train_data_array, caches);



    println!("{:?}", grads);
}
