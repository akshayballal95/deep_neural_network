use deep_neural_network::{dnn::{Parameters, DeepNeuralNetwork}, utils::*};

fn main() {
    let layer_dims: Vec<usize> = vec![12288, 20, 7, 5, 1];
    let learning_rate: f32 = 0.0075;
    let lambda: f32 = 0.05;
    let network = DeepNeuralNetwork {
        layer_dims,
        learning_rate,
        lambda,
    };
   
    let (x_train_data, y_train_data) = load_data_as_dataframe("datasets/training_set.csv");
    let (x_test_data, y_test_data) = load_data_as_dataframe("datasets/test_set.csv");

    let x_train_data_array = array_from_dataframe(&x_train_data) / 255.0;
    let y_train_data_array = array_from_dataframe(&y_train_data);

    let x_test_data_array = array_from_dataframe(&x_test_data) / 255.0;
    let y_test_data_array = array_from_dataframe(&y_test_data);

    let parameters = Parameters::new(network.initialize_parameters());

    let iterations: usize = 1500;

    // let parameters = network.train_model(&x_train_data_array, &y_train_data_array, parameters, iterations, learning_rate);

    // write_parameters_to_json_file(&parameters.parameters, "weights.json");

    let parameters = Parameters::new(load_weights_from_json()); 

    
    let y_hat = network.predict(&x_train_data_array, &parameters);
    println!("Training Set Accuracy: {}%", network.score(&y_hat,&y_train_data_array ));
    

    let y_hat = network.predict(&x_test_data_array, &parameters);
    println!("Test Set Accuracy: {}%", network.score(&y_hat,&y_test_data_array));


    let img_array = load_image("cat.jpeg");
    let prediction = network.predict(&img_array, &parameters);
    println!("For First: {}", prediction.sum());


    
    let img_array = load_image("cat2.jpg");
    let prediction = network.predict(&img_array, &parameters);
    println!("For Second: {}", prediction.sum());


    let img_array = load_image("dog.jpg");
    let prediction = network.predict(&img_array, &parameters);
    println!("For Third: {}", prediction.sum());

}