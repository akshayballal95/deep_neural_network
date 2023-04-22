use std::{
    fs::{File, OpenOptions},
    io::{self, BufRead, BufReader, Write},
};

use clap::Parser;

use deep_neural_network::{
    args::{Cli, Task},
    dnn::{DeepNeuralNetwork, Parameters},
    utils::*,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let file = File::open("network.json").unwrap();
    let reader = BufReader::new(file);
    let network: DeepNeuralNetwork = serde_json::from_reader(reader).unwrap();

    let args = Cli::parse();

    match args.task {
        Task::Train(train_command) => {
            let (x_train_data, y_train_data) =
                load_data_as_dataframe(&train_command.training_data_path)?;
            let (x_test_data, y_test_data) = load_data_as_dataframe(&train_command.test_data_path)?;

            let x_train_data_array = array_from_dataframe(&x_train_data) / 255.0;
            let y_train_data_array = array_from_dataframe(&y_train_data);

            let x_test_data_array = array_from_dataframe(&x_test_data) / 255.0;
            let y_test_data_array = array_from_dataframe(&y_test_data);

            let parameters = Parameters::new(network.initialize_parameters());

            let iterations: usize = train_command.iterations;

            let parameters = network.train_model(
                &x_train_data_array,
                &y_train_data_array,
                parameters,
                iterations,
                network.learning_rate,
            );

            write_parameters_to_json_file(&parameters.parameters, train_command.save_path);

            let y_hat = network.predict(&x_train_data_array, &parameters);
            println!(
                "Training Set Accuracy: {}%",
                network.score(&y_hat, &y_train_data_array)
            );

            let y_hat = network.predict(&x_test_data_array, &parameters);
            println!(
                "Test Set Accuracy: {}%",
                network.score(&y_hat, &y_test_data_array)
            );

            Ok(())
        }

        Task::Test(testcommand) => {
            let weights = load_weights_from_json(&testcommand.model_path)?;
            let parameters = Parameters::new(weights);

            let img_array = load_image(&testcommand.path.into())?;
            let prediction = network.predict(&img_array, &parameters);
            println!("Prediction {}", prediction.sum());

            Ok(())
        }

        Task::Create => {
            let mut layer_dims_string = String::new();
            print!("Enter Layer Dimensions : ");
            io::stdout().flush().unwrap();
            std::io::stdin()
                .lock()
                .read_line(&mut layer_dims_string)
                .unwrap();
            let layer_dims: Vec<usize> = layer_dims_string
                .trim()
                .split(",")
                .map(|x| x.parse().unwrap())
                .collect();

            let mut learning_rate_string = String::new();
            print!("Enter Learning Rate : ");
            io::stdout().flush().unwrap();
            std::io::stdin()
                .lock()
                .read_line(&mut learning_rate_string)
                .unwrap();
            let learning_rate: f32 = learning_rate_string.trim().parse().unwrap();

            let mut lambda_string = String::new();
            print!("Enter Lambda : ");
            io::stdout().flush().unwrap();
            std::io::stdin()
                .lock()
                .read_line(&mut lambda_string)
                .unwrap();
            let lambda: f32 = lambda_string.trim().parse().unwrap();

            let net = DeepNeuralNetwork {
                layer_dims,
                learning_rate,
                lambda,
            };
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open("network.json")
                .unwrap();

            _ = serde_json::to_writer(file, &net);

            Ok(())

        }
    }
}
