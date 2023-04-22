use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};


#[derive(Subcommand, Debug)]
pub enum Task {
    /// Train the neural network
    Train(TrainCommand),

    /// Test the neural network on your image
    Test(TestCommand),

    /// Create a new network
    Create,
}

#[derive(Args, Debug)]
pub struct TestCommand {
    /// Path to the image
    pub path: PathBuf,

    ///Path of the model to use - file type should be json
    pub model_path: PathBuf
}

#[derive(Debug, Args)]
pub struct TrainCommand {
    /// Path to the training data
    pub training_data_path: PathBuf,

    /// Path to the test data
    pub test_data_path: PathBuf,

    /// Path and name of the trained model file - file will be saved as json
    pub save_path:PathBuf,

    /// Number of iterations
    pub iterations: usize,
}

#[derive(Parser, Debug)]
#[clap(author, version, about)]
pub struct Cli {
    #[clap(subcommand)]
    pub task: Task,
}