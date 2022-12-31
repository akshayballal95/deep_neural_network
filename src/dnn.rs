use ndarray::prelude::*;
use rand::Rng;
use std::collections::HashMap;

use crate::utils::*;
#[derive(Clone)]
pub struct LinearCache {
    pub a: Array2<f32>,
    pub w: Array2<f32>,
    pub b: Array1<f32>,
}

pub struct ActivationCache {
    pub z: Array2<f32>,
}

#[derive(Clone)]
pub struct DeepNeuralNetwork {
    pub layer_dims: Vec<usize>,
    pub learning_rate: f32,
}
#[derive(Debug)]
pub enum Matrix {
    Weight(Array2<f32>),
    Bias(Array1<f32>),
}

impl From<Array2<f32>> for Matrix {
    fn from(matrix: Array2<f32>) -> Self {
        Matrix::Weight(matrix)
    }
}

impl From<Array1<f32>> for Matrix {
    fn from(vector: Array1<f32>) -> Self {
        Matrix::Bias(vector)
    }
}

fn linear_forward(a: &Array2<f32>, w: &Array2<f32>, b: &Array1<f32>) -> (Array2<f32>, LinearCache) {
    let z = w.dot(a) + b;

    let cache = LinearCache {
        a: a.clone(),
        w: w.clone(),
        b: b.clone(),
    };

    return (z, cache);
}

fn linear_forward_activation(
    a_prev: &Array2<f32>,
    w: &Array2<f32>,
    b: &Array1<f32>,
    activation: &str,
) -> Option<(Array2<f32>, (LinearCache, ActivationCache))> {
    if activation == "sigmoid" {
        let (z, linear_cache) = linear_forward(a_prev, w, b);
        let (a, activation_cache) = sigmoid(z);
        let cache = (linear_cache, activation_cache);
        Some((a, cache))
    } else if activation == "relu" {
        let (z, linear_cache) = linear_forward(a_prev, w, b);
        let (a, activation_cache) = sigmoid(z);
        let cache = (linear_cache, activation_cache);
        Some((a, cache))
    } else {
        None
    }
}

impl DeepNeuralNetwork {
    ///  # Arguments
    ///  * `layer_dims` - array (list) containing the dimensions of each layer in our network
    ///  # Returns
    /// * `parameters` -
    ///     * hasshmap containing your parameters "W1", "b1", ..., \
    ///     * "WL", "bL": Wl -- weight matrix of shape (layer_dims\[l], layer_dims\[l-1])\
    ///     * bl -- bias vector of shape (layer_dims\[l], 1)
    pub fn initialize_parameters(self) -> HashMap<String, Matrix> {
        let mut rng = rand::thread_rng();

        let number_of_layers = self.layer_dims.len();

        let mut parameters: HashMap<String, Matrix> = HashMap::new();

        for l in 1..number_of_layers {
            let w: Vec<f32> = (0..self.layer_dims[1] * self.layer_dims[0])
                .map(|_| rng.gen_range(0.0..1.0) * 0.01)
                .collect();
            let w_matrix =
                Array::from_shape_vec((self.layer_dims[1], self.layer_dims[0]), w).unwrap();

            let b: Vec<f32> = vec![0.0; self.layer_dims[l]];
            let b_vector = Array::from_vec(b);

            let weight_string = ["W", &l.to_string()].join("").to_string();
            let biases_string = ["b", &l.to_string()].join("").to_string();

            parameters.insert(weight_string, Matrix::from(w_matrix));
            parameters.insert(biases_string, Matrix::from(b_vector));
        }

        return parameters;
    }

    pub fn l_model_forward(
        self,
        x: Array2<f32>,
        parameters: HashMap<String, Matrix>,
    ) -> (Array2<f32>, HashMap<String, (LinearCache, ActivationCache)>) {
        let number_of_layers = self.layer_dims.len();
        let mut a = x;
        let mut caches = HashMap::new();

        for l in 1..3 {
            let a_prev = a.clone();
            let weight_string = ["W", &l.to_string()].join("").to_string();
            let bias_string = ["b", &l.to_string()].join("").to_string();

            let w = match &parameters[&weight_string] {
                Matrix::Weight(weights_matrix) => Some(weights_matrix),
                _ => None,
            };

            let b = match &parameters[&bias_string] {
                Matrix::Bias(bias_vector) => Some(bias_vector),
                _ => None,
            };

            a = linear_forward_activation(&a_prev, w.unwrap(), b.unwrap(), "relu")
                .unwrap()
                .0;
            let cache = linear_forward_activation(&a_prev, w.unwrap(), b.unwrap(), "relu")
                .unwrap()
                .1;

            caches.insert(l.to_string(), cache);
        }

        let weight_string = ["W", &number_of_layers.to_string()].join("").to_string();
        let bias_string = ["b", &number_of_layers.to_string()].join("").to_string();

        let w = match &parameters[&weight_string] {
            Matrix::Weight(weights_matrix) => Some(weights_matrix),
            _ => None,
        };

        let b = match &parameters[&bias_string] {
            Matrix::Bias(bias_vector) => Some(bias_vector),
            _ => None,
        };

        let (al, cache) = linear_forward_activation(&a, w.unwrap(), b.unwrap(), "sigmoid").unwrap();

        caches.insert(number_of_layers.to_string(), cache);
        (al, caches)
    }

    pub fn cost(self) {
        //TODO
    }

    pub fn backward_propogation(self) {
        //TODO
    }

    pub fn upgrade_gradients(self) {}
}
