use ndarray::prelude::*;
use rand::Rng;
use std::{collections::HashMap, ops::Div};

use crate::utils::*;
#[derive(Clone, Debug)]
pub struct LinearCache {
    pub a: Array2<f32>,
    pub w: Array2<f32>,
    pub b: Array2<f32>,
}

#[derive(Debug, Clone)]
pub struct ActivationCache {
    pub z: Array2<f32>,
}

#[derive(Clone)]
pub struct DeepNeuralNetwork {
    pub layer_dims: Vec<usize>,
    pub learning_rate: f32,
}

fn linear_forward(a: &Array2<f32>, w: &Array2<f32>, b: &Array2<f32>) -> (Array2<f32>, LinearCache) {
    let z = w.dot(a) + b;

    let cache = LinearCache {
        a: a.clone(),
        w: w.clone(),
        b: b.clone(),
    };

    return (z, cache);
}

fn linear_backward(
    dz: &Array2<f32>,
    linear_cache: LinearCache,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let (a_prev, w, b) = (linear_cache.a, linear_cache.w, linear_cache.b);
    let m = a_prev.shape()[1] as f32;
    let dw = (1.0 / m) * (dz.dot(&a_prev.reversed_axes()));
    let db_vec = ((1.0/m)*dz.sum_axis(Axis(1))).to_vec();
    let db = Array2::from_shape_vec((db_vec.len(), 1), db_vec).unwrap();
    let da_prev = w.reversed_axes().dot(dz);

    return (da_prev, dw, db);
}

fn linear_backward_activation(
    da: &Array2<f32>,
    cache: (LinearCache, ActivationCache),
    activation: &str,
) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>), String> {
    let (linear_cache, activation_cache) = cache;
    if activation == "relu" {
        let dz = da * relu_backward(activation_cache.z);
        let (da_prev, dw, db) = linear_backward(&dz, linear_cache);
        Ok((da_prev, dw, db))
    } else if activation == "sigmoid" {
        let dz = da * sigmoid_backward(activation_cache.z);
        let (da_prev, dw, db) = linear_backward(&dz, linear_cache);
        Ok((da_prev, dw, db))
    } else {
        Err("wrong activation".to_string())
    }
}

trait Log {
    fn log(&self) -> Array2<f32>;
}

impl Log for Array2<f32> {
    fn log(&self) -> Array2<f32> {
        self.map(|x| x.log(std::f32::consts::E))
    }
}

fn linear_forward_activation(
    a_prev: &Array2<f32>,
    w: &Array2<f32>,
    b: &Array2<f32>,
    activation: &str,
) -> Option<(Array2<f32>, (LinearCache, ActivationCache))> {
    if activation == "sigmoid" {
        let (z, linear_cache) = linear_forward(a_prev, w, b);
        let (a, activation_cache) = sigmoid_activation(z);
        let cache = (linear_cache, activation_cache);
        Some((a, cache))
    } else if activation == "relu" {
        let (z, linear_cache) = linear_forward(a_prev, w, b);
        let (a, activation_cache) = relu_activation(z);
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
    ///     * hashmap containing your parameters "W1", "b1", ..., \
    ///     * "WL", "bL": Wl -- weight matrix of shape (layer_dims\[l], layer_dims\[l-1])\
    ///     * bl -- bias vector of shape (layer_dims\[l], 1)
    pub fn initialize_parameters(&self) -> HashMap<String, Array2<f32>> {
        let mut rng = rand::thread_rng(); // random number generator

        let number_of_layers = self.layer_dims.len(); // this includes input layer

        let mut parameters: HashMap<String, Array2<f32>> = HashMap::new();

        // Start from 1 because the 0th layer is input layer
        for l in 1..number_of_layers {
            // Create a list of (layer_dims[l] * self.layer_dims[l-1]) integers and
            // multiply each with a random float
            let w: Vec<f32> = (0..self.layer_dims[l] * self.layer_dims[l - 1])
                .map(|_| rng.gen_range(0.0..1.0) * 0.01)
                .collect();
            let w_matrix =
                Array::from_shape_vec((self.layer_dims[l], self.layer_dims[l - 1]), w).unwrap();

            let b: Vec<f32> = vec![0.0; self.layer_dims[l]];
            let b_vector = Array::from_shape_vec((self.layer_dims[l], 1), b).unwrap();

            let weight_string = ["W", &l.to_string()].join("").to_string();
            let biases_string = ["b", &l.to_string()].join("").to_string();

            parameters.insert(weight_string, w_matrix);
            parameters.insert(biases_string, b_vector);
        }

        return parameters;
    }

    ///  # Arguments
    ///  * `x` - input data matrix of shape (num_of_input_features, num_of_examples)
    ///  * `parameters` -
    ///     * hashmap containing your parameters "W1", "b1", ..., \
    ///     * "WL", "bL": Wl -- weight matrix of shape (layer_dims\[l], layer_dims\[l-1])\
    ///     * bl -- bias vector of shape (layer_dims\[l], 1)
    ///  # Returns
    ///  * tuple `(al, caches)`
    ///  * `al` - last activation layer of shape (num_of_hidden_units_in_last, num_of_examples)
    ///  * `caches` - Tuple of (LinearCache, ActivationCache) - LinearCache contains `a`, `w`, `b` and ActivationCache contains `z`
    ///
    pub fn l_model_forward(
        &self,
        x: &Array2<f32>,
        parameters: &HashMap<String, Array2<f32>>,
    ) -> (Array2<f32>, HashMap<String, (LinearCache, ActivationCache)>) {
        let number_of_layers = self.layer_dims.len() - 1;
        let mut a = x.clone();
        let mut caches = HashMap::new();

        for l in 1..number_of_layers {
            let a_prev = a.clone();
            let weight_string = ["W", &l.to_string()].join("").to_string();
            let bias_string = ["b", &l.to_string()].join("").to_string();

            let w = &parameters[&weight_string];
            let b = &parameters[&bias_string];

            let (a_temp, cache_temp ) = linear_forward_activation(&a_prev, w, b, "relu").unwrap();

            a = a_temp;
            let cache = cache_temp;

            caches.insert(l.to_string(), cache);
        }

        let weight_string = ["W", &(number_of_layers).to_string()].join("").to_string();
        let bias_string = ["b", &(number_of_layers).to_string()].join("").to_string();

        let w = &parameters[&weight_string];
        let b = &parameters[&bias_string];

        let (al, cache) = linear_forward_activation(&a, w, b, "sigmoid").unwrap();

        caches.insert(number_of_layers.to_string(), cache);
        (al, caches)
    }

    pub fn cost(&self, al: &Array2<f32>, y: &Array2<f32>) -> f32 {
        let m = y.shape()[1] as f32;
        let cost = -(1.0 / m)
            * (y.dot(&al.clone().reversed_axes().log())
                + (1.0 - y).dot(&(1.0 - al).reversed_axes().log()));

        return cost.sum();
    }

    pub fn l_model_backward(
        &self,
        al: &Array2<f32>,
        y: &Array2<f32>,
        caches: HashMap<String, (LinearCache, ActivationCache)>,
    ) -> HashMap<String, Array2<f32>> {
        let mut grads = HashMap::new();
        let num_of_layers = self.layer_dims.len() - 1;

        let dal = -(y/al - (1.0 - y)/(1.0 - al));

        println!("{}",al);

        let current_cache = caches[&num_of_layers.to_string()].clone();
        let (mut da_prev, mut dw, mut db) =
            linear_backward_activation(&dal, current_cache, "sigmoid").unwrap();

        let weight_string = ["dW", &num_of_layers.to_string()].join("").to_string();
        let bias_string = ["db", &num_of_layers.to_string()].join("").to_string();
        let activation_string = ["dA", &num_of_layers.to_string()].join("").to_string();

        grads.insert(weight_string, dw);
        grads.insert(bias_string, db);
        grads.insert(activation_string, da_prev.clone());

        for l in (1..num_of_layers).rev() {
            let current_cache = caches[&l.to_string()].clone();
            (da_prev, dw, db) =
                linear_backward_activation(&da_prev, current_cache, "relu").unwrap();

            let weight_string = ["dW", &l.to_string()].join("").to_string();
            let bias_string = ["db", &l.to_string()].join("").to_string();
            let activation_string = ["dA", &l.to_string()].join("").to_string();

            grads.insert(weight_string, dw);
            grads.insert(bias_string, db);
            grads.insert(activation_string, da_prev.clone());
        }

        grads
    }

    pub fn update_parameters(
        &self,
        mut parameters: HashMap<String, Array2<f32>>,
        grads: HashMap<String, Array2<f32>>,
        learning_rate: f32,
    ) -> HashMap<String, Array2<f32>>{
        let num_of_layers = self.layer_dims.len() - 1;
        for l in 1..num_of_layers + 1 {
            let weight_string_grad = ["dW", &l.to_string()].join("").to_string();
            let bias_string_grad = ["db", &l.to_string()].join("").to_string();
            let weight_string = ["W", &l.to_string()].join("").to_string();
            let bias_string = ["b", &l.to_string()].join("").to_string();

            *parameters.get_mut(&weight_string).unwrap() = parameters[&weight_string].clone()
                - learning_rate * grads[&weight_string_grad].clone();
            *parameters.get_mut(&bias_string).unwrap() =
                parameters[&bias_string].clone() - learning_rate * grads[&bias_string_grad].clone();
        }
        parameters
    }

    pub fn predict(&self,x_test_data:Array2<f32>, y_test_data:&Array2<f32> , parameters: HashMap<String, Array2<f32>>)->f32{
        let (al,_) = self.l_model_forward(&x_test_data, &parameters);
        

        let y_hat = al.map(|x| x*(x>&0.5) as i32 as f32);
        // println!("{:?}", al);

        (y_hat-y_test_data).map(|x| x.abs()).sum()/y_test_data.len() as f32
        
       


    }
    
}
