use ndarray::prelude::*;
use polars::prelude::*;
use std::f32::consts::E;
use crate::dnn::ActivationCache;

fn relu(z:f32)->f32{
    if z>0.0{
        z
    }
    else {
        0.0
    }
}

fn relu_prime(z:f32)->f32{
    if z>0.0{
        1.0
    }
    else {
        0.0
    }
}

fn sigmoid(z:f32)->f32{
    1.0/(1.0+E.powf(z))
}

fn sigmoid_prime(z:f32)->f32{
    sigmoid(z)*(1.0-sigmoid(z))
}

/// returns matrix with sigmoid activation applied to all values
///  # Arguments
///  * `Z` - A matrix   
pub fn sigmoid_activation(z: Array2<f32>) -> (Array2<f32>,ActivationCache){
    (z.map(|x| sigmoid(*x)), ActivationCache{z})
}

pub fn sigmoid_backward(z: Array2<f32>) -> Array2<f32>{
   z.map(|x| sigmoid_prime(*x))
}

pub fn relu_backward(z: Array2<f32>) -> Array2<f32>{
    z.map(|x| relu_prime(*x))
 }

pub fn relu_activation(z: Array2<f32>)->(Array2<f32>,ActivationCache){
    (z.map(|x| relu(*x)), ActivationCache{z})
}

/**
Loads data from a .csv file to a Polars DataFrame
*/
pub fn load_data_as_dataframe(path:&str) -> (DataFrame,DataFrame){
    let data = CsvReader::from_path(path).unwrap()
    .finish().unwrap();

    let x_train_data = data.drop("y").unwrap();
    let y_train_data = data.select(["y"]).unwrap();

    (x_train_data, y_train_data)
}

/**
Converts DataFrame to ndarray - Array2<f32>
*/
pub fn array_from_dataframe(dataframe:&DataFrame)->Array2<f32>{
    dataframe.to_ndarray::<Float32Type>().unwrap().reversed_axes()
}