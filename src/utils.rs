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

/// returns matrix with sigmoid activation applied to all values
///  # Arguments
///  * `Z` - A matrix   
pub fn sigmoid_activation(z: Array2<f32>) -> (Array2<f32>,ActivationCache){
    (z.map(|x| 1.0/(1.0+E.powf(*x))), ActivationCache{z})
}

pub fn relu_activation(z: Array2<f32>)->(Array2<f32>,ActivationCache){
    (z.map(|x| relu(*x)), ActivationCache{z})
}

pub fn load_data(path:&str) -> PolarsResult<DataFrame>{
    CsvReader::from_path(path).unwrap()
    .finish()
}