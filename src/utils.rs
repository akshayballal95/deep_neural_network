use ndarray::prelude::*;
use std::f32::consts::E;

use crate::dnn::ActivationCache;

/// returns matrix with sigmoid activation applied to all values
///  # Arguments
///  * `Z` - A matrix   
pub fn sigmoid(z: Array2<f32>) -> (Array2<f32>,ActivationCache){
    (z.map(|x| 1.0/(1.0+E.powf(*x))), ActivationCache{z})
}

pub fn relu(){
    //TODO
}

