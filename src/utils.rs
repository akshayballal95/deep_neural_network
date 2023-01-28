use crate::dnn::ActivationCache;
use ndarray::prelude::*;
use polars::{prelude::*, export::num::ToPrimitive};
use std::{collections::HashMap, f32::consts::E, fs::OpenOptions, path::PathBuf};
use plotters::prelude::*;
use image::{self, imageops::FilterType::Gaussian};

fn relu(z: f32) -> f32 {
    if z > 0.0 {
        z
    } else {
        0.0
    }
}

pub fn relu_prime(z: f32) -> f32 {
    if z > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + E.powf(-z))
}

fn sigmoid_prime(z: f32) -> f32 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

/// returns matrix with sigmoid activation applied to all values
///  # Arguments
///  * `Z` - A matrix   
pub fn sigmoid_activation(z: Array2<f32>) -> (Array2<f32>, ActivationCache) {
    (z.map(|x| sigmoid(*x)), ActivationCache { z })
}

pub fn sigmoid_backward(da: &Array2<f32>, z: Array2<f32>) -> Array2<f32> {
    da * z.map(|x| sigmoid_prime(*x))
}

pub fn relu_backward(da: &Array2<f32>, z: Array2<f32>) -> Array2<f32> {
    da * z.map(|x| relu_prime(*x))
}

pub fn relu_activation(z: Array2<f32>) -> (Array2<f32>, ActivationCache) {
    (z.map(|x| relu(*x)), ActivationCache { z })
}

/**
Loads data from a .csv file to a Polars DataFrame
*/
pub fn load_data_as_dataframe(path: PathBuf) -> (DataFrame, DataFrame) {
    let data = CsvReader::from_path(path).unwrap().finish().unwrap();

    let x_train_data = data.drop("y").unwrap();
    let y_train_data = data.select(["y"]).unwrap();

    (x_train_data, y_train_data)
}

/**
Converts DataFrame to ndarray - Array2<f32>
*/
pub fn array_from_dataframe(dataframe: &DataFrame) -> Array2<f32> {
    dataframe
        .to_ndarray::<Float32Type>()
        .unwrap()
        .reversed_axes()
}

pub fn write_parameters_to_json_file(parameters: &HashMap<String, Array2<f32>>, file_path:&str){
    let file = OpenOptions::new()
    .create(true)
    .write(true)
    .truncate(true)
    .open(file_path)
    .unwrap();

    _ = serde_json::to_writer(file, parameters);

}

pub fn load_weights_from_json(path:PathBuf) ->HashMap<String, Array2<f32>>  {
    let text = std::fs::read_to_string(path).unwrap();
    let weights_json: serde_json::Value = serde_json::from_str(&text).unwrap();

    let mut parameters: HashMap<String, Array2<f32>> = HashMap::new();

    for (key, val) in weights_json.as_object().unwrap() {
        let dims = val["dim"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap().to_usize().unwrap())
            .collect::<Vec<usize>>();
        let data = val["data"].as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap().to_f32().unwrap())
        .collect::<Vec<f32>>();

        let matrix = Array2::from_shape_vec((dims[0],dims[1]), data).unwrap();
        parameters.insert(key.to_string(), matrix);

    }
    parameters
}



pub fn plot(data: Vec<f32>, iters: usize) {
    let root = BitMapBackend::new("0.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("COST CURVE", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0usize..iters, 0.0f32..1f32)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(LineSeries::new(
            (0..iters).step_by(100).map(|x| (x, data[x / 100])),
            &RED,
        ))
        .unwrap()
        .label("COST")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
}

pub fn load_image(path: PathBuf) -> Array2<f32>{

    let img = image::open(path).unwrap();
    let img_r = img.resize_exact(64, 64, Gaussian);
    let img_array:Array2<f32> = Array2::from_shape_vec((12288,1), img_r.to_rgb32f().as_raw().to_vec()).unwrap();

    img_array
}


