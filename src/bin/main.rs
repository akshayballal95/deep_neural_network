use std::{fs::OpenOptions, collections::HashMap};

use deep_neural_network::{dnn::DeepNeuralNetwork, utils::*};
use ndarray::{Array2, arr2};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};


#[derive(Serialize, Deserialize, Debug)]
struct Parameter{
    parameter:HashMap<String,Array2<f32>>
}

fn plot(data: Vec<f32>, iters:usize) {
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
            (0..iters).map(|x| (x, data[x/100])),
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
fn main() {
    let layer_dims: Vec<usize> = vec![12288, 20, 7, 5, 1];
    let learning_rate: f32 = 0.0075;
    let network = DeepNeuralNetwork {
        layer_dims,
        learning_rate,
    };
    let mut parameters = network.initialize_parameters();


    let (x_train_data, y_train_data) = load_data_as_dataframe("datasets/training_set.csv");
    let (x_test_data, y_test_data) = load_data_as_dataframe("datasets/test_set.csv");

    let x_train_data_array = array_from_dataframe(&x_train_data) / 255.0;
    let y_train_data_array = array_from_dataframe(&y_train_data) ;

    let x_test_data_array = array_from_dataframe(&x_test_data) / 255.0;
    let y_test_data_array = array_from_dataframe(&y_test_data);

    let mut costs: Vec<f32> = vec![];

    let iterations:usize = 100;
    let mut a:Array2<f32>=arr2(&[[1.0]]);

    for i in 0..iterations{
        let (al, caches) = network.l_model_forward(
            &x_train_data_array,
            &parameters,
        );
        a = al.clone();

        let cost = network.cost(&al, &y_train_data_array);

        let grads = network.l_model_backward(&al, &y_train_data_array, caches);

        parameters = network.update_parameters(parameters, grads.clone(), learning_rate);
        
        if i%100 == 0 {
            costs.append(&mut vec![cost]);
            // println!("{:?}",parameters);
            println!("Epoch : {}/{}    Cost: {:?}",i,iterations,cost);
        }

    }
    // let score = network.predict(x_train_data_array, &y_train_data_array, parameters);

    // println!("{}", score);

    plot(costs,iterations);
    
    // let al:Array2<f32> = Array2::from_shape_vec((1,209),al_vec).unwrap();
    // let al= arr2(&[[0.010973942, 0.0017315926, 0.9779758, 0.00000026187286, 0.00042358856, 0.0037234777, 0.0011249179, 0.9934006, 0.0008115289, 0.00038743153, 0.0033236113, 0.9732529, 0.01482084, 0.9993285, 0.9823529, 0.000793152, 0.0000629548, 0.0003709066, 0.0014002852, 0.99141777, 0.0000084077155, 0.0011399348, 0.014082213, 0.0006661747, 0.9858393, 0.96293896, 0.00054200424, 0.99688184, 0.0017873253, 0.9798813, 0.007142868, 0.005622543, 0.014669548, 0.016661143, 0.011208701, 0.00039638067, 0.0000065054733, 0.00023023837, 0.8652213, 0.04762893, 0.00005105911, 0.93860286, 0.99282116, 0.000041587748, 0.0006859212, 0.007980004, 0.0023027877, 0.9849481, 0.000014685713, 0.00007581595, 0.93368804, 0.0116329845, 0.006788344, 0.055740215, 0.9904795, 0.0000013606275, 0.99498427, 0.9848043, 0.0013720866, 0.99171466, 0.9910868, 0.98184055, 0.0001844688, 0.00031253812, 0.00045305214, 0.0016289268, 0.0011675964, 0.008195481, 0.97210115, 0.000025262092, 0.0002722995, 0.9974814, 0.0028713513, 0.017241398, 0.0009634107, 0.2921238, 0.00000014325673, 0.00015650921, 0.006085703, 0.015071292, 0.000191692, 0.0000022472677, 0.00024176265, 0.9865746, 0.95571244, 0.00026821272, 0.0064556045, 0.0006163293, 0.94155645, 0.00054004876, 0.0258761, 0.0041508107, 0.9541271, 0.97578627, 0.9990895, 0.008926002, 0.010196396, 0.99411774, 0.00022125522, 0.015571987, 0.0059190216, 0.0001749506, 0.98544484, 0.0029791447, 0.994356, 0.0016140232, 0.99954766, 0.9801695, 0.97101766, 0.9826241, 0.99951303, 0.99304366, 0.00035667003, 0.0015959209, 0.0006386773, 0.0002313483, 0.000007781805, 0.9969399, 0.0000049752894, 0.008330417, 0.000020250289, 0.95925105, 0.0002848793, 0.0000000030873628, 0.99197525, 0.00000015715437, 0.9889403, 0.008842014, 0.99805886, 0.97742474, 0.027131625, 0.006728265, 0.0005783095, 0.9958776, 0.9762365, 0.97407573, 0.9875167, 0.99959415, 0.0000030207755, 0.0003639524, 0.00011448123, 0.00006217638, 0.99824, 0.00008340924, 0.95443714, 0.95979214, 0.9961194, 0.006425333, 0.95331156, 0.99518603, 0.06985586, 0.0042456277, 0.000033623983, 0.99597687, 0.004107099, 0.03856281, 0.99831104, 0.000012532127, 0.00016517728, 0.0006252023, 0.000007366179, 0.0004217241, 0.98540556, 0.001666708, 0.9876425, 0.000025215206, 0.99960893, 0.000100849255, 0.00023592456, 0.97648925, 0.9883016, 0.9964625, 0.02432428, 0.000015774442, 0.9988128, 0.9822181, 0.00012298665, 0.9702674, 0.0059909816, 0.98720944, 0.0023063598, 0.00043856128, 0.00021673889, 0.00040473076, 0.00000021008269, 0.981294, 0.00005330909, 0.0077242013, 0.9885147, 0.00005665182, 0.00011777921, 0.000806975, 0.98894995, 0.0000023263578, 0.0000039092115, 0.02669507, 0.06409359, 0.9955754, 0.0036534506, 0.00004263914, 0.9885123, 0.000019075731, 0.0000014465334, 0.000054946893, 0.010487361, 0.02034023, 0.0015609512, 0.032677643, 0.018382879]]);

    let score = network.predict(x_train_data_array,a, &y_train_data_array );

    let file = OpenOptions::new()
    .create(true)
    .write(true)
    .truncate(true)
    .open("weights.json").unwrap();

    let write = serde_json::to_writer(file, &parameters);

    println!("{}",score)

}
