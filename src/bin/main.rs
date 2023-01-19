use deep_neural_network::{dnn::DeepNeuralNetwork, utils::*};
use plotters::prelude::*;
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
            (0..data.len()).map(|x| (x, data[x])),
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
    let layer_dims: Vec<usize> = [12288, 20, 7, 5, 1].to_vec();
    let learning_rate: f32 = 0.0075;
    let network = DeepNeuralNetwork {
        layer_dims,
        learning_rate,
    };
    let mut parameters = network.initialize_parameters();

    let (x_train_data, y_train_data) = load_data_as_dataframe("datasets/training_set.csv");

    let x_train_data_array = array_from_dataframe(&x_train_data) / 255.0;
    let y_train_data_array = array_from_dataframe(&y_train_data) / 255.0;

    let mut costs: Vec<f32> = vec![];

    let iterations:usize = 500;

    for i in 0..iterations{
        let (al, caches) = network.l_model_forward(
            &x_train_data_array,
            &parameters,
        );

        let cost = network.cost(&al, &y_train_data_array);

        let grads = network.l_model_backward(&al, &y_train_data_array, caches);

        parameters = network.update_parameters(parameters, grads.clone(), learning_rate);

        costs.append(&mut vec![cost]);
        println!("Epoch : {}/{}    Cost: {:?}",i,iterations,cost);

    }

    plot(costs,iterations);
}
