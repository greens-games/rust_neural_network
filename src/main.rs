use std::collections::HashMap;
use ndarray::{Array,  Dim, linalg};
use std::cmp;
use polars;

fn main() {
    let mut test_data: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 2]>> = Array::<f32, _>::zeros((1,5));
    generate_random(&mut test_data);
    let mut network: HashMap<String, ndarray::ArrayBase<ndarray::OwnedRepr<f32>, Dim<[usize; 2]>>> = init_hidden_layers();
    test_data.iter_mut().for_each(|n| {
        print!("{}",n);
    })
}

//Generates random numbers between -0.5 and 0.5 for weights and biases
//Can expand this later to take in value range to generate numbers between
fn generate_random(a: &mut Array::<f32,Dim<[usize; 2]>>) {
    for i in a {
        *i = rand::random::<f32>()-0.5;
    }
}

//create a hash map defining our network?
//each entry can be a layer with the key being the name of the layer
//the value being a 2d array of params
fn init_hidden_layers() -> HashMap<String, Array::<f32,Dim<[usize; 2]>>> {
    let mut network: HashMap<String, Array::<f32,Dim<[usize; 2]>>> = HashMap::new();

    network.insert(String::from("weights_1"), Array::<f32, _>::zeros((5,10)));
    network.insert(String::from("baises_1"), Array::<f32, _>::zeros((1,10)));
    network.insert(String::from("weights_2"), Array::<f32, _>::zeros((10,10)));
    network.insert(String::from("biases_2"), Array::<f32, _>::zeros((1,10)));

    network.iter_mut().for_each(|(k,v)| {
        generate_random(v);
    });

    network
}

fn relu() {
    //for each element in our data set, get the max of 0 and entry
}


fn foward_prop(network: &mut HashMap<String, Array::<f32,Dim<[usize; 2]>>>, ) {

}