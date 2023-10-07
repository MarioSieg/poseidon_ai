use std::{
	fs::File,
	io::{Read, Write},
};

use serde::{Deserialize, Serialize};
use serde_json::{from_str, json};

use super::{activations::Activation, tensor::Tensor2D};

pub struct Network<'f> {
	layers: Vec<usize>,
	weights: Vec<Tensor2D>,
	biases: Vec<Tensor2D>,
	data: Vec<Tensor2D>,
	learning_rate: f32,
	activation: Activation<'f>,
}

#[derive(Serialize, Deserialize)]
struct SaveData {
	weights: Vec<Vec<Vec<f32>>>,
	biases: Vec<Vec<Vec<f32>>>,
}

impl Network<'_> {
	pub fn new(
		layers: Vec<usize>,
		learning_rate: f32,
		activation: Activation,
	) -> Network {
		let mut weights = vec![];
		let mut biases = vec![];

		for i in 0..layers.len() - 1 {
			weights.push(Tensor2D::random(layers[i + 1], layers[i]));
			biases.push(Tensor2D::random(layers[i + 1], 1));
		}

		Network {
			layers,
			weights,
			biases,
			data: vec![],
			learning_rate,
			activation,
		}
	}

	pub fn feed_forward(&mut self, inputs: Vec<f32>) -> Vec<f32> {
		if inputs.len() != self.layers[0] {
			panic!("Invalid inputs length");
		}

		let mut current = Tensor2D::from(vec![inputs]).transpose();
		self.data = vec![current.clone()];

		for i in 0..self.layers.len() - 1 {
			current = self.weights[i]
				.multiply(&current)
				.add(&self.biases[i])
				.map(self.activation.function);
			self.data.push(current.clone());
		}

		current.transpose().data[0].to_owned()
	}

	pub fn back_propogate(&mut self, outputs: Vec<f32>, targets: Vec<f32>) {
		if targets.len() != self.layers[self.layers.len() - 1] {
			panic!("Invalid targets length");
		}

		let parsed = Tensor2D::from(vec![outputs]).transpose();
		let mut errors = Tensor2D::from(vec![targets]).transpose().subtract(&parsed);
		let mut gradients = parsed.map(self.activation.derivative);

		for i in (0..self.layers.len() - 1).rev() {
			gradients = gradients
				.dot_multiply(&errors)
				.map(&|x| x * self.learning_rate);

			self.weights[i] = self.weights[i].add(&gradients.multiply(&self.data[i].transpose()));
			self.biases[i] = self.biases[i].add(&gradients);

			errors = self.weights[i].transpose().multiply(&errors);
			gradients = self.data[i].map(self.activation.derivative);
		}
	}

	pub fn train(&mut self, inputs: Vec<Vec<f32>>, targets: Vec<Vec<f32>>, epochs: u64) {
		for i in 1..=epochs {
			if epochs < 100 || i % (epochs / 100) == 0 {
				println!("Epoch {} of {}", i, epochs);
			}
			for j in 0..inputs.len() {
				let outputs = self.feed_forward(inputs[j].clone());
				self.back_propogate(outputs, targets[j].clone());
			}
		}
	}

	pub fn save(&self, file: String) {
		let mut file = File::create(file).expect("Unable to touch save file");

		file.write_all(
			json!({
				"weights": self.weights.clone().into_iter().map(|matrix| matrix.data).collect::<Vec<Vec<Vec<f32>>>>(),
				"biases": self.biases.clone().into_iter().map(|matrix| matrix.data).collect::<Vec<Vec<Vec<f32>>>>()
			}).to_string().as_bytes(),
		).expect("Unable to write to save file");
	}

	pub fn load(&mut self, file: String) {
		let mut file = File::open(file).expect("Unable to open save file");
		let mut buffer = String::new();

		file.read_to_string(&mut buffer)
			.expect("Unable to read save file");

		let save_data: SaveData = from_str(&buffer).expect("Unable to serialize save data");

		let mut weights = vec![];
		let mut biases = vec![];

		for i in 0..self.layers.len() - 1 {
			weights.push(Tensor2D::from(save_data.weights[i].clone()));
			biases.push(Tensor2D::from(save_data.biases[i].clone()));
		}

		self.weights = weights;
		self.biases = biases;
	}
}
