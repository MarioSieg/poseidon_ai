use activations::SIGMOID;
use neoronet::Network;

pub mod activations;
pub mod tensor;
pub mod neoronet;

fn main() {
	let inputs = vec![
		vec![0.0, 0.0],
		vec![0.0, 1.0],
		vec![1.0, 0.0],
		vec![1.0, 1.0],
	];
	let targets = vec![
		vec![0.0],
		vec![0.0],
		vec![0.0],
		vec![1.0]
	];

	let mut network = Network::new(vec![2, 3, 1], 0.1, SIGMOID);

	println!("{:?}", network.feed_forward(vec![0.0, 0.0]));
	println!("{:?}", network.feed_forward(vec![0.0, 1.0]));
	println!("{:?}", network.feed_forward(vec![1.0, 0.0]));
	println!("{:?}", network.feed_forward(vec![1.0, 1.0]));

	network.train(inputs, targets, 100);

	println!("{:?}", network.feed_forward(vec![0.0, 0.0]));
	println!("{:?}", network.feed_forward(vec![0.0, 1.0]));
	println!("{:?}", network.feed_forward(vec![1.0, 0.0]));
	println!("{:?}", network.feed_forward(vec![1.0, 1.0]));
}
