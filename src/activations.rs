use std::f32::consts::E;

#[derive(Clone)]
pub struct Activation<'f> {
	pub function: &'f dyn Fn(f32) -> f32,
	pub derivative: &'f dyn Fn(f32) -> f32,
}

pub const IDENTITY: Activation = Activation {
	function: &|x| x,
	derivative: &|_| 1.0,
};

pub const SIGMOID: Activation = Activation {
	function: &|x| 1.0 / (1.0 + E.powf(-x)),
	derivative: &|x| x * (1.0 - x),
};

pub const TANH: Activation = Activation {
	function: &|x| x.tanh(),
	derivative: &|x| 1.0 - (x.powi(2)),
};

pub const RELU: Activation = Activation {
	function: &|x| x.max(0.0),
	derivative: &|x| if x > 0.0 { 1.0 } else { 0.0 },
};
