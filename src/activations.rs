// (c) Copyright Mario "Neo" Sieg 2023. All rights reserved. mario.sieg.64@gmail.com
// Activation functions and their gradients

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ActivationFunction {
    Logistic,
    Relu,
    Relie,
    Linear,
    Ramp,
    TanH,
    Plse,
    Leaky,
    Elu,
    Loggy,
    Stair,
    HardTan,
    LhTan,
    Selu,
}

impl std::fmt::Display for ActivationFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format!("{:?}", self).to_lowercase().fmt(f)
    }
}

pub fn activate(x: f32, f: ActivationFunction) -> f32 {
    use activation::*;
    match f {
        ActivationFunction::Logistic => logistic(x),
        ActivationFunction::Relu => relu(x),
        ActivationFunction::Relie => relie(x),
        ActivationFunction::Linear => linear(x),
        ActivationFunction::Ramp => ramp(x),
        ActivationFunction::TanH => tanh(x),
        ActivationFunction::Plse => plse(x),
        ActivationFunction::Leaky => leaky(x),
        ActivationFunction::Elu => elu(x),
        ActivationFunction::Loggy => loggy(x),
        ActivationFunction::Stair => stair(x),
        ActivationFunction::HardTan => hard_tan(x),
        ActivationFunction::LhTan => lh_tan(x),
        ActivationFunction::Selu => selu(x),
    }
}

pub fn activate_vector(v: &mut [f32], f: ActivationFunction) {
    for x in v {
        *x = activate(*x, f);
    }
}

pub fn gradient(x: f32, f: ActivationFunction) -> f32 {
    use gradient::*;
    match f {
        ActivationFunction::Logistic => logistic(x),
        ActivationFunction::Relu => relu(x),
        ActivationFunction::Relie => relie(x),
        ActivationFunction::Linear => linear(x),
        ActivationFunction::Ramp => ramp(x),
        ActivationFunction::TanH => tanh(x),
        ActivationFunction::Plse => plse(x),
        ActivationFunction::Leaky => leaky(x),
        ActivationFunction::Elu => elu(x),
        ActivationFunction::Loggy => loggy(x),
        ActivationFunction::Stair => stair(x),
        ActivationFunction::HardTan => hard_tan(x),
        ActivationFunction::LhTan => lh_tan(x),
        ActivationFunction::Selu => selu(x),
    }
}

pub fn gradient_vector(v: &mut [f32], f: ActivationFunction) {
    for x in v {
        *x = gradient(*x, f);
    }
}

mod activation {
    #[inline(always)]
    pub fn logistic(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[inline(always)]
    pub fn relu(x: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }

    #[inline(always)]
    pub fn relie(x: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            0.01 * x
        }
    }

    #[inline(always)]
    pub fn linear(x: f32) -> f32 {
        x
    }

    #[inline(always)]
    pub fn ramp(x: f32) -> f32 {
        if x > 0.0 {
            x + 0.1 * x
        } else {
            0.1 * x
        }
    }

    #[inline(always)]
    pub fn tanh(x: f32) -> f32 {
        let exp = (x * 2.0).exp();
        (exp - 1.0) / (exp + 1.0)
    }

    #[inline(always)]
    pub fn plse(x: f32) -> f32 {
        if x < -4.0 {
            0.01 * (x + 4.0)
        } else if x > 4.0 {
            0.01 * (x - 4.0) + 1.0
        } else {
            0.125 * x + 0.5
        }
    }

    #[inline(always)]
    pub fn leaky(x: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            0.1 * x
        }
    }

    #[inline(always)]
    pub fn elu(x: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            x.exp() - 1.0
        }
    }

    #[inline(always)]
    pub fn loggy(x: f32) -> f32 {
        2.0 / (1.0 + (-x).exp()) - 1.0
    }

    #[inline(always)]
    pub fn stair(x: f32) -> f32 {
        let n = x.floor();
        if n % 2.0 == 0.0 {
            n
        } else {
            n + 0.5
        }
    }

    #[inline(always)]
    pub fn hard_tan(x: f32) -> f32 {
        if x < -1.0 {
            -1.0
        } else if x > 1.0 {
            1.0
        } else {
            x
        }
    }

    #[inline(always)]
    pub fn lh_tan(x: f32) -> f32 {
        if x < 0.0 {
            0.001 * x
        } else if x > 1.0 {
            0.001 * (x - 1.0) + 1.0
        } else {
            x
        }
    }

    #[inline(always)]
    pub fn selu(x: f32) -> f32 {
        if x >= 0.0 {
            1.0507 * x
        } else {
            1.0507 * 1.6732 * (x.exp() - 1.0)
        }
    }
}

mod gradient {
    #[inline(always)]
    pub fn logistic(x: f32) -> f32 {
        (1.0 - x) * x
    }

    #[inline(always)]
    pub fn relu(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    #[inline(always)]
    pub fn relie(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.01
        }
    }

    #[inline(always)]
    pub fn linear(x: f32) -> f32 {
        1.0
    }

    #[inline(always)]
    pub fn ramp(x: f32) -> f32 {
        if x > 0.0 {
            1.1
        } else {
            0.1
        }
    }

    #[inline(always)]
    pub fn tanh(x: f32) -> f32 {
        1.0 - x * x
    }

    #[inline(always)]
    pub fn plse(x: f32) -> f32 {
        if x < 0.0 || x > 1.0 {
            0.01
        } else {
            0.125
        }
    }

    #[inline(always)]
    pub fn leaky(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.1
        }
    }

    #[inline(always)]
    pub fn elu(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            x.exp()
        }
    }

    #[inline(always)]
    pub fn loggy(x: f32) -> f32 {
        let y = (x + 1.0) / 2.0;
        2.0 * (1.0 - y) * y
    }

    #[inline(always)]
    pub fn stair(x: f32) -> f32 {
        if x.floor() == x {
            0.0
        } else {
            1.0
        }
    }

    #[inline(always)]
    pub fn hard_tan(x: f32) -> f32 {
        if x > -1.0 && x < 1.0 {
            1.0
        } else {
            0.0
        }
    }

    #[inline(always)]
    pub fn lh_tan(x: f32) -> f32 {
        if x > 0.0 && x < 1.0 {
            1.0
        } else {
            0.001
        }
    }

    #[inline(always)]
    pub fn selu(x: f32) -> f32 {
        if x >= 0.0 {
            1.0507
        } else {
            x + 1.0507 * 1.6732
        }
    }
}
