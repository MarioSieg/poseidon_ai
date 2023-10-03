// (c) Copyright Mario "Neo" Sieg 2023. All rights reserved. mario.sieg.64@gmail.com
// BLAS - Basic Linear Algebra Subprograms
// TODO: Optimize for SIMD

pub fn reorg_cpu(
    x: &mut [f32],
    w: usize,
    h: usize,
    c: usize,
    batch: usize,
    stride: usize,
    forward: bool,
    out: &mut [f32],
) {
    let out_c = c / (stride * stride);

    for b in 0..batch {
        for k in 0..c {
            for j in 0..h {
                for i in 0..w {
                    let in_index = i + w * (j + h * (k + c * b));
                    let c2 = k % out_c;
                    let offset = k / out_c;
                    let w2 = i * stride + offset % stride;
                    let h2 = j * stride + offset / stride;
                    let out_index = w2 + w * stride * (h2 + h * stride * (c2 + out_c * b));

                    if forward {
                        out[out_index] = x[in_index];
                    } else {
                        out[in_index] = x[out_index];
                    }
                }
            }
        }
    }
}

pub fn flatten(x: &mut [f32], size: usize, layers: usize, batch: usize, forward: bool) {
    let mut swap = vec![0.0; size * layers * batch];
    for b in 0..batch {
        for c in 0..layers {
            for i in 0..size {
                let i1 = b * layers * size + c * size + i;
                let i2 = b * layers * size + i * layers + c;

                if forward {
                    swap[i2] = x[i1];
                } else {
                    swap[i1] = x[i2];
                }
            }
        }
    }
    x.copy_from_slice(&swap);
}

pub fn weighted_sum_cpu(a: &[f32], b: Option<&[f32]>, s: &[f32], c: &mut [f32]) {
    for i in 0..c.len() {
        c[i] = s[i] * a[i] + (1.0 - s[i]) * (b.map_or(0.0, |bb| bb[i]));
    }
}

pub fn weighted_delta_cpu(
    a: &[f32],
    b: Option<&[f32]>,
    s: &[f32],
    mut da: Option<&mut [f32]>,
    mut db: Option<&mut [f32]>,
    ds: &mut [f32],
    dc: &[f32],
) {
    for i in 0..dc.len() {
        if let Some(ref mut da) = da {
            da[i] += dc[i] * s[i];
        }
        if let Some(ref mut db) = db {
            db[i] += dc[i] * (1.0 - s[i]);
        }
        ds[i] += dc[i] * (a[i] - b.map_or(0.0, |bb| bb[i]));
    }
}

pub fn shortcut_cpu(
    batch: usize,
    w1: usize,
    h1: usize,
    c1: usize,
    add: &[f32],
    w2: usize,
    h2: usize,
    c2: usize,
    s1: f32,
    s2: f32,
    out: &mut [f32],
) {
    let stride = w1 / w2;
    let sample = w2 / w1;

    assert_eq!(stride, h1 / h2);
    assert_eq!(sample, h2 / h1);

    let stride = if stride < 1 { 1 } else { stride };
    let sample = if sample < 1 { 1 } else { sample };

    let minw = std::cmp::min(w1, w2);
    let minh = std::cmp::min(h1, h2);
    let minc = std::cmp::min(c1, c2);

    for b in 0..batch {
        for k in 0..minc {
            for j in 0..minh {
                for i in 0..minw {
                    let out_index = i * sample + w2 * (j * sample + h2 * (k + c2 * b));
                    let add_index = i * stride + w1 * (j * stride + h1 * (k + c1 * b));

                    out[out_index] = s1 * out[out_index] + s2 * add[add_index];
                }
            }
        }
    }
}

pub fn mean_cpu(x: &[f32], batch: usize, filters: usize, spatial: usize, mean: &mut [f32]) {
    let scale = 1.0 / (batch * spatial) as f32;

    for i in 0..filters {
        mean[i] = 0.0;
        for j in 0..batch {
            for k in 0..spatial {
                let index = j * filters * spatial + i * spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

pub fn variance_cpu(
    x: &[f32],
    mean: &[f32],
    batch: usize,
    filters: usize,
    spatial: usize,
    variance: &mut [f32],
) {
    let scale = 1.0 / ((batch * spatial) as f32 - 1.0);

    for i in 0..filters {
        variance[i] = 0.0;
        for j in 0..batch {
            for k in 0..spatial {
                let index = j * filters * spatial + i * spatial + k;
                variance[i] += (x[index] - mean[i]).powi(2);
            }
        }
        variance[i] *= scale;
    }
}

pub fn l2normalize_cpu(
    x: &mut [f32],
    dx: &mut [f32],
    batch: usize,
    filters: usize,
    spatial: usize,
) {
    for b in 0..batch {
        for i in 0..spatial {
            let mut sum = 0.0;
            for f in 0..filters {
                let index = b * filters * spatial + f * spatial + i;
                sum += x[index].powi(2);
            }
            sum = sum.sqrt();
            for f in 0..filters {
                let index = b * filters * spatial + f * spatial + i;
                x[index] /= sum;
                dx[index] = (1.0 - x[index]) / sum;
            }
        }
    }
}
