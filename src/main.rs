fn convolution_2d(input: &[&[f32]], kernel: &[&[f32]], padding: f32) -> Vec<f32> {
    let mut r = Vec::new();
    let input_rows: usize = input.len();
    let input_cols: usize = input[0].len();
    let kernel_rows: usize = kernel.len();
    let kernel_cols: usize = kernel[0].len();
    let output_rows: usize = input_rows - kernel_rows + 1;
    let output_cols: usize = input_cols - kernel_cols + 1;
    for i in 0..output_rows {
        for j in 0..output_cols {
            let mut sum = 0.0;
            for m in 0..kernel_rows {
                for n in 0..kernel_cols {
                    sum += input[i + m][j + n] * kernel[m][n];
                }
            }
            r.push(sum + padding);
        }
    }
    r
}

fn main() {
    let input: &[&[f32]] = &[
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        &[6.0, 7.0, 8.0, 9.0, 10.0],
        &[11.0, 12.0, 13.0, 14.0, 15.0],
        &[16.0, 17.0, 18.0, 19.0, 20.0],
        &[21.0, 22.0, 23.0, 24.0, 25.0],
    ];
    let kernel: &[&[f32]] = &[
        &[1.0, 1.0, 1.0],
        &[1.0, 1.0, 1.0],
        &[1.0, 1.0, 1.0]
    ];
    let r = convolution_2d(input, &kernel, 0.0);
    println!("{:?}", r);
}
