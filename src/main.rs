mod fft;

use nalgebra::Complex;
use fft::fft2d_inplace;

fn main() {
    let mut input: Vec<_> = (1..=4).map(|i| Complex::new(i as f64, 0.0)).collect();
    let mut buffer = input.clone();

    fft2d_inplace(&mut input, Some(&mut buffer), 2);

    println!("{:?}\n{:?}", input, buffer);
}