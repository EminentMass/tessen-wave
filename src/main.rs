mod fft;
mod swapspace;
use std::time::Instant;
use num_complex::Complex;

use swapspace::{Swap, FFT2D};
use fft::fft_to;

fn main() {
    let mut buf_a: Box<[Complex<f32>]> = (1..=4).map(|i| Complex::new(i as f32, 0.0)).collect();
    let mut buf_b = buf_a.clone();

    let input_const = buf_a.clone();
    let mut input = buf_a.clone();
    let mut output = buf_a.clone();

    fft_to(&mut input, &mut output);
    
    println!("Swap method");
    println!("Buffer A: {:?}", buf_a);
    println!("Buffer B: {:?}", buf_b);
    println!("{:}", (0..190).map(|_| '_').collect::<String>());
    let mut swap = Swap::from_ab(&mut buf_a[..], &mut buf_b[..]);
    swap.fft2d();
    println!("Buffer A: {:?}", buf_a);
    println!("Buffer B: {:?}", buf_b);

    println!("");
    println!("Non swap method");
    println!("Input: {:?}", input_const);
    println!("Output: {:?}", output);
}

// this function times the overhead of the for loop. This is probably negligable in most cases
fn _time<F>(operation: F, cycles: usize) -> f32
    where F: Fn() -> () {

        let start = Instant::now();
        for _ in 0..cycles {
            operation()
        }
        start.elapsed().as_secs_f32()/cycles as f32
}