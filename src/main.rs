mod fft;
mod ocean_generator;

use nalgebra::{Vector2, Complex};
use rand::prelude::*;
use rand_distr::StandardNormal;
use fft::fft2d_inplace;

use bmp::{Image, Pixel, px};
use ocean_generator::h0;

fn main() {
    /*
    let mut rng = rand::thread_rng();
    let y: f64 = rng.sample(StandardNormal);

    let mut input: Vec<_> = (1..=4).map(|i| Complex::new(i as f64, 0.0)).collect();
    let mut buffer = input.clone();

    fft2d_inplace(&mut input, Some(&mut buffer), 2);

    println!("{:?}\n{:?}", input, buffer);
    println!("{:}", y);
    */

    let mut rng = thread_rng();

    let mut h0rand = |k| {
        let rand: Complex<f64> = Complex::new(rng.sample(StandardNormal), rng.sample(StandardNormal));

        h0(k, rand)
    };

    let mut fourier: Vec<Complex<f64>> = (0..(256*256)).map(|_| h0rand(Vector2::new(0.0, 1.0))).collect();

    match fourier.as_slice() {
        [a, b, c, d, e, ..] => println!("{:} {:} {:} {:} {:}", a, b, c, d, e),
        [..] => return
    }
    let mut buffer = fourier.clone();

    fft2d_inplace(&mut fourier, Some(&mut buffer), 256);

    let mut img = Image::new(256, 256);

    for (x, y) in img.coordinates() {
        let com = fourier[(x + y*256) as usize];

        let r = com.re;
        let g = com.im;
        let b = 0.0;
        // scale values here

        img.set_pixel(x, y, px!(r, g, b))
    }

    let _ = img.save("h0.bmp");
}