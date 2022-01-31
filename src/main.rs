use std::time::Instant;
use num_complex::Complex;
use std::f32::consts::PI;

const I: Complex<f32> = Complex { re: 0.0, im: 1.0 };

fn slow_fft(coefficients: &[Complex<f32>]) -> Vec<Complex<f32>> {
    // maybe refuse too large coefficients or non powers of two

    let length = coefficients.len();

    let mut out = Vec::with_capacity(length);

    for j in 0..length {
        out.push({
            let mut x = Complex::new(0f32,0f32);
            for i in 0..length {
                x += coefficients[i] * ((2.0*I*PI)/(length as f32)).exp().powi(i as i32 * j as i32);
            }
            x
        });
    }
    out
}

fn fast_fft(inputs: &[Complex<f32>]) -> Vec<Complex<f32>>{
    let l = inputs.len();
    fn fft_inner(
        buf_a: &mut [Complex<f32>],
        buf_b: &mut [Complex<f32>],
        n: usize,
        step: usize,
    ) {
        if step >= n {
            return;
        }

        fft_inner(buf_b, buf_a, n, step * 2);
        fft_inner(&mut buf_b[step..], &mut buf_a[step..], n, step * 2);
        let (left, right) = buf_a.split_at_mut(n/2);

        for i in (0..n).step_by(step*2){
            let t = (-I*PI * (i as f32) / (n as f32)).exp() * buf_b[i+step];
            left[i / 2] = buf_b[i] + t;
            right[i / 2] = buf_b[i] - t;
        }
    }

    // make buffer a power of two so operation is possible
    let n_orig = inputs.len();
    let n = n_orig.next_power_of_two();

    let mut buf_a = inputs.to_vec();

    buf_a.append(&mut vec![Complex {re: 0.0, im: 0.0}; n - n_orig]);

    // swap between reading from one buffer and writing to the other
    let mut buf_b = buf_a.clone();
    fft_inner(&mut buf_a, &mut buf_b, n, 1);
    buf_a
}

fn main() {
    /*
    let inputs: Vec<Complex<f32>>= [1,2,1,2].iter().map(|i| Complex::new(*i as f32, 0f32)).collect();

    let outs = slow_fft(&inputs);

    println!("{:?}", outs);
    */
    let oper = || {
        let inputs: Vec<Complex<f32>> = (0..4096).map(|i| Complex::new(i as f32, 0f32)).collect();
        //slow_fft(&inputs);
        fast_fft(&inputs);
    };

    println!("{:?}", time(oper, 30));
}

// this function times the overhead of the for loop. This is probably negligable in most cases
fn time<F>(operation: F, cycles: usize) -> f32
    where F: Fn() -> () {

        let start = Instant::now();
        for i in 0..cycles {
            operation()
        }
        start.elapsed().as_secs_f32()/cycles as f32
}