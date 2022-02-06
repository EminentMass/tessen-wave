use num_complex::Complex;
use std::f32::consts::PI as PI32;

const I: Complex<f32> = Complex { re: 0.0, im: 1.0 };

// destroys data in inputs
pub fn fft_to(inputs: &mut [Complex<f32>], outputs: &mut [Complex<f32>]) {
    assert!(inputs.len() == outputs.len());
    assert!(inputs.len() == inputs.len().next_power_of_two());
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
            let t = (-I*PI32 * (i as f32) / (n as f32)).exp() * buf_b[i+step];
            left[i / 2] = buf_b[i] + t;
            right[i / 2] = buf_b[i] - t;
        }
    }

    // swap between reading from one buffer and writing to the other
    outputs.clone_from_slice(inputs);
    fft_inner(outputs, inputs, l, 1);
}
