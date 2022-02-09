use num_traits::{Float, FloatConst, Num};
use nalgebra::Complex;

/*
If input is not a power of two this will create a buffer to perform the fft.
On the other hand a bad buffer will cause an assert
*/
pub fn fft_inplace<T>(input: &mut [Complex<T>], maybe_buf: Option<&mut [Complex<T>]>)
where T: Num + Float + FloatConst
{
    let n = input.len();
    let npt = input.len().next_power_of_two();

    // may append nothing if input is a power of two
    let pad = |x: &mut Vec<_>| {
        x.append(&mut vec![Complex { re: T::zero(), im: T::zero() }; npt - n])
    };

    // this is only initialied if maybe_buf is None
    let mut owned_buf_b;

    let buffer_b = if let Some(b) = maybe_buf {
        // We could just create a new large enough buffer but it is probably better to alert the programmer when they are trying to use a buffer that is too small.
        debug_assert!(b.len() >= npt);
        b
    } else {
        // initialize owned_buf_b with the contents of input and return a reference for later use in fft_inner
        owned_buf_b = input.to_vec();
        pad(&mut owned_buf_b);
        &mut owned_buf_b
    };

    // if the buffer is not the correct shape we create a new one and copy the values over
    if input.len().is_power_of_two() {

        buffer_b.copy_from_slice(input);
        fft_inner(input, buffer_b, npt, 1);    
    }else {
        let mut buffer_a = input.to_vec();
        pad(&mut buffer_a) ;

        fft_inner(&mut buffer_a, buffer_b, npt, 1);

        // have to copy over or the output will be lost
        input.copy_from_slice(&buffer_a);
    };        
}

fn fft_inner<T>(
    buf_a: &mut [Complex<T>],
    buf_b: &mut [Complex<T>],
    n: usize,
    step: usize,
) where
    T: Num + Float + FloatConst,
{
    if step >= n {
        return;
    }

    // each iteration swaps the buffers.
    fft_inner(buf_b, buf_a, n, step * 2);
    fft_inner(&mut buf_b[step..], &mut buf_a[step..], n, step * 2);

    let (left, right) = buf_a.split_at_mut(n/2);

    for i in (0..n).step_by(step*2){

        let iovern = T::from(i).unwrap() / T::from(n).unwrap();

        let t = (-Complex::i()*T::PI() * iovern).exp() * buf_b[i+step];
        left[i / 2] = buf_b[i] + t;
        right[i / 2] = buf_b[i] - t;
    }
}

/*
This method only works on square power of two arrays and will assert otherwise
*/
pub fn fft2d_inplace<T>(input: &mut [Complex<T>], maybe_buf: Option<&mut [Complex<T>]>, l: usize)
where
    T: Num + Float + FloatConst
{
    debug_assert_eq!(l*l, input.len());
    debug_assert!(input.len().is_power_of_two());

    let mut owned_buf_b;

    let buffer_b = if let Some(b) = maybe_buf {
        // We could just create a new large enough buffer but it is probably better to alert the programmer when they are trying to use a buffer that is too small.
        assert!(b.len() >= l*l);
        b
    } else {
        // initialize owned_buf_b with the contents of input and return a reference for later use in fft_inner
        owned_buf_b = input.to_vec();
        &mut owned_buf_b
    };

    column_fft(input, buffer_b, l);
    transpose_to(input, buffer_b, l);
    column_fft(buffer_b, input, l);
    transpose_to(buffer_b, input, l);
}

fn transpose_to<T: Copy>(input: &[T], out: &mut [T], l: usize){
    let ind = |x, y| y*l+x;
    for i in 0..l {
        for j in 0..l {
            out[ind(j, i)] = input[ind(i, j)];
        }
    }
}

fn column_fft<T>(input: &mut [Complex<T>], buf: &mut [Complex<T>], l: usize)
where
    T: Num + Float + FloatConst
{
    for i in 0..l {
        // one column
        let r = i * l .. (i + 1) * l;
        fft_inplace(&mut input[r.clone()], Some(&mut buf[r]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose() {
        let input = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16];
        let mut out = [0; 16];

        transpose_to(&input, &mut out, 4);

        assert_eq!(out, [
            1, 5, 9,  13,
            2, 6, 10, 14,
            3, 7, 11, 15,
            4, 8, 12, 16,
        ]);
    }
}