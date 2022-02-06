#![allow(dead_code)]

use std::ops::{ RangeTo, Range, RangeFrom };
use num_traits::{Float, FloatConst, Num};
use num_complex::Complex;

#[derive(Debug)]
pub struct Swap<T> {
    pub a: T,
    pub b: T,
}

pub trait Swappable<T> {
    fn swap(&mut self);
    fn into_ab(self) -> (T, T);
}

impl<T> Swappable<T> for Swap<T> {
    fn swap(&mut self) {
        std::mem::swap(&mut self.a, &mut self.b);
    }
    fn into_ab(self) -> (T, T) {
        (self.a, self.b)
    }
}

impl<T> Swap<T> {
    pub fn from_ab<'a>(a: T, b: T) -> Swap<T> {
        Swap {
            a,
            b,
        }
    }
    
    pub fn swap_binding(self) -> Self {
        Self {
            a: self.b,
            b: self.a,
        }
    }
}

// TODO: replace with generics
impl<T> Swap<&mut [T]> {
    fn index(& mut self, range: Range<usize>) -> Swap<&mut [T]> {
        Swap {
            a: &mut self.a[range.clone()],
            b: &mut self.b[range],
        }
    }
    fn index_from(&mut self, range: RangeFrom<usize>) -> Swap<&mut [T]>{
        Swap {
            a: &mut self.a[range.clone()],
            b: &mut self.b[range],
        }
    }
    fn index_to(&mut self, range: RangeTo<usize>) -> Swap<&mut [T]> {
        Swap {
            a: &mut self.a[range],
            b: &mut self.b[range],
        }
    }
}
impl<T: ?Sized> Swap<&mut T> {
    fn swapped(&mut self) -> Swap<&mut T> {
        Swap {
            a: &mut self.b,
            b: &mut self.a,
        }
    }
}

pub trait FFT {
    fn fft(&mut self);
    fn fft_inner(&mut self, n: usize, step: usize);
}

impl<T: Num + Float + FloatConst> FFT for Swap<& mut [Complex<T>]> {
    fn fft(& mut self) {
        let l = self.a.len();
        assert!(l.is_power_of_two());
        // swap between reading from one buffer and writing to the other on each iteration
        
        // ensure both buffers are the same
        self.b.clone_from_slice(self.a);
        self.fft_inner(l, 1);
    }

    fn fft_inner(
        & mut self,
        n: usize,
        step: usize,
    ) {
        if step >= n {
            return;
        }

        // each iteration swaps the buffers.
        self.swapped().fft_inner(n, step * 2);
        self.swapped().index_from(step..).fft_inner(n, step * 2);

        let (left, right) = self.a.split_at_mut(n/2);

        for i in (0..n).step_by(step*2){

            let iovern = T::from(i).unwrap() / T::from(n).unwrap(); // this should never fail unwrapping as the memory for any array outside the bounds of the float size would be ridiculous.

            let t = (-Complex::i()*T::PI() * iovern).exp() * self.b[i+step];
            left[i / 2] = self.b[i] + t;
            right[i / 2] = self.b[i] - t;
        }
    }
}

pub trait FFT2D {
    fn transpose(&mut self);
    fn row_fft(&mut self);
    fn fft2d(&mut self) {
        self.row_fft();
        self.transpose();
        self.row_fft();
        self.transpose();
    }
}

impl<T: Num + Float + FloatConst> FFT2D for Swap<&mut [Complex<T>]> {
    fn transpose(&mut self) {

        let l = (self.a.len() as f64).sqrt() as usize;

        let ind = |x, y| y*l+x;
        for i in 0..l {
            for j in 0..l {
                self.b[ind(j, i)] = self.a[ind(i, j)];
            }
        }
        self.swap();
    }
    fn row_fft(&mut self) {
        let l = (self.a.len() as f64).sqrt() as usize;

        for i in 0..l {
            self.index(i*l..(i+1)*l).fft();
        }
    }
}

/*
const fn integer_sqrt(y: usize) -> usize {

    // stolen from wikipedia
    let mut l = 0;
    let mut m;
    let mut r = y + 1;

    while l != r - 1 {
        m = (l + r) / 2;
        if m*m <= y {
            l = m;
        } else {
            r = m
        }
    }
    l
}
*/

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn transpose() {
        let mut buffer_a: Box<[Complex<f64>]> = (0..4).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut buffer_b = buffer_a.clone();
        let mut swappable = Swap::from_ab(&mut buffer_a[..], &mut buffer_b[..]);

        swappable.transpose();

        let buffer_a_ref = swappable.a;

        let answer_key: Box<[Complex<f64>]> = [     0, 2,
                                                    1, 3].iter().map(|i| Complex::new(*i as f64, 0.0)).collect();

        for (a, b) in buffer_a_ref.iter().zip(answer_key.iter()) {
            assert_eq!(a, b, "output: {:?}", buffer_a_ref);
        }
    }

    #[test]
    fn test_swap() {
        let mut swap = Swap::from_ab(10, 20);

        swap.swap();

        assert_eq!(swap.into_ab(), (20, 10));

        let x = 10;
        let y = 10;

        let mut swap = Swap::from_ab(&x, &y);

        swap.swap();

        assert_eq!(swap.into_ab(), (&y, &x));
    }
}