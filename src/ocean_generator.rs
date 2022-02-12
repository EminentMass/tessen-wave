use nalgebra::Vector2;
use nalgebra::{RealField, ComplexField};
use num_traits::{FloatConst};

pub fn h0<T, Y>(k: Vector2<T>, random: Y) -> Y
where
    T: FloatConst + RealField + Copy,
    Y: ComplexField<RealField = T>,
    f64: Into<T>
{
    random.scale(T::one()/T::SQRT_2() * ph(k).sqrt())
}

pub fn ph<T>(big_k: Vector2<T>) -> T
where
    T: FloatConst + RealField + Copy,
    f64: Into<T>,
{
    let g = 9.8.into(); // gravity
    let v = 40.0.into(); // wind speed
    let l = v*v/g;
    let a = 4.0.into(); // numerical constant
    let w: Vector2<T> = Vector2::new(1.0.into(), 1.0.into()).normalize(); // wind direction

    // Phillips spectrum change this if waves arent right
    let k = big_k.dot(&big_k);

    let numer = (-1.0.into()/(k*l.powi(2))).exp();
    let denom = k.powi(2);

     numer/denom * a * (big_k.dot(&w)).abs().powi(2)
}