use crate::matrix_adt::Matrix;
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Shl, ShlAssign, Shr, ShrAssign};

// Scalar addition
impl<T> Add<T> for &Matrix<T>
where T: Clone + Add<Output = T>
{
    type Output = Matrix<T>;
    fn add(self, scalar: T) -> Matrix<T> {
        let data = self.data.iter().map(|x| x.clone() + scalar.clone()).collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

impl<T> AddAssign<T> for Matrix<T>
where T: Clone + AddAssign
{
    fn add_assign(&mut self, scalar: T) {
        for val in &mut self.data { *val += scalar.clone(); }
    }
}

// Scalar subtraction
impl<T> Sub<T> for &Matrix<T>
where T: Clone + Sub<Output = T>
{
    type Output = Matrix<T>;
    fn sub(self, scalar: T) -> Matrix<T> {
        let data = self.data.iter().map(|x| x.clone() - scalar.clone()).collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

impl<T> SubAssign<T> for Matrix<T>
where T: Clone + SubAssign
{
    fn sub_assign(&mut self, scalar: T) {
        for val in &mut self.data { *val -= scalar.clone(); }
    }
}

// Scalar multiplication
impl<T> Mul<T> for &Matrix<T>
where T: Clone + Mul<Output = T>
{
    type Output = Matrix<T>;
    fn mul(self, scalar: T) -> Matrix<T> {
        let data = self.data.iter().map(|x| x.clone() * scalar.clone()).collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

impl<T> MulAssign<T> for Matrix<T>
where T: Clone + MulAssign
{
    fn mul_assign(&mut self, scalar: T) {
        for val in &mut self.data { *val *= scalar.clone(); }
    }
}

// Scalar division
impl<T> Div<T> for &Matrix<T>
where T: Clone + Div<Output = T>
{
    type Output = Matrix<T>;
    fn div(self, scalar: T) -> Matrix<T> {
        let data = self.data.iter().map(|x| x.clone() / scalar.clone()).collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

impl<T> DivAssign<T> for Matrix<T>
where T: Clone + DivAssign
{
    fn div_assign(&mut self, scalar: T) {
        for val in &mut self.data { *val /= scalar.clone(); }
    }
}

// Element-wise addition
impl<T> Add<&Matrix<T>> for &Matrix<T>
where T: Clone + Add<Output = T>
{
    type Output = Matrix<T>;
    fn add(self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data = self.data.iter()
            .zip(&other.data)
            .map(|(a,b)| a.clone() + b.clone())
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

impl<T> AddAssign<&Matrix<T>> for Matrix<T>
where T: Clone + AddAssign
{
    fn add_assign(&mut self, other: &Matrix<T>) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for (a,b) in self.data.iter_mut().zip(&other.data) {
            *a += b.clone();
        }
    }
}

// Element-wise subtraction
impl<T> Sub<&Matrix<T>> for &Matrix<T>
where T: Clone + Sub<Output = T>
{
    type Output = Matrix<T>;
    fn sub(self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data = self.data.iter()
            .zip(&other.data)
            .map(|(a,b)| a.clone() - b.clone())
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

impl<T> SubAssign<&Matrix<T>> for Matrix<T>
where T: Clone + SubAssign
{
    fn sub_assign(&mut self, other: &Matrix<T>) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for (a,b) in self.data.iter_mut().zip(&other.data) {
            *a -= b.clone();
        }
    }
}

// Element-wise multiplication (Hadamard)
impl<T> Mul<&Matrix<T>> for &Matrix<T>
where T: Clone + Mul<Output = T>
{
    type Output = Matrix<T>;
    fn mul(self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data = self.data.iter()
            .zip(&other.data)
            .map(|(a,b)| a.clone() * b.clone())
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

impl<T> MulAssign<&Matrix<T>> for Matrix<T>
where T: Clone + MulAssign
{
    fn mul_assign(&mut self, other: &Matrix<T>) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for (a,b) in self.data.iter_mut().zip(&other.data) {
            *a *= b.clone();
        }
    }
}

// Element-wise division
impl<T> Div<&Matrix<T>> for &Matrix<T>
where T: Clone + Div<Output = T>
{
    type Output = Matrix<T>;
    fn div(self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data = self.data.iter()
            .zip(&other.data)
            .map(|(a,b)| a.clone() / b.clone())
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

impl<T> DivAssign<&Matrix<T>> for Matrix<T>
where T: Clone + DivAssign
{
    fn div_assign(&mut self, other: &Matrix<T>) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for (a,b) in self.data.iter_mut().zip(&other.data) {
            *a /= b.clone();
        }
    }
}

// Bitwise operations
impl<T> BitAnd<&Matrix<T>> for &Matrix<T>
where T: Clone + BitAnd<Output = T>
{
    type Output = Matrix<T>;
    fn bitand(self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data = self.data.iter()
            .zip(&other.data)
            .map(|(a,b)| a.clone() & b.clone())
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

impl<T> BitAndAssign<&Matrix<T>> for Matrix<T>
where T: Clone + BitAndAssign
{
    fn bitand_assign(&mut self, other: &Matrix<T>) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for (a,b) in self.data.iter_mut().zip(&other.data) {
            *a &= b.clone();
        }
    }
}

impl<T> BitOr<&Matrix<T>> for &Matrix<T>
where T: Clone + BitOr<Output = T>
{
    type Output = Matrix<T>;
    fn bitor(self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data = self.data.iter()
            .zip(&other.data)
            .map(|(a,b)| a.clone() | b.clone())
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

impl<T> BitOrAssign<&Matrix<T>> for Matrix<T>
where T: Clone + BitOrAssign
{
    fn bitor_assign(&mut self, other: &Matrix<T>) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for (a,b) in self.data.iter_mut().zip(&other.data) {
            *a |= b.clone();
        }
    }
}

impl<T> BitXor<&Matrix<T>> for &Matrix<T>
where T: Clone + BitXor<Output = T>
{
    type Output = Matrix<T>;
    fn bitxor(self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data = self.data.iter()
            .zip(&other.data)
            .map(|(a,b)| a.clone() ^ b.clone())
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

impl<T> BitXorAssign<&Matrix<T>> for Matrix<T>
where T: Clone + BitXorAssign
{
    fn bitxor_assign(&mut self, other: &Matrix<T>) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for (a,b) in self.data.iter_mut().zip(&other.data) {
            *a ^= b.clone();
        }
    }
}

// Shift operations
impl<T> Shl<&Matrix<T>> for &Matrix<T>
where T: Clone + Shl<Output = T>
{
    type Output = Matrix<T>;
    fn shl(self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data = self.data.iter()
            .zip(&other.data)
            .map(|(a,b)| a.clone() << b.clone())
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

impl<T> ShlAssign<&Matrix<T>> for Matrix<T>
where T: Clone + ShlAssign
{
    fn shl_assign(&mut self, other: &Matrix<T>) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for (a,b) in self.data.iter_mut().zip(&other.data) {
            *a <<= b.clone();
        }
    }
}

impl<T> Shr<&Matrix<T>> for &Matrix<T>
where T: Clone + Shr<Output = T>
{
    type Output = Matrix<T>;
    fn shr(self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data = self.data.iter()
            .zip(&other.data)
            .map(|(a,b)| a.clone() >> b.clone())
            .collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }
}

impl<T> ShrAssign<&Matrix<T>> for Matrix<T>
where T: Clone + ShrAssign
{
    fn shr_assign(&mut self, other: &Matrix<T>) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for (a,b) in self.data.iter_mut().zip(&other.data) {
            *a >>= b.clone();
        }
    }
}


// Dot product
impl<T> Matrix<T>
where T: Clone + Default + Add<Output = T> + Mul<Output = T>
{
    pub fn dot(&self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.cols, other.rows);
        let mut result = Matrix::new(self.rows, other.cols);
        for r in 0..self.rows {
            for c in 0..other.cols {
                let mut sum = T::default();
                for k in 0..self.cols {
                    sum = sum + self[(r,k)].clone() * other[(k,c)].clone();
                }
                result[(r,c)] = sum;
            }
        }
        result
    }
}
