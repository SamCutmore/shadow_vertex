use crate::linalg::Matrix;
use std::ops::{Add, Mul,};

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