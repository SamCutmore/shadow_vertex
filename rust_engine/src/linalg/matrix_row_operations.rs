use crate::linalg::{Row, RowMut};
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};

macro_rules! impl_row_binary_ops {
    ($trait:ident, $method:ident) => {
        impl<'a, 'b, T> $trait<&'b Row<T>> for &'a Row<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Row<T>;
            fn $method(self, rhs: &'b Row<T>) -> Self::Output {
                assert_eq!(self.data.len(), rhs.data.len(), "Dimension mismatch");
                Row {
                    data: self
                        .data
                        .iter()
                        .zip(&rhs.data)
                        .map(|(&a, &b)| a.$method(b))
                        .collect(),
                }
            }
        }
        impl<T> $trait<Row<T>> for Row<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Row<T>;
            fn $method(self, rhs: Row<T>) -> Self::Output {
                (&self).$method(&rhs)
            }
        }
        impl<'a, T> $trait<&'a Row<T>> for Row<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Row<T>;
            fn $method(self, rhs: &'a Row<T>) -> Self::Output {
                (&self).$method(rhs)
            }
        }
        impl<'a, T> $trait<Row<T>> for &'a Row<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Row<T>;
            fn $method(self, rhs: Row<T>) -> Self::Output {
                self.$method(&rhs)
            }
        }

        // Scalar: Row op T
        impl<T> $trait<T> for Row<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Row<T>;
            fn $method(self, rhs: T) -> Self::Output {
                Row {
                    data: self.data.into_iter().map(|a| a.$method(rhs)).collect(),
                }
            }
        }
        impl<'a, T> $trait<T> for &'a Row<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = Row<T>;
            fn $method(self, rhs: T) -> Self::Output {
                Row {
                    data: self.data.iter().map(|&a| a.$method(rhs)).collect(),
                }
            }
        }
    };
}

macro_rules! impl_row_assign_ops {
    ($assign_trait:ident, $assign_method:ident) => {
        impl<T> $assign_trait<Row<T>> for Row<T>
        where
            T: $assign_trait,
        {
            fn $assign_method(&mut self, rhs: Row<T>) {
                assert_eq!(self.data.len(), rhs.data.len());
                for (a, b) in self.data.iter_mut().zip(rhs.data) {
                    a.$assign_method(b);
                }
            }
        }
        impl<'a, T> $assign_trait<&'a Row<T>> for Row<T>
        where
            T: Copy + $assign_trait,
        {
            fn $assign_method(&mut self, rhs: &'a Row<T>) {
                assert_eq!(self.data.len(), rhs.data.len());
                for (a, b) in self.data.iter_mut().zip(&rhs.data) {
                    a.$assign_method(*b);
                }
            }
        }
        impl<'a, T> $assign_trait<Row<T>> for RowMut<'a, T>
        where
            T: $assign_trait,
        {
            fn $assign_method(&mut self, rhs: Row<T>) {
                assert_eq!(self.data.len(), rhs.data.len());
                for (a, b) in self.iter_mut().zip(rhs.data) {
                    a.$assign_method(b);
                }
            }
        }
        impl<'a, 'b, T> $assign_trait<&'b Row<T>> for RowMut<'a, T>
        where
            T: Copy + $assign_trait,
        {
            fn $assign_method(&mut self, rhs: &'b Row<T>) {
                assert_eq!(self.data.len(), rhs.data.len());
                for (a, b) in self.iter_mut().zip(&rhs.data) {
                    a.$assign_method(*b);
                }
            }
        }
        impl<T> $assign_trait<T> for Row<T>
        where
            T: Copy + $assign_trait,
        {
            fn $assign_method(&mut self, rhs: T) {
                for a in self.data.iter_mut() {
                    a.$assign_method(rhs);
                }
            }
        }
        impl<'a, T> $assign_trait<T> for RowMut<'a, T>
        where
            T: Copy + $assign_trait,
        {
            fn $assign_method(&mut self, rhs: T) {
                for a in self.iter_mut() {
                    a.$assign_method(rhs);
                }
            }
        }
    };
}

impl_row_binary_ops!(Add, add);
impl_row_assign_ops!(AddAssign, add_assign);

impl_row_binary_ops!(Sub, sub);
impl_row_assign_ops!(SubAssign, sub_assign);

impl_row_binary_ops!(Mul, mul);
impl_row_assign_ops!(MulAssign, mul_assign);

impl_row_binary_ops!(Div, div);
impl_row_assign_ops!(DivAssign, div_assign);

impl<T> Row<T> {
    /// Performs `self -= rhs * scalar` in place without allocating a temporary row.
    pub fn sub_assign_scaled(&mut self, rhs: &Row<T>, scalar: T)
    where
        T: Copy + SubAssign + Mul<Output = T>,
    {
        for (a, &b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a -= b * scalar;
        }
    }
}

impl<'a, T> RowMut<'a, T> {
    /// Performs `self -= rhs * scalar` in place without allocating a temporary row.
    pub fn sub_assign_scaled(&mut self, rhs: &Row<T>, scalar: T)
    where
        T: Copy + SubAssign + Mul<Output = T>,
    {
        for (a, &b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a -= b * scalar;
        }
    }
}