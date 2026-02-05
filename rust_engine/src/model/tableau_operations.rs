use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};
use crate::model::{Tableau, TableauRow, TableauRowMut};
use num_traits::{One, Zero};

impl<'a, T> TableauRowMut<'a, T> {
    /// Performs `self -= rhs * scalar` in place without allocating a temporary TableauRow.
    pub fn sub_assign_scaled(&mut self, rhs: &TableauRow<T>, scalar: T)
    where
        T: Copy + SubAssign + Mul<Output = T>,
    {
        self.coefficients.sub_assign_scaled(&rhs.coefficients, scalar);
        self.slack.sub_assign_scaled(&rhs.slack, scalar);
        *self.rhs -= rhs.rhs * scalar;
    }
}

/// Pivot selection outcome: Optimal, Unbounded, or Pivot(row, col).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PivotResult {
    Optimal,
    Unbounded,
    Pivot(usize, usize),
}

#[inline]
fn assert_same_shape<T>(a: &TableauRow<T>, b: &TableauRow<T>) {
    debug_assert_eq!(
        a.coefficients.data.len() + a.slack.data.len(),
        b.coefficients.data.len() + b.slack.data.len()
    );
}

macro_rules! impl_tableau_row_binary_ops {
    ($trait:ident, $method:ident) => {
        impl<'a, 'b, T> $trait<&'b TableauRow<T>> for &'a TableauRow<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = TableauRow<T>;
            fn $method(self, rhs: &'b TableauRow<T>) -> TableauRow<T> {
                assert_same_shape(self, rhs);
                TableauRow {
                    coefficients: (&self.coefficients).$method(&rhs.coefficients),
                    slack: (&self.slack).$method(&rhs.slack),
                    rhs: self.rhs.$method(rhs.rhs),
                }
            }
        }
        impl<'b, T> $trait<&'b TableauRow<T>> for TableauRow<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = TableauRow<T>;
            fn $method(self, rhs: &'b TableauRow<T>) -> TableauRow<T> {
                (&self).$method(rhs)
            }
        }
        impl<'a, T> $trait<TableauRow<T>> for &'a TableauRow<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = TableauRow<T>;
            fn $method(self, rhs: TableauRow<T>) -> TableauRow<T> {
                self.$method(&rhs)
            }
        }
        impl<T> $trait<TableauRow<T>> for TableauRow<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = TableauRow<T>;
            fn $method(self, rhs: TableauRow<T>) -> TableauRow<T> {
                (&self).$method(&rhs)
            }
        }
        impl<'a, T> $trait<T> for &'a TableauRow<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = TableauRow<T>;
            fn $method(self, rhs: T) -> TableauRow<T> {
                TableauRow {
                    coefficients: (&self.coefficients).$method(rhs),
                    slack: (&self.slack).$method(rhs),
                    rhs: self.rhs.$method(rhs),
                }
            }
        }
        impl<T> $trait<T> for TableauRow<T>
        where
            T: Copy + $trait<Output = T>,
        {
            type Output = TableauRow<T>;
            fn $method(self, rhs: T) -> TableauRow<T> {
                (&self).$method(rhs)
            }
        }
    };
}

macro_rules! impl_tableau_row_assign_ops {
    ($assign_trait:ident, $assign_method:ident) => {
        impl<'a, T> $assign_trait<&'a TableauRow<T>> for TableauRow<T>
        where
            T: Copy + $assign_trait,
        {
            fn $assign_method(&mut self, rhs: &'a TableauRow<T>) {
                self.coefficients.$assign_method(&rhs.coefficients);
                self.slack.$assign_method(&rhs.slack);
                self.rhs.$assign_method(rhs.rhs);
            }
        }
        impl<T> $assign_trait<T> for TableauRow<T>
        where
            T: Copy + $assign_trait,
        {
            fn $assign_method(&mut self, rhs: T) {
                self.coefficients.$assign_method(rhs);
                self.slack.$assign_method(rhs);
                self.rhs.$assign_method(rhs);
            }
        }
        impl<'a, 'b, T> $assign_trait<&'b TableauRow<T>> for TableauRowMut<'a, T>
        where
            T: Copy + $assign_trait,
        {
            fn $assign_method(&mut self, rhs: &'b TableauRow<T>) {
                self.coefficients.$assign_method(&rhs.coefficients);
                self.slack.$assign_method(&rhs.slack);
                (*self.rhs).$assign_method(rhs.rhs);
            }
        }
        impl<'a, T> $assign_trait<TableauRow<T>> for TableauRowMut<'a, T>
        where
            T: Copy + $assign_trait,
        {
            fn $assign_method(&mut self, rhs: TableauRow<T>) {
                self.coefficients.$assign_method(rhs.coefficients);
                self.slack.$assign_method(rhs.slack);
                (*self.rhs).$assign_method(rhs.rhs);
            }
        }
        impl<'a, T> $assign_trait<T> for TableauRowMut<'a, T>
        where
            T: Copy + $assign_trait,
        {
            fn $assign_method(&mut self, rhs: T) {
                self.coefficients.$assign_method(rhs);
                self.slack.$assign_method(rhs);
                (*self.rhs).$assign_method(rhs);
            }
        }
    };
}

impl<T> Tableau<T> 
where T: Zero + PartialOrd + Clone + Copy + Div<Output = T> 
{
    /// Z-row entries (column index, reduced cost) for coefficients then slack.
    fn z_row_entries(&self) -> impl Iterator<Item = (usize, T)> + '_ {
        let n = self.z_coeffs.len();
        self.z_coeffs
            .iter()
            .enumerate()
            .map(move |(j, &v)| (j, v))
            .chain(
                self.z_slack
                    .iter()
                    .enumerate()
                    .map(move |(j, &v)| (n + j, v)),
            )
    }

    /// Pivot column by Dantzig rule (most negative reduced cost).
    pub fn find_pivot_col_most_negative(&self) -> Option<usize> {
        let mut best_col = None;
        let mut min_val = T::zero();
        for (j, val) in self.z_row_entries() {
            if val < min_val {
                min_val = val;
                best_col = Some(j);
            }
        }
        best_col
    }

    /// Pivot column by Bland rule (first negative reduced cost).
    pub fn find_pivot_col_bland(&self) -> Option<usize> {
        self.z_row_entries()
            .find(|(_, val)| *val < T::zero())
            .map(|(j, _)| j)
    }

    /// Minimum-ratio test: returns leaving row for the given entering column, or None.
    pub fn ratio_test(&self, col: usize) -> Option<usize> {
        let mut best_row = None;
        let mut min_ratio: Option<T> = None;

        for i in 0..self.rows() {
            let entry = self[(i, col)];

            if entry > T::zero() {
                let ratio = self.rhs[i] / entry;
                if min_ratio.is_none() || ratio < min_ratio.unwrap() {
                    min_ratio = Some(ratio);
                    best_row = Some(i);
                }
            }
        }
        best_row
    }

    /// Chooses pivot (Dantzig column, ratio test row); returns Optimal, Unbounded, or Pivot(row, col).
    pub fn find_pivot_indices(&self) -> PivotResult {
        match self.find_pivot_col_most_negative() {
            None => PivotResult::Optimal,
            Some(col) => match self.ratio_test(col) {
                Some(row) => PivotResult::Pivot(row, col),
                None => PivotResult::Unbounded,
            },
        }
    }

    /// Same as find_pivot_indices but uses Bland's rule (first negative column) to avoid cycling.
    pub fn find_pivot_indices_bland(&self) -> PivotResult {
        match self.find_pivot_col_bland() {
            None => PivotResult::Optimal,
            Some(col) => match self.ratio_test(col) {
                Some(row) => PivotResult::Pivot(row, col),
                None => PivotResult::Unbounded,
            },
        }
    }

    /// Current BFS as a vector of length n_vars (non-basic vars = 0, basic = RHS of defining row).
    pub fn current_vertex(&self, n_vars: usize) -> Vec<T>
    where
        T: Zero + Clone,
    {
        let mut vertex = vec![T::zero(); n_vars];
        for (row, &var_idx) in self.basis.iter().enumerate() {
            if var_idx < n_vars {
                vertex[var_idx] = self.rhs[row].clone();
            }
        }
        vertex
    }

    /// Returns true when no reduced cost is negative.
    pub fn is_optimal(&self) -> bool {
        self.find_pivot_col_most_negative().is_none()
    }
}

impl<T> Tableau<T>
where
    T: Zero
        + One
        + PartialOrd
        + Clone
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + SubAssign
        + MulAssign,
{
    /// Performs a pivot at (row_idx, col_idx); updates basis and all rows including z.
    pub fn pivot(&mut self, row_idx: usize, col_idx: usize) {
        let z_factor = self.z_row()[col_idx];
        let pivot_element = self[(row_idx, col_idx)];
        let inv_pivot = T::one() / pivot_element;

        {
            let mut p_row = self.row_mut(row_idx);
            p_row *= inv_pivot;
        }
        let norm = self.row(row_idx);

        for i in 0..self.rows() {
            if i != row_idx {
                let factor = self[(i, col_idx)];
                let mut current = self.row_mut(i);
                current.sub_assign_scaled(&norm, factor);
            }
        }

        {
            let mut z_row = self.z_row_mut();
            z_row.sub_assign_scaled(&norm, z_factor);
        }
        self.basis[row_idx] = col_idx;
    }
}

impl_tableau_row_binary_ops!(Add, add);
impl_tableau_row_assign_ops!(AddAssign, add_assign);

impl_tableau_row_binary_ops!(Sub, sub);
impl_tableau_row_assign_ops!(SubAssign, sub_assign);

impl_tableau_row_binary_ops!(Mul, mul);
impl_tableau_row_assign_ops!(MulAssign, mul_assign);

impl_tableau_row_binary_ops!(Div, div);
impl_tableau_row_assign_ops!(DivAssign, div_assign);

