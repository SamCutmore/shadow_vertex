use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};
use crate::model::{Tableau, TableauRow, TableauRowMut};
use num_traits::{One, Zero};

#[inline]
fn assert_same_shape<T>(a: &TableauRow<T>, b: &TableauRow<T>) {
    debug_assert_eq!(
        a.coefficients.data.len() + a.slack.data.len(),
        b.coefficients.data.len() + b.slack.data.len()
    );
}

impl<T> Tableau<T> 
where T: Zero + PartialOrd + Clone + Copy + Div<Output = T> 
{
    /// Dantzig's Rule
    pub fn find_pivot_col_most_negative(&self) -> Option<usize> {
        let mut best_col = None;
        let mut min_val = T::zero();

        for (j, val) in self.z_coeffs.iter().enumerate() {
            if *val < min_val {
                min_val = *val;
                best_col = Some(j);
            }
        }

        let n = self.z_coeffs.len();
        for (j, val) in self.z_slack.iter().enumerate() {
            if *val < min_val {
                min_val = *val;
                best_col = Some(n + j);
            }
        }
        best_col
    }

    /// Bland's Rule
    pub fn find_pivot_col_bland(&self) -> Option<usize> {
        for (j, val) in self.z_coeffs.iter().enumerate() {
            if *val < T::zero() {
                return Some(j);
            }
        }

        let n = self.z_coeffs.len();
        for (j, val) in self.z_slack.iter().enumerate() {
            if *val < T::zero() {
                return Some(n + j);
            }
        }
        None
    }

    /// Minimum Ratio Test
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
}

impl<T> Tableau<T> 
where T: Zero + One + PartialOrd + Clone + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> 
{
    pub fn pivot(&mut self, row_idx: usize, col_idx: usize) {
        let num_cols = self.cols(); 
        let var_cols = num_cols - 1; // Exclude RHS from the variable loops
        
        let z_factor = self.z_row()[col_idx];
        let pivot_element = self[(row_idx, col_idx)];
        let inv_pivot = T::one() / pivot_element;

        {
            let mut p_row = self.row_mut(row_idx);
            for j in 0..num_cols {
                p_row[j] = p_row[j] * inv_pivot;
            }
        }
        
        let normalized_vars: Vec<T> = (0..var_cols).map(|j| self[(row_idx, j)]).collect();
        let normalized_rhs = self.row_mut(row_idx)[var_cols];

        for i in 0..self.rows() {
            if i != row_idx {
                let factor = self[(i, col_idx)];
                {
                    let mut current_row = self.row_mut(i);
                    for j in 0..var_cols {
                        current_row[j] = current_row[j] - (factor * normalized_vars[j]);
                    }
                    current_row[var_cols] = current_row[var_cols] - (factor * normalized_rhs);
                }
            }
        }

        {
            let mut z_row = self.z_row_mut();
            for j in 0..var_cols {
                z_row[j] = z_row[j] - (z_factor * normalized_vars[j]);
            }
        }
        
        self.z_rhs = self.z_rhs - (z_factor * normalized_rhs);
        self.basis[row_idx] = col_idx;
    }
}

// ====================================================
// Addition
// ====================================================

// ==========================
// TableauRow + TableauRow
// ==========================

// &TableauRow + &TableauRow
impl<'a, 'b, T> Add<&'b TableauRow<T>> for &'a TableauRow<T>
where T: Copy + Add<Output = T>,
{
    type Output = TableauRow<T>;

    fn add(self, rhs: &'b TableauRow<T>) -> TableauRow<T> {
        assert_same_shape(self, rhs);
        TableauRow {
            coefficients: &self.coefficients + &rhs.coefficients,
            slack: &self.slack + &rhs.slack,
            rhs: self.rhs + rhs.rhs,
        }
    }
}

// TableauRow + &TableauRow
impl<'b, T> Add<&'b TableauRow<T>> for TableauRow<T>
where T: Copy + Add<Output = T>,
{
    type Output = TableauRow<T>;
    fn add(self, rhs: &'b TableauRow<T>) -> TableauRow<T> {
        &self + rhs
    }
}

// &TableauRow + TableauRow
impl<'a, T> Add<TableauRow<T>> for &'a TableauRow<T>
where T: Copy + Add<Output = T>,
{
    type Output = TableauRow<T>;
    fn add(self, rhs: TableauRow<T>) -> TableauRow<T> {
        self + &rhs
    }
}

// TableauRow + TableauRow
impl<T> Add<TableauRow<T>> for TableauRow<T>
where T: Copy + Add<Output = T>,
{
    type Output = TableauRow<T>;
    fn add(self, rhs: TableauRow<T>) -> TableauRow<T> {
        &self + &rhs
    }
}

// ==========================
// TableauRow + scalar
// ==========================

// &TableauRow + scalar
impl<'a, T> Add<T> for &'a TableauRow<T>
where T: Copy + Add<Output = T>,
{
    type Output = TableauRow<T>;
    fn add(self, rhs: T) -> TableauRow<T> {
        TableauRow {
            coefficients: &self.coefficients + rhs,
            slack: &self.slack + rhs,
            rhs: self.rhs + rhs,
        }
    }
}

// TableauRow + scalar
impl<T> Add<T> for TableauRow<T>
where T: Copy + Add<Output = T>,
{
    type Output = TableauRow<T>;
    fn add(self, rhs: T) -> TableauRow<T> {
        &self + rhs
    }
}

// ==========================
// TableauRow += ...
// ==========================

impl<'a, T> AddAssign<&'a TableauRow<T>> for TableauRow<T>
where T: Copy + AddAssign 
{
    fn add_assign(&mut self, rhs: &'a TableauRow<T>) {
        self.coefficients += &rhs.coefficients;
        self.slack += &rhs.slack;
        self.rhs += rhs.rhs;
    }
}

impl<T> AddAssign<T> for TableauRow<T>
where T: Copy + AddAssign,
{
    fn add_assign(&mut self, rhs: T) {
        self.coefficients += rhs;
        self.slack += rhs;
        self.rhs += rhs;
    }
}

// ==========================
// TableauRowMut += ...
// ==========================

// TableauRowMut += &TableauRow
impl<'a, T> AddAssign<&TableauRow<T>> for TableauRowMut<'a, T>
where T: Copy + AddAssign,
{
    fn add_assign(&mut self, rhs: &TableauRow<T>) {
        self.coefficients += &rhs.coefficients;
        self.slack += &rhs.slack;
        *self.rhs += rhs.rhs;
    }
}

// TableauRowMut += TableauRow
impl<'a, T> AddAssign<TableauRow<T>> for TableauRowMut<'a, T>
where T: Copy + AddAssign,
{
    fn add_assign(&mut self, rhs: TableauRow<T>) {
        self.coefficients += rhs.coefficients;
        self.slack += rhs.slack;
        *self.rhs += rhs.rhs;
    }
}

// TableauRowMut += scalar
impl<'a, T> AddAssign<T> for TableauRowMut<'a, T>
where T: Copy + AddAssign,
{
    fn add_assign(&mut self, rhs: T) {
        self.coefficients += rhs;
        self.slack += rhs;
        *self.rhs += rhs;
    }
}

// ====================================================
// Subtraction
// ====================================================

// &TableauRow - &TableauRow
impl<'a, 'b, T> Sub<&'b TableauRow<T>> for &'a TableauRow<T>
where T: Copy + Sub<Output = T>,
{
    type Output = TableauRow<T>;

    fn sub(self, rhs: &'b TableauRow<T>) -> TableauRow<T> {
        assert_same_shape(self, rhs);
        TableauRow {
            coefficients: &self.coefficients - &rhs.coefficients,
            slack: &self.slack - &rhs.slack,
            rhs: self.rhs - rhs.rhs,
        }
    }
}

// TableauRow - &TableauRow
impl<'b, T> Sub<&'b TableauRow<T>> for TableauRow<T>
where T: Copy + Sub<Output = T>,
{
    type Output = TableauRow<T>;
    fn sub(self, rhs: &'b TableauRow<T>) -> TableauRow<T> {
        &self - rhs
    }
}

// &TableauRow - TableauRow
impl<'a, T> Sub<TableauRow<T>> for &'a TableauRow<T>
where T: Copy + Sub<Output = T>,
{
    type Output = TableauRow<T>;
    fn sub(self, rhs: TableauRow<T>) -> TableauRow<T> {
        self - &rhs
    }
}

// TableauRow - TableauRow
impl<T> Sub<TableauRow<T>> for TableauRow<T>
where T: Copy + Sub<Output = T>,
{
    type Output = TableauRow<T>;
    fn sub(self, rhs: TableauRow<T>) -> TableauRow<T> {
        &self - &rhs
    }
}

// &TableauRow - scalar
impl<'a, T> Sub<T> for &'a TableauRow<T>
where T: Copy + Sub<Output = T>,
{
    type Output = TableauRow<T>;
    fn sub(self, rhs: T) -> TableauRow<T> {
        TableauRow {
            coefficients: &self.coefficients - rhs,
            slack: &self.slack - rhs,
            rhs: self.rhs - rhs,
        }
    }
}

// TableauRow - scalar
impl<T> Sub<T> for TableauRow<T>
where T: Copy + Sub<Output = T>,
{
    type Output = TableauRow<T>;
    fn sub(self, rhs: T) -> TableauRow<T> {
        &self - rhs
    }
}

// TableauRow -= ...
impl<'a, T> SubAssign<&'a TableauRow<T>> for TableauRow<T>
where T: Copy + SubAssign 
{
    fn sub_assign(&mut self, rhs: &'a TableauRow<T>) {
        self.coefficients -= &rhs.coefficients;
        self.slack -= &rhs.slack;
        self.rhs -= rhs.rhs;
    }
}

impl<T> SubAssign<T> for TableauRow<T>
where T: Copy + SubAssign,
{
    fn sub_assign(&mut self, rhs: T) {
        self.coefficients -= rhs;
        self.slack -= rhs;
        self.rhs -= rhs;
    }
}

// TableauRowMut -= ...
impl<'a, T> SubAssign<&TableauRow<T>> for TableauRowMut<'a, T>
where T: Copy + SubAssign,
{
    fn sub_assign(&mut self, rhs: &TableauRow<T>) {
        self.coefficients -= &rhs.coefficients;
        self.slack -= &rhs.slack;
        *self.rhs -= rhs.rhs;
    }
}

impl<'a, T> SubAssign<TableauRow<T>> for TableauRowMut<'a, T>
where T: Copy + SubAssign,
{
    fn sub_assign(&mut self, rhs: TableauRow<T>) {
        self.coefficients -= rhs.coefficients;
        self.slack -= rhs.slack;
        *self.rhs -= rhs.rhs;
    }
}

impl<'a, T> SubAssign<T> for TableauRowMut<'a, T>
where T: Copy + SubAssign,
{
    fn sub_assign(&mut self, rhs: T) {
        self.coefficients -= rhs;
        self.slack -= rhs;
        *self.rhs -= rhs;
    }
}

// ====================================================
// Multiplication
// ====================================================

// &TableauRow * &TableauRow
impl<'a, 'b, T> Mul<&'b TableauRow<T>> for &'a TableauRow<T>
where T: Copy + Mul<Output = T>,
{
    type Output = TableauRow<T>;
    fn mul(self, rhs: &'b TableauRow<T>) -> TableauRow<T> {
        assert_same_shape(self, rhs);
        TableauRow {
            coefficients: &self.coefficients * &rhs.coefficients,
            slack: &self.slack * &rhs.slack,
            rhs: self.rhs * rhs.rhs,
        }
    }
}

impl<'b, T> Mul<&'b TableauRow<T>> for TableauRow<T>
where T: Copy + Mul<Output = T>,
{
    type Output = TableauRow<T>;
    fn mul(self, rhs: &'b TableauRow<T>) -> TableauRow<T> { &self * rhs }
}

impl<'a, T> Mul<TableauRow<T>> for &'a TableauRow<T>
where T: Copy + Mul<Output = T>,
{
    type Output = TableauRow<T>;
    fn mul(self, rhs: TableauRow<T>) -> TableauRow<T> { self * &rhs }
}

impl<T> Mul<TableauRow<T>> for TableauRow<T>
where T: Copy + Mul<Output = T>,
{
    type Output = TableauRow<T>;
    fn mul(self, rhs: TableauRow<T>) -> TableauRow<T> { &self * &rhs }
}

// TableauRow * scalar
impl<'a, T> Mul<T> for &'a TableauRow<T>
where T: Copy + Mul<Output = T>,
{
    type Output = TableauRow<T>;
    fn mul(self, rhs: T) -> TableauRow<T> {
        TableauRow {
            coefficients: &self.coefficients * rhs,
            slack: &self.slack * rhs,
            rhs: self.rhs * rhs,
        }
    }
}

impl<T> Mul<T> for TableauRow<T>
where T: Copy + Mul<Output = T>,
{
    type Output = TableauRow<T>;
    fn mul(self, rhs: T) -> TableauRow<T> { &self * rhs }
}

// Assignments
impl<'a, T> MulAssign<&'a TableauRow<T>> for TableauRow<T>
where T: Copy + MulAssign 
{
    fn mul_assign(&mut self, rhs: &'a TableauRow<T>) {
        self.coefficients *= &rhs.coefficients;
        self.slack *= &rhs.slack;
        self.rhs *= rhs.rhs;
    }
}

impl<T> MulAssign<T> for TableauRow<T>
where T: Copy + MulAssign,
{
    fn mul_assign(&mut self, rhs: T) {
        self.coefficients *= rhs;
        self.slack *= rhs;
        self.rhs *= rhs;
    }
}

impl<'a, T> MulAssign<&TableauRow<T>> for TableauRowMut<'a, T>
where T: Copy + MulAssign,
{
    fn mul_assign(&mut self, rhs: &TableauRow<T>) {
        self.coefficients *= &rhs.coefficients;
        self.slack *= &rhs.slack;
        *self.rhs *= rhs.rhs;
    }
}

impl<'a, T> MulAssign<TableauRow<T>> for TableauRowMut<'a, T>
where T: Copy + MulAssign,
{
    fn mul_assign(&mut self, rhs: TableauRow<T>) {
        self.coefficients *= rhs.coefficients;
        self.slack *= rhs.slack;
        *self.rhs *= rhs.rhs;
    }
}

impl<'a, T> MulAssign<T> for TableauRowMut<'a, T>
where T: Copy + MulAssign,
{
    fn mul_assign(&mut self, rhs: T) {
        self.coefficients *= rhs;
        self.slack *= rhs;
        *self.rhs *= rhs;
    }
}

// ====================================================
// Division
// ====================================================

// &TableauRow / &TableauRow
impl<'a, 'b, T> Div<&'b TableauRow<T>> for &'a TableauRow<T>
where T: Copy + Div<Output = T>,
{
    type Output = TableauRow<T>;
    fn div(self, rhs: &'b TableauRow<T>) -> TableauRow<T> {
        assert_same_shape(self, rhs);
        TableauRow {
            coefficients: &self.coefficients / &rhs.coefficients,
            slack: &self.slack / &rhs.slack,
            rhs: self.rhs / rhs.rhs,
        }
    }
}

impl<'b, T> Div<&'b TableauRow<T>> for TableauRow<T>
where T: Copy + Div<Output = T>,
{
    type Output = TableauRow<T>;
    fn div(self, rhs: &'b TableauRow<T>) -> TableauRow<T> { &self / rhs }
}

impl<'a, T> Div<TableauRow<T>> for &'a TableauRow<T>
where T: Copy + Div<Output = T>,
{
    type Output = TableauRow<T>;
    fn div(self, rhs: TableauRow<T>) -> TableauRow<T> { self / &rhs }
}

impl<T> Div<TableauRow<T>> for TableauRow<T>
where T: Copy + Div<Output = T>,
{
    type Output = TableauRow<T>;
    fn div(self, rhs: TableauRow<T>) -> TableauRow<T> { &self / &rhs }
}

// TableauRow / scalar
impl<'a, T> Div<T> for &'a TableauRow<T>
where T: Copy + Div<Output = T>,
{
    type Output = TableauRow<T>;
    fn div(self, rhs: T) -> TableauRow<T> {
        TableauRow {
            coefficients: &self.coefficients / rhs,
            slack: &self.slack / rhs,
            rhs: self.rhs / rhs,
        }
    }
}

impl<T> Div<T> for TableauRow<T>
where T: Copy + Div<Output = T>,
{
    type Output = TableauRow<T>;
    fn div(self, rhs: T) -> TableauRow<T> { &self / rhs }
}

// Assignments
impl<'a, T> DivAssign<&'a TableauRow<T>> for TableauRow<T>
where T: Copy + DivAssign 
{
    fn div_assign(&mut self, rhs: &'a TableauRow<T>) {
        self.coefficients /= &rhs.coefficients;
        self.slack /= &rhs.slack;
        self.rhs /= rhs.rhs;
    }
}

impl<T> DivAssign<T> for TableauRow<T>
where T: Copy + DivAssign,
{
    fn div_assign(&mut self, rhs: T) {
        self.coefficients /= rhs;
        self.slack /= rhs;
        self.rhs /= rhs;
    }
}

impl<'a, T> DivAssign<&TableauRow<T>> for TableauRowMut<'a, T>
where T: Copy + DivAssign,
{
    fn div_assign(&mut self, rhs: &TableauRow<T>) {
        self.coefficients /= &rhs.coefficients;
        self.slack /= &rhs.slack;
        *self.rhs /= rhs.rhs;
    }
}

impl<'a, T> DivAssign<TableauRow<T>> for TableauRowMut<'a, T>
where T: Copy + DivAssign,
{
    fn div_assign(&mut self, rhs: TableauRow<T>) {
        self.coefficients /= rhs.coefficients;
        self.slack /= rhs.slack;
        *self.rhs /= rhs.rhs;
    }
}

impl<'a, T> DivAssign<T> for TableauRowMut<'a, T>
where T: Copy + DivAssign,
{
    fn div_assign(&mut self, rhs: T) {
        self.coefficients /= rhs;
        self.slack /= rhs;
        *self.rhs /= rhs;
    }
}