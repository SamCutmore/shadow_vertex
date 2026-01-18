use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};
use crate::tableau::{TableauRow, TableauRowMut};

#[inline]
fn assert_same_shape<T>(a: &TableauRow<T>, b: &TableauRow<T>) {
    debug_assert_eq!(
        a.coefficients.data.len() + a.slack.data.len(),
        b.coefficients.data.len() + b.slack.data.len()
    );
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