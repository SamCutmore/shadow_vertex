use crate::linalg::{Row, RowMut};
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};

// Addition
impl<T> Add for Row<T>
where T: Add<Output = T>
{
    type Output = Row<T>;
    fn add(self, rhs: Self) -> Row<T> {
        Row { data: self.data.into_iter().zip(rhs.data.into_iter()).map(|(a, b)| a + b).collect() }
    }
}

impl<T> Add<T> for Row<T>
where T: Copy + Add<Output = T>,
{
    type Output = Row<T>;
    fn add(self, rhs: T) -> Row<T> {
        Row { data: self.data.into_iter().map(|a| a + rhs).collect() }
    }
}

//References
impl<'a, 'b, T> Add<&'b Row<T>> for &'a Row<T>
where T: Copy + Add<Output = T>,
{
    type Output = Row<T>;

    fn add(self, rhs: &'b Row<T>) -> Row<T> {
        Row {
            data: self.data.iter().zip(rhs.data.iter()).map(|(a, b)| *a + *b).collect(),
        }
    }
}

impl<'a, T> Add<T> for &'a Row<T>
where T: Copy + Add<Output = T>,
{
    type Output = Row<T>;

    fn add(self, rhs: T) -> Row<T> {
        Row {
            data: self.data.iter().map(|&x| x + rhs).collect(),
        }
    }
}

impl<T> AddAssign<T> for Row<T>
where T: Copy + AddAssign,
{
    fn add_assign(&mut self, rhs: T) {
        for a in self.data.iter_mut() {
            *a += rhs;
        }
    }
}

impl<T> AddAssign<Row<T>> for Row<T>
where T: AddAssign,
{
    fn add_assign(&mut self, rhs: Row<T>) {
        for (a, b) in self.iter_mut().zip(rhs.data) {
            *a += b;
        }
    }
}

impl<'a, T> AddAssign<T> for RowMut<'a, T>
where T: Copy + AddAssign,
{
    fn add_assign(&mut self, rhs: T) {
        for a in self.data.iter_mut() {
            *a += rhs;
        }
    }
}

impl<'a, T> AddAssign<&'a Row<T>> for Row<T>
where T: Copy + AddAssign,
{
    fn add_assign(&mut self, rhs: &'a Row<T>) {
        for (a, b) in self.data.iter_mut().zip(&rhs.data) {
            *a += *b;
        }
    }
}

impl<'a, T> AddAssign<&Row<T>> for RowMut<'a, T>
where T: Copy + AddAssign,
{
    fn add_assign(&mut self, rhs: &Row<T>) {
        for (a, b) in self.iter_mut().zip(&rhs.data) {
            *a += *b;
        }
    }
}

impl<'a, T> AddAssign<Row<T>> for RowMut<'a, T>
where T: AddAssign
{
    fn add_assign(&mut self, rhs: Row<T>) {
        for (a, b) in self.iter_mut().zip(rhs.data) {
            *a += b;
        }
    }
}

// ==========================
// Subtraction
// ==========================

impl<T> Sub for Row<T>
where T: Sub<Output = T>
{
    type Output = Row<T>;
    fn sub(self, rhs: Self) -> Row<T> {
        Row { data: self.data.into_iter().zip(rhs.data.into_iter()).map(|(a, b)| a - b).collect() }
    }
}

impl<T> Sub<T> for Row<T>
where T: Copy + Sub<Output = T>,
{
    type Output = Row<T>;
    fn sub(self, rhs: T) -> Row<T> {
        Row { data: self.data.into_iter().map(|a| a - rhs).collect() }
    }
}

impl<'a, 'b, T> Sub<&'b Row<T>> for &'a Row<T>
where T: Copy + Sub<Output = T>,
{
    type Output = Row<T>;
    fn sub(self, rhs: &'b Row<T>) -> Row<T> {
        Row {
            data: self.data.iter().zip(rhs.data.iter()).map(|(a, b)| *a - *b).collect(),
        }
    }
}

impl<'a, T> Sub<T> for &'a Row<T>
where T: Copy + Sub<Output = T>,
{
    type Output = Row<T>;
    fn sub(self, rhs: T) -> Row<T> {
        Row {
            data: self.data.iter().map(|&x| x - rhs).collect(),
        }
    }
}

impl<T> SubAssign<T> for Row<T>
where T: Copy + SubAssign,
{
    fn sub_assign(&mut self, rhs: T) {
        for a in self.data.iter_mut() {
            *a -= rhs;
        }
    }
}

impl<T> SubAssign<Row<T>> for Row<T>
where T: SubAssign,
{
    fn sub_assign(&mut self, rhs: Row<T>) {
        for (a, b) in self.iter_mut().zip(rhs.data) {
            *a -= b;
        }
    }
}

impl<'a, T> SubAssign<&'a Row<T>> for Row<T>
where T: Copy + SubAssign,
{
    fn sub_assign(&mut self, rhs: &'a Row<T>) {
        for (a, b) in self.data.iter_mut().zip(&rhs.data) {
            *a -= *b;
        }
    }
}

impl<'a, T> SubAssign<T> for RowMut<'a, T>
where T: Copy + SubAssign,
{
    fn sub_assign(&mut self, rhs: T) {
        for a in self.data.iter_mut() {
            *a -= rhs;
        }
    }
}

impl<'a, T> SubAssign<&Row<T>> for RowMut<'a, T>
where T: Copy + SubAssign,
{
    fn sub_assign(&mut self, rhs: &Row<T>) {
        for (a, b) in self.iter_mut().zip(&rhs.data) {
            *a -= *b;
        }
    }
}

impl<'a, T> SubAssign<Row<T>> for RowMut<'a, T>
where T: SubAssign
{
    fn sub_assign(&mut self, rhs: Row<T>) {
        for (a, b) in self.iter_mut().zip(rhs.data) {
            *a -= b;
        }
    }
}

// ==========================
// Multiplication
// ==========================

impl<T> Mul for Row<T>
where T: Mul<Output = T>
{
    type Output = Row<T>;
    fn mul(self, rhs: Self) -> Row<T> {
        Row { data: self.data.into_iter().zip(rhs.data.into_iter()).map(|(a, b)| a * b).collect() }
    }
}

impl<T> Mul<T> for Row<T>
where T: Copy + Mul<Output = T>,
{
    type Output = Row<T>;
    fn mul(self, rhs: T) -> Row<T> {
        Row { data: self.data.into_iter().map(|a| a * rhs).collect() }
    }
}

impl<'a, 'b, T> Mul<&'b Row<T>> for &'a Row<T>
where T: Copy + Mul<Output = T>,
{
    type Output = Row<T>;
    fn mul(self, rhs: &'b Row<T>) -> Row<T> {
        Row {
            data: self.data.iter().zip(rhs.data.iter()).map(|(a, b)| *a * *b).collect(),
        }
    }
}

impl<'a, T> Mul<T> for &'a Row<T>
where T: Copy + Mul<Output = T>,
{
    type Output = Row<T>;
    fn mul(self, rhs: T) -> Row<T> {
        Row {
            data: self.data.iter().map(|&x| x * rhs).collect(),
        }
    }
}

impl<T> MulAssign<T> for Row<T>
where T: Copy + MulAssign,
{
    fn mul_assign(&mut self, rhs: T) {
        for a in self.data.iter_mut() {
            *a *= rhs;
        }
    }
}

impl<T> MulAssign<Row<T>> for Row<T>
where T: MulAssign,
{
    fn mul_assign(&mut self, rhs: Row<T>) {
        for (a, b) in self.iter_mut().zip(rhs.data) {
            *a *= b;
        }
    }
}

impl<'a, T> MulAssign<&'a Row<T>> for Row<T>
where T: Copy + MulAssign,
{
    fn mul_assign(&mut self, rhs: &'a Row<T>) {
        for (a, b) in self.data.iter_mut().zip(&rhs.data) {
            *a *= *b;
        }
    }
}

impl<'a, T> MulAssign<T> for RowMut<'a, T>
where T: Copy + MulAssign,
{
    fn mul_assign(&mut self, rhs: T) {
        for a in self.data.iter_mut() {
            *a *= rhs;
        }
    }
}

impl<'a, T> MulAssign<&Row<T>> for RowMut<'a, T>
where T: Copy + MulAssign,
{
    fn mul_assign(&mut self, rhs: &Row<T>) {
        for (a, b) in self.iter_mut().zip(&rhs.data) {
            *a *= *b;
        }
    }
}

impl<'a, T> MulAssign<Row<T>> for RowMut<'a, T>
where T: MulAssign
{
    fn mul_assign(&mut self, rhs: Row<T>) {
        for (a, b) in self.iter_mut().zip(rhs.data) {
            *a *= b;
        }
    }
}

// ==========================
// Division
// ==========================

impl<T> Div for Row<T>
where T: Div<Output = T>
{
    type Output = Row<T>;
    fn div(self, rhs: Self) -> Row<T> {
        Row { data: self.data.into_iter().zip(rhs.data.into_iter()).map(|(a, b)| a / b).collect() }
    }
}

impl<T> Div<T> for Row<T>
where T: Copy + Div<Output = T>,
{
    type Output = Row<T>;
    fn div(self, rhs: T) -> Row<T> {
        Row { data: self.data.into_iter().map(|a| a / rhs).collect() }
    }
}

impl<'a, 'b, T> Div<&'b Row<T>> for &'a Row<T>
where T: Copy + Div<Output = T>,
{
    type Output = Row<T>;
    fn div(self, rhs: &'b Row<T>) -> Row<T> {
        Row {
            data: self.data.iter().zip(rhs.data.iter()).map(|(a, b)| *a / *b).collect(),
        }
    }
}

impl<'a, T> Div<T> for &'a Row<T>
where T: Copy + Div<Output = T>,
{
    type Output = Row<T>;
    fn div(self, rhs: T) -> Row<T> {
        Row {
            data: self.data.iter().map(|&x| x / rhs).collect(),
        }
    }
}

impl<T> DivAssign<T> for Row<T>
where T: Copy + DivAssign,
{
    fn div_assign(&mut self, rhs: T) {
        for a in self.data.iter_mut() {
            *a /= rhs;
        }
    }
}

impl<T> DivAssign<Row<T>> for Row<T>
where T: DivAssign,
{
    fn div_assign(&mut self, rhs: Row<T>) {
        for (a, b) in self.iter_mut().zip(rhs.data) {
            *a /= b;
        }
    }
}

impl<'a, T> DivAssign<&'a Row<T>> for Row<T>
where T: Copy + DivAssign,
{
    fn div_assign(&mut self, rhs: &'a Row<T>) {
        for (a, b) in self.data.iter_mut().zip(&rhs.data) {
            *a /= *b;
        }
    }
}

impl<'a, T> DivAssign<T> for RowMut<'a, T>
where T: Copy + DivAssign,
{
    fn div_assign(&mut self, rhs: T) {
        for a in self.data.iter_mut() {
            *a /= rhs;
        }
    }
}

impl<'a, T> DivAssign<&Row<T>> for RowMut<'a, T>
where T: Copy + DivAssign,
{
    fn div_assign(&mut self, rhs: &Row<T>) {
        for (a, b) in self.iter_mut().zip(&rhs.data) {
            *a /= *b;
        }
    }
}

impl<'a, T> DivAssign<Row<T>> for RowMut<'a, T>
where T: DivAssign
{
    fn div_assign(&mut self, rhs: Row<T>) {
        for (a, b) in self.iter_mut().zip(rhs.data) {
            *a /= b;
        }
    }
}