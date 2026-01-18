use crate::matrix_adt::*;
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone)]
pub struct Tableau<T> {
    pub coefficients: Matrix<T>, // - Constraint Matrix
    pub slack: Matrix<T>,       // - Slack Variables
    pub rhs: Vec<T>,            // - Right Hand Side
    pub basis: Vec<usize>,      // - Basic Variables
    pub nonbasis: Vec<usize>,   // - Non-basic Variables
}

impl<T> Tableau<T>
where T: Clone + Default,
{
    pub fn from_standard_form(coefficients: Matrix<T>,slack: Matrix<T>, rhs: Vec<T>,) -> Self {
        let m = coefficients.rows;
        let n = coefficients.cols;

        assert_eq!(slack.rows, m, "slack rows must equal constraint rows");
        assert_eq!(slack.cols, m, "slack must be square (m x m)");
        assert_eq!(rhs.len(), m, "rhs length must equal number of rows");

        let basis: Vec<usize> = (n..n + m).collect();
        let nonbasis: Vec<usize> = (0..n).collect();

        Self {
            coefficients,
            slack,
            rhs,
            basis,
            nonbasis,
        }
    }
}

impl<T> Tableau<T> {
    pub fn rows(&self) -> usize {
        self.coefficients.rows
    }

    pub fn cols(&self) -> usize {
        self.coefficients.cols + self.slack.cols + 1
    }
}

impl<T> Index<(usize, usize)> for Tableau<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (r, c) = index;
        debug_assert!(r < self.rows());
        debug_assert!(c < self.cols());

        let a_cols = self.coefficients.cols;
        let s_cols = self.slack.cols;

        if c < a_cols {
            &self.coefficients
[(r, c)]
        } else if c < a_cols + s_cols {
            &self.slack[(r, c - a_cols)]
        } else {
            &self.rhs[r]
        }
    }
}

impl<T> IndexMut<(usize, usize)> for Tableau<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (r, c) = index;
        debug_assert!(r < self.rows());
        debug_assert!(c < self.cols());

        let a_cols = self.coefficients.cols;
        let s_cols = self.slack.cols;

        if c < a_cols {
            &mut self.coefficients
[(r, c)]
        } else if c < a_cols + s_cols {
            &mut self.slack[(r, c - a_cols)]
        } else {
            &mut self.rhs[r]
        }
    }
}

#[derive(Debug, Clone)]
pub struct TableauRow<T> {
    pub coefficients: Row<T>,
    pub slack: Row<T>,
    pub rhs: T,
}

pub struct TableauRowMut<'a, T> {
    pub coefficients: RowMut<'a, T>,
    pub slack: RowMut<'a, T>,
    pub rhs: &'a mut T,
}

impl<T: Clone> Tableau<T> {
    pub fn row(&self, r: usize) -> TableauRow<T> {
        TableauRow {
            coefficients: self.coefficients.row(r),
            slack: self.slack.row(r),
            rhs: self.rhs[r].clone(),
        }
    }

    pub fn row_mut(&mut self, r: usize) -> TableauRowMut<'_, T> {
        TableauRowMut {
            coefficients : self.coefficients.row_mut(r),
            slack: self.slack.row_mut(r),
            rhs: &mut self.rhs[r],
        }
    }
}

impl<T> TableauRow<T> {
    pub fn cols(&self) -> usize {
        self.coefficients.data.len() + self.slack.data.len() + 1
    }
}

impl<T> Index<usize> for TableauRow<T> {
    type Output = T;

    fn index(&self, c: usize) -> &Self::Output {
        let a = self.coefficients.data.len();
        let s = self.slack.data.len();

        debug_assert!(c < a + s + 1);

        if c < a {
            &self.coefficients
.data[c]
        } else if c < a + s {
            &self.slack.data[c - a]
        } else {
            &self.rhs
        }
    }
}

impl<'a, T> TableauRowMut<'a, T> {
    pub fn cols(&self) -> usize {
        self.coefficients.data.len() + self.slack.data.len() + 1
    }
}

impl<'a, T> Index<usize> for TableauRowMut<'a, T> {
    type Output = T;

    fn index(&self, c: usize) -> &Self::Output {
        let a = self.coefficients.data.len();
        let s = self.slack.data.len();

        debug_assert!(c < a + s + 1);

        if c < a {
            &self.coefficients
.data[c]
        } else if c < a + s {
            &self.slack.data[c - a]
        } else {
            &self.rhs
        }
    }
}

impl<'a, T> IndexMut<usize> for TableauRowMut<'a, T> {
    fn index_mut(&mut self, c: usize) -> &mut Self::Output {
        let a = self.coefficients.data.len();
        let s = self.slack.data.len();

        debug_assert!(c < a + s + 1);

        if c < a {
            &mut self.coefficients
.data[c]
        } else if c < a + s {
            &mut self.slack.data[c - a]
        } else {
            self.rhs
        }
    }
}