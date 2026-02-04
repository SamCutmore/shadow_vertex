use crate::linalg::{Matrix, Row, RowMut};
use num_traits::Zero;
use std::ops::{Index, IndexMut};

#[derive(Clone, Copy)]
enum TableauColumn {
    Coeff(usize),
    Slack(usize),
    Rhs,
}

fn resolve_row_column(a_len: usize, s_len: usize, c: usize) -> TableauColumn {
    if c < a_len {
        TableauColumn::Coeff(c)
    } else if c < a_len + s_len {
        TableauColumn::Slack(c - a_len)
    } else {
        TableauColumn::Rhs
    }
}

/// Tableau: coefficient matrix, slack matrix, RHS, basis, nonbasis, and z-row (z_coeffs, z_slack, z_rhs).
#[derive(Debug, Clone)]
pub struct Tableau<T> {
    pub coefficients: Matrix<T>,
    pub slack: Matrix<T>,
    pub rhs: Vec<T>,
    pub basis: Vec<usize>,
    pub nonbasis: Vec<usize>,
    pub z_coeffs: Vec<T>,
    pub z_slack: Vec<T>,
    pub z_rhs: T,
}

impl<T> Tableau<T>
where
    T: Clone + Default,
{
    /// Builds a tableau from coefficient matrix, slack matrix, RHS, and z-row; basis = [n..n+m], nonbasis = [0..n].
    pub fn new(coefficients: Matrix<T>, slack: Matrix<T>, rhs: Vec<T>, z_coeffs: Vec<T>, z_slack: Vec<T>, z_rhs: T) -> Self {
        let m = coefficients.rows;
        let n = coefficients.cols;

        assert_eq!(slack.rows, m, "Slack rows must equal constraint rows");
        assert_eq!(slack.cols, m, "Slack must be square (m x m)");
        assert_eq!(rhs.len(), m, "RHS length must equal number of rows");
        
        assert_eq!(z_coeffs.len(), n, "Objective coefficients must match number of variables");
        assert_eq!(z_slack.len(), m, "Objective slack vector must match number of constraints");

        let basis: Vec<usize> = (n..n + m).collect();
        let nonbasis: Vec<usize> = (0..n).collect();

        Self {
            coefficients,
            slack,
            rhs,
            basis,
            nonbasis,
            z_coeffs,
            z_slack,
            z_rhs,
        }
    }
}

impl<T> Tableau<T> {
    /// Number of constraint rows.
    pub fn rows(&self) -> usize {
        self.coefficients.rows
    }

    pub fn cols(&self) -> usize {
        self.coefficients.cols + self.slack.cols + 1
    }

    fn resolve_column(&self, c: usize) -> TableauColumn {
        let a_cols = self.coefficients.cols;
        let s_cols = self.slack.cols;
        if c < a_cols {
            TableauColumn::Coeff(c)
        } else if c < a_cols + s_cols {
            TableauColumn::Slack(c - a_cols)
        } else {
            TableauColumn::Rhs
        }
    }
}

impl<T> Tableau<T>
where
    T: PartialOrd + Zero,
{
    /// Returns true if any RHS entry is negative.
    pub fn has_negative_rhs(&self) -> bool {
        self.rhs.iter().any(|v| *v < T::zero())
    }
}

impl<T> Index<(usize, usize)> for Tableau<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (r, c) = index;
        debug_assert!(r < self.rows());
        debug_assert!(c < self.cols());
        match self.resolve_column(c) {
            TableauColumn::Coeff(j) => &self.coefficients[(r, j)],
            TableauColumn::Slack(j) => &self.slack[(r, j)],
            TableauColumn::Rhs => &self.rhs[r],
        }
    }
}

impl<T> IndexMut<(usize, usize)> for Tableau<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (r, c) = index;
        debug_assert!(r < self.rows());
        debug_assert!(c < self.cols());
        match self.resolve_column(c) {
            TableauColumn::Coeff(j) => &mut self.coefficients[(r, j)],
            TableauColumn::Slack(j) => &mut self.slack[(r, j)],
            TableauColumn::Rhs => &mut self.rhs[r],
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
    /// Returns a copy of the given constraint row (coefficients, slack, rhs).
    pub fn row(&self, r: usize) -> TableauRow<T> {
        TableauRow {
            coefficients: self.coefficients.row(r),
            slack: self.slack.row(r),
            rhs: self.rhs[r].clone(),
        }
    }

    /// Returns a mutable view of the given constraint row.
    pub fn row_mut(&mut self, r: usize) -> TableauRowMut<'_, T> {
        TableauRowMut {
            coefficients: self.coefficients.row_mut(r),
            slack: self.slack.row_mut(r),
            rhs: &mut self.rhs[r],
        }
    }

    /// Returns a copy of the z-row.
    pub fn z_row(&self) -> TableauRow<T> {
        TableauRow {
            coefficients: Row { data: self.z_coeffs.clone() },
            slack: Row { data: self.z_slack.clone() },
            rhs: self.z_rhs.clone(),
        }
    }

    /// Returns a mutable view of the z-row.
    pub fn z_row_mut(&mut self) -> TableauRowMut<'_, T> {
        TableauRowMut {
            coefficients: RowMut { data: &mut self.z_coeffs },
            slack: RowMut { data: &mut self.z_slack },
            rhs: &mut self.z_rhs,
        }
    }
}

impl<T> TableauRow<T> {
    /// Logical column count for this row.
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
        match resolve_row_column(a, s, c) {
            TableauColumn::Coeff(j) => &self.coefficients.data[j],
            TableauColumn::Slack(j) => &self.slack.data[j],
            TableauColumn::Rhs => &self.rhs,
        }
    }
}

impl<'a, T> TableauRowMut<'a, T> {
    /// Logical column count for this row.
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
        match resolve_row_column(a, s, c) {
            TableauColumn::Coeff(j) => &self.coefficients.data[j],
            TableauColumn::Slack(j) => &self.slack.data[j],
            TableauColumn::Rhs => &self.rhs,
        }
    }
}

impl<'a, T> IndexMut<usize> for TableauRowMut<'a, T> {
    fn index_mut(&mut self, c: usize) -> &mut Self::Output {
        let a = self.coefficients.data.len();
        let s = self.slack.data.len();
        debug_assert!(c < a + s + 1);
        match resolve_row_column(a, s, c) {
            TableauColumn::Coeff(j) => &mut self.coefficients.data[j],
            TableauColumn::Slack(j) => &mut self.slack.data[j],
            TableauColumn::Rhs => self.rhs,
        }
    }
}