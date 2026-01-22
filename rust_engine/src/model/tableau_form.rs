use crate::linalg::{Matrix, Row, RowMut};
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone)]
pub struct Tableau<T> {
    pub coefficients: Matrix<T>,    // - Coefficient Matrix
    pub slack: Matrix<T>,           // - Slack Variables
    pub rhs: Vec<T>,                // - Right Hand Side
    pub basis: Vec<usize>,          // - Basic Variables
    pub nonbasis: Vec<usize>,       // - Non-basic Variables

    pub z_coeffs: Vec<T>,           // - Z Row Values
    pub z_slack: Vec<T>,
    pub z_rhs: T,
}

impl<T> Tableau<T>
where T: Clone + Default,
{
    pub fn new(coefficients: Matrix<T>, slack: Matrix<T>, rhs: Vec<T>, z_coeffs: Vec<T>, z_slack: Vec<T>, z_rhs: T,) -> Self {
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
            &self.coefficients[(r, c)]
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
            &mut self.coefficients[(r, c)]
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
            coefficients: self.coefficients.row_mut(r),
            slack: self.slack.row_mut(r),
            rhs: &mut self.rhs[r],
        }
    }

    pub fn z_row(&self) -> TableauRow<T> {
        TableauRow {
            coefficients: Row { data: self.z_coeffs.clone() },
            slack: Row { data: self.z_slack.clone() },
            rhs: self.z_rhs.clone(),
        }
    }

    pub fn z_row_mut(&mut self) -> TableauRowMut<'_, T> {
        TableauRowMut {
            coefficients: RowMut { data: &mut self.z_coeffs },
            slack: RowMut { data: &mut self.z_slack },
            rhs: &mut self.z_rhs,
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
            &self.coefficients.data[c]
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
            &self.coefficients.data[c]
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
            &mut self.coefficients.data[c]
        } else if c < a + s {
            &mut self.slack.data[c - a]
        } else {
            self.rhs
        }
    }
}