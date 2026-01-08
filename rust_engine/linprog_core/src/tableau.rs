use crate::matrix_adt::*;
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone)]
pub struct Tableau<T> {
    pub constraints: Matrix<T>, // - Constraint Matrix
    pub slack: Matrix<T>,       // - Slack Variables
    pub rhs: Vec<T>,            // - Right Hand Side
    pub cost: Vec<T>,           // - Cost
    pub basis: Vec<usize>,      // - Basic Variables
    pub nonbasis: Vec<usize>,   // - Non-basic Variables
}

impl<T> Tableau<T>
where T: Clone + Default,
{
    pub fn from_standard_form(constraints: Matrix<T>,slack: Matrix<T>, rhs: Vec<T>, cost: Vec<T>, ) -> Self {
        let m = constraints.rows;
        let n = constraints.cols;

        assert_eq!(slack.rows, m, "slack rows must equal constraint rows");
        assert_eq!(slack.cols, m, "slack must be square (m x m)");
        assert_eq!(rhs.len(), m, "rhs length must equal number of rows");
        assert_eq!(cost.len(), n + m, "cost length must equal total number of variables");

        let basis: Vec<usize> = (n..n + m).collect();
        let nonbasis: Vec<usize> = (0..n).collect();

        Self {
            constraints,
            slack,
            rhs,
            cost,
            basis,
            nonbasis,
        }
    }
}

impl<T> Tableau<T> {
    pub fn rows(&self) -> usize {
        self.constraints.rows
    }

    pub fn cols(&self) -> usize {
        self.constraints.cols + self.slack.cols + 1
    }
}

impl<T> Index<(usize, usize)> for Tableau<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (r, c) = index;
        debug_assert!(r < self.rows());
        debug_assert!(c < self.cols());

        let a_cols = self.constraints.cols;
        let s_cols = self.slack.cols;

        if c < a_cols {
            &self.constraints[(r, c)]
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

        let a_cols = self.constraints.cols;
        let s_cols = self.slack.cols;

        if c < a_cols {
            &mut self.constraints[(r, c)]
        } else if c < a_cols + s_cols {
            &mut self.slack[(r, c - a_cols)]
        } else {
            &mut self.rhs[r]
        }
    }
}

#[derive(Debug, Clone)]
pub struct TableauRow<T> {
    pub constraints: Row<T>,
    pub slack: Row<T>,
    pub rhs: T,
}

pub struct TableauRowMut<'a, T> {
    pub constraints: RowMut<'a, T>,
    pub slack: RowMut<'a, T>,
    pub rhs: &'a mut T,
}

impl<T: Clone> Tableau<T> {
    pub fn row(&self, r: usize) -> TableauRow<T> {
        TableauRow {
            constraints: self.constraints.row(r),
            slack: self.slack.row(r),
            rhs: self.rhs[r].clone(),
        }
    }

    pub fn row_mut(&mut self, r: usize) -> TableauRowMut<'_, T> {
        TableauRowMut {
            constraints: self.constraints.row_mut(r),
            slack: self.slack.row_mut(r),
            rhs: &mut self.rhs[r],
        }
    }
}

impl<T> TableauRow<T> {
    pub fn cols(&self) -> usize {
        self.constraints.data.len() + self.slack.data.len() + 1
    }
}

impl<T> Index<usize> for TableauRow<T> {
    type Output = T;

    fn index(&self, c: usize) -> &Self::Output {
        let a = self.constraints.data.len();
        let s = self.slack.data.len();

        debug_assert!(c < a + s + 1);

        if c < a {
            &self.constraints.data[c]
        } else if c < a + s {
            &self.slack.data[c - a]
        } else {
            &self.rhs
        }
    }
}

impl<'a, T> TableauRowMut<'a, T> {
    pub fn cols(&self) -> usize {
        self.constraints.data.len() + self.slack.data.len() + 1
    }
}

impl<'a, T> Index<usize> for TableauRowMut<'a, T> {
    type Output = T;

    fn index(&self, c: usize) -> &Self::Output {
        let a = self.constraints.data.len();
        let s = self.slack.data.len();

        debug_assert!(c < a + s + 1);

        if c < a {
            &self.constraints.data[c]
        } else if c < a + s {
            &self.slack.data[c - a]
        } else {
            &self.rhs
        }
    }
}

impl<'a, T> IndexMut<usize> for TableauRowMut<'a, T> {
    fn index_mut(&mut self, c: usize) -> &mut Self::Output {
        let a = self.constraints.data.len();
        let s = self.slack.data.len();

        debug_assert!(c < a + s + 1);

        if c < a {
            &mut self.constraints.data[c]
        } else if c < a + s {
            &mut self.slack.data[c - a]
        } else {
            self.rhs
        }
    }
}