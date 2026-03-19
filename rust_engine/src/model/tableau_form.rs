use crate::linalg::{Matrix, Row, RowMut};
use num_traits::Zero;
use std::ops::{Index, IndexMut};

/// Unified simplex tableau stored as a single (m+1) x (n+m+1) matrix:
///
/// ```text
///          col 0..n      col n..n+m     col n+m
///        ┌────────────┬────────────┬──────────┐
/// row 0  │            │            │          │
///   ..   │     A      │     S      │    b     │  m constraint rows
/// row m-1│            │            │          │
///        ├────────────┼────────────┼──────────┤
/// row m  │  z_coeffs  │  z_slack   │  z_rhs   │  objective row
///        └────────────┴────────────┴──────────┘
/// ```
#[derive(Debug, Clone)]
pub struct Tableau<T> {
    pub data: Matrix<T>,
    /// Number of structural (decision) variables.
    pub n: usize,
    /// Number of constraints.
    pub m: usize,
    pub basis: Vec<usize>,
    pub nonbasis: Vec<usize>,
}

impl<T> Tableau<T>
where
    T: Clone + Default,
{
    /// Builds a tableau from a pre-assembled (m+1) x (n+m+1) matrix.
    pub fn new(data: Matrix<T>, n: usize, m: usize) -> Self {
        assert_eq!(data.rows, m + 1, "Matrix must have m+1 rows");
        assert_eq!(data.cols, n + m + 1, "Matrix must have n+m+1 columns");

        let basis: Vec<usize> = (n..n + m).collect();
        let nonbasis: Vec<usize> = (0..n).collect();

        Self { data, n, m, basis, nonbasis }
    }

    /// Assembles a tableau from separate coefficient matrix, slack matrix, RHS,
    /// and z-row components into a single unified matrix.
    pub fn from_parts(
        coefficients: Matrix<T>,
        slack: Matrix<T>,
        rhs: Vec<T>,
        z_coeffs: Vec<T>,
        z_slack: Vec<T>,
        z_rhs: T,
    ) -> Self {
        let m = coefficients.rows;
        let n = coefficients.cols;

        assert_eq!(slack.rows, m, "Slack rows must equal constraint rows");
        assert_eq!(slack.cols, m, "Slack must be square (m x m)");
        assert_eq!(rhs.len(), m, "RHS length must equal number of rows");
        assert_eq!(z_coeffs.len(), n, "Objective coefficients must match number of variables");
        assert_eq!(z_slack.len(), m, "Objective slack vector must match number of constraints");

        let total_cols = n + m + 1;
        let mut data = Matrix::with_capacity(m + 1, total_cols);

        for i in 0..m {
            let mut row_data = Vec::with_capacity(total_cols);
            for j in 0..n { row_data.push(coefficients[(i, j)].clone()); }
            for j in 0..m { row_data.push(slack[(i, j)].clone()); }
            row_data.push(rhs[i].clone());
            data.push_row(&row_data);
        }

        let mut z_row_data = Vec::with_capacity(total_cols);
        z_row_data.extend(z_coeffs);
        z_row_data.extend(z_slack);
        z_row_data.push(z_rhs);
        data.push_row(&z_row_data);

        Self::new(data, n, m)
    }
}

impl<T> Tableau<T> {
    /// Number of constraint rows (excludes z-row).
    pub fn rows(&self) -> usize {
        self.m
    }

    /// Total columns including RHS.
    pub fn cols(&self) -> usize {
        self.n + self.m + 1
    }

    /// Number of variable columns (structural + slack, excludes RHS).
    pub fn num_vars(&self) -> usize {
        self.n + self.m
    }

    pub fn rhs_col(&self) -> usize {
        self.n + self.m
    }
}

impl<T: Clone> Tableau<T> {
    /// RHS value for constraint row i.
    pub fn rhs(&self, i: usize) -> T {
        self.data[(i, self.rhs_col())].clone()
    }

    /// Z-row RHS value (current objective).
    pub fn z_rhs(&self) -> T {
        self.data[(self.m, self.rhs_col())].clone()
    }

    /// Mutable reference to z-row RHS.
    pub fn z_rhs_mut(&mut self) -> &mut T {
        let r = self.m;
        let c = self.rhs_col();
        &mut self.data[(r, c)]
    }

    /// Returns a copy of the given row (all columns including RHS).
    pub fn row(&self, r: usize) -> Row<T> {
        self.data.row(r)
    }

    /// Returns a mutable view of the given row.
    pub fn row_mut(&mut self, r: usize) -> RowMut<'_, T> {
        self.data.row_mut(r)
    }

    /// Returns a copy of the z-row.
    pub fn z_row(&self) -> Row<T> {
        self.data.row(self.m)
    }

    /// Returns a mutable view of the z-row.
    pub fn z_row_mut(&mut self) -> RowMut<'_, T> {
        self.data.row_mut(self.m)
    }
}

impl<T: Copy> Tableau<T> {
    /// Copies an owned Row into matrix row `i`.
    /// Enables `tab.set_row(i, tab.row(j) - tab.row(k))`.
    pub fn set_row(&mut self, r: usize, row: &Row<T>) {
        assert_eq!(row.data.len(), self.data.cols, "Row length must match tableau width");
        let range = r * self.data.cols..(r + 1) * self.data.cols;
        self.data.data[range].copy_from_slice(&row.data);
    }

    /// Sets the z-row RHS value.
    pub fn set_z_rhs(&mut self, val: T) {
        let r = self.m;
        let c = self.rhs_col();
        self.data[(r, c)] = val;
    }

    /// Sets the z-row variable entries from a slice and the RHS in one call.
    pub fn set_z_row(&mut self, coeffs: &[T], rhs: T) {
        assert_eq!(coeffs.len(), self.num_vars(), "Coefficients length must match num_vars");
        let m = self.m;
        for (j, &v) in coeffs.iter().enumerate() {
            self.data[(m, j)] = v;
        }
        self.set_z_rhs(rhs);
    }

    /// Returns the z-row variable entries (excludes RHS) as an owned Vec.
    pub fn z_row_vars(&self) -> Vec<T> {
        let m = self.m;
        (0..self.num_vars()).map(|j| self.data[(m, j)]).collect()
    }
}

impl<T> Tableau<T>
where
    T: PartialOrd + Zero,
{
    /// Returns true if any constraint-row RHS entry is negative.
    pub fn has_negative_rhs(&self) -> bool {
        let c = self.rhs_col();
        (0..self.m).any(|i| self.data[(i, c)] < T::zero())
    }
}

impl<T> Index<(usize, usize)> for Tableau<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<(usize, usize)> for Tableau<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index]
    }
}
