use crate::linalg::Matrix;
use crate::model::tableau_form::Tableau;
use super::Goal;
use num_traits::Zero;

/// Standard form LP: A, b, c, goal, and slack column indices.
#[derive(Clone)]
pub struct StandardForm<T> {
    pub a: Matrix<T>,
    pub b: Vec<T>,
    pub c: Vec<T>,
    pub goal: Goal,
    pub slack_indices: Vec<usize>,
}

impl<T> StandardForm<T> {
    /// Builds standard form; panics if dimensions do not match.
    pub fn new(a: Matrix<T>, b: Vec<T>, c: Vec<T>, goal: Goal, slack_indices: Vec<usize>) -> Self {
        assert_eq!(a.rows, b.len(), "Matrix A rows must match vector b length");
        assert_eq!(a.cols, c.len(), "Matrix A columns must match vector c length");

        StandardForm {
            a,
            b,
            c,
            goal,
            slack_indices,
        }
    }

    /// Number of original (non-slack) variables.
    pub fn n_vars(&self) -> usize {
        self.a.cols - self.slack_indices.len()
    }

    /// Number of constraints (rows of A).
    pub fn n_constraints(&self) -> usize {
        self.a.rows
    }
}

impl<T> StandardForm<T>
where
    T: Clone + Default + Zero,
{
    /// Converts to tableau; requires one slack per row and slack columns as last m columns.
    pub fn into_tableau(self) -> Tableau<T> {
        let m = self.a.rows;
        let n = self.n_vars();
        assert_eq!(
            self.slack_indices.len(),
            m,
            "into_tableau requires one slack per row (no equality constraints)"
        );
        for (i, &idx) in self.slack_indices.iter().enumerate() {
            assert_eq!(idx, n + i, "slack columns must be the last m columns");
        }

        let total_cols = n + m + 1;
        let mut data = Matrix::with_capacity(m + 1, total_cols);

        for r in 0..m {
            let mut row_data = Vec::with_capacity(total_cols);
            for c in 0..n { row_data.push(self.a[(r, c)].clone()); }
            for c in 0..m { row_data.push(self.a[(r, n + c)].clone()); }
            row_data.push(self.b[r].clone());
            data.push_row(&row_data);
        }

        let mut z_row_data = Vec::with_capacity(total_cols);
        z_row_data.extend_from_slice(&self.c[0..n]);
        z_row_data.extend_from_slice(&self.c[n..n + m]);
        z_row_data.push(T::zero());
        data.push_row(&z_row_data);

        Tableau::new(data, n, m)
    }
}
