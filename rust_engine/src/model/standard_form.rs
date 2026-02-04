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

        let mut coefficients = Matrix::with_capacity(m, n);
        let mut slack = Matrix::with_capacity(m, m);
        for r in 0..m {
            let coeff_row: Vec<T> = (0..n).map(|c| self.a[(r, c)].clone()).collect();
            coefficients.push_row(&coeff_row);
            let slack_row: Vec<T> = (0..m).map(|c| self.a[(r, n + c)].clone()).collect();
            slack.push_row(&slack_row);
        }
        let rhs = self.b;
        let z_coeffs = self.c[0..n].to_vec();
        let z_slack = self.c[n..n + m].to_vec();
        let z_rhs = T::zero();
        Tableau::new(coefficients, slack, rhs, z_coeffs, z_slack, z_rhs)
    }
}
