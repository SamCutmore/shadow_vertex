use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div};
use crate::model::Tableau;
use num_traits::{One, Zero};

/// Pivot selection outcome: Optimal, Unbounded, or Pivot(row, col).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PivotResult {
    Optimal,
    Unbounded,
    Pivot(usize, usize),
}

impl<T> Tableau<T>
where
    T: Zero + PartialOrd + Clone + Copy + Div<Output = T>,
{
    /// Z-row entries (column index, value) for variable columns only (excludes RHS).
    fn z_row_entries(&self) -> impl Iterator<Item = (usize, T)> + '_ {
        let m = self.m;
        (0..self.num_vars()).map(move |j| (j, self.data[(m, j)]))
    }

    /// Pivot column by Dantzig rule (most negative reduced cost).
    pub fn find_pivot_col_most_negative(&self) -> Option<usize> {
        let mut best_col = None;
        let mut min_val = T::zero();
        for (j, val) in self.z_row_entries() {
            if val < min_val {
                min_val = val;
                best_col = Some(j);
            }
        }
        best_col
    }

    /// Pivot column by Bland rule (first negative reduced cost).
    pub fn find_pivot_col_bland(&self) -> Option<usize> {
        self.z_row_entries()
            .find(|(_, val)| *val < T::zero())
            .map(|(j, _)| j)
    }

    /// Minimum-ratio test: returns leaving row for the given entering column, or None.
    /// Ties are broken by smallest row index.
    pub fn ratio_test(&self, col: usize) -> Option<usize> {
        let mut best_row = None;
        let mut min_ratio: Option<T> = None;
        let rhs_col = self.rhs_col();

        for i in 0..self.m {
            let entry = self.data[(i, col)];
            if entry > T::zero() {
                let ratio = self.data[(i, rhs_col)] / entry;
                if min_ratio.is_none() || ratio < min_ratio.unwrap() {
                    min_ratio = Some(ratio);
                    best_row = Some(i);
                }
            }
        }
        best_row
    }

    /// Minimum-ratio test with smallest-basis-variable tie-breaking: among
    /// rows that achieve the minimum ratio, the row whose basis variable has
    /// the smallest index is chosen.
    pub fn ratio_test_smallest_basis(&self, col: usize) -> Option<usize> {
        let mut best_row = None;
        let mut min_ratio: Option<T> = None;
        let mut best_basis_var: Option<usize> = None;
        let rhs_col = self.rhs_col();

        for i in 0..self.m {
            let entry = self.data[(i, col)];
            if entry > T::zero() {
                let ratio = self.data[(i, rhs_col)] / entry;
                let update = if min_ratio.is_none() {
                    true
                } else if ratio < min_ratio.unwrap() {
                    true
                } else if ratio == min_ratio.unwrap() {
                    self.basis[i] < best_basis_var.unwrap()
                } else {
                    false
                };
                if update {
                    min_ratio = Some(ratio);
                    best_row = Some(i);
                    best_basis_var = Some(self.basis[i]);
                }
            }
        }
        best_row
    }

    /// Chooses pivot (Dantzig column, ratio test row); returns Optimal, Unbounded, or Pivot(row, col).
    pub fn find_pivot_indices(&self) -> PivotResult {
        match self.find_pivot_col_most_negative() {
            None => PivotResult::Optimal,
            Some(col) => match self.ratio_test(col) {
                Some(row) => PivotResult::Pivot(row, col),
                None => PivotResult::Unbounded,
            },
        }
    }

    /// Pivot column by largest-index rule: last variable with negative
    /// reduced cost.
    pub fn find_pivot_col_largest_index(&self) -> Option<usize> {
        let mut best_col = None;
        for (j, val) in self.z_row_entries() {
            if val < T::zero() {
                best_col = Some(j);
            }
        }
        best_col
    }

    /// Largest-index entering + smallest-basis-variable leaving:
    /// a cycling-prone combination.
    pub fn find_pivot_indices_cycling_prone(&self) -> PivotResult {
        match self.find_pivot_col_largest_index() {
            None => PivotResult::Optimal,
            Some(col) => match self.ratio_test_smallest_basis(col) {
                Some(row) => PivotResult::Pivot(row, col),
                None => PivotResult::Unbounded,
            },
        }
    }

    /// Same as find_pivot_indices but uses Bland's rule to avoid cycling.
    pub fn find_pivot_indices_bland(&self) -> PivotResult {
        match self.find_pivot_col_bland() {
            None => PivotResult::Optimal,
            Some(col) => match self.ratio_test(col) {
                Some(row) => PivotResult::Pivot(row, col),
                None => PivotResult::Unbounded,
            },
        }
    }

    /// Current BFS as a vector of length n_vars (non-basic vars = 0, basic = RHS).
    pub fn current_vertex(&self, n_vars: usize) -> Vec<T>
    where
        T: Zero + Clone,
    {
        let mut vertex = vec![T::zero(); n_vars];
        let rhs_col = self.rhs_col();
        for (row, &var_idx) in self.basis.iter().enumerate() {
            if var_idx < n_vars {
                vertex[var_idx] = self.data[(row, rhs_col)].clone();
            }
        }
        vertex
    }

    /// Returns true when no reduced cost is negative.
    pub fn is_optimal(&self) -> bool {
        self.find_pivot_col_most_negative().is_none()
    }

    /// Reduced costs: `r_j = w_j - w_B^T * col_j` for each variable column.
    pub fn reduced_costs(&self, w: &[T]) -> Vec<T>
    where
        T: Zero + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
    {
        (0..self.num_vars()).map(|j| {
            let dot: T = self.basis.iter().enumerate()
                .map(|(i, &bi)| w[bi] * self[(i, j)])
                .fold(T::zero(), |a, b| a + b);
            w[j] - dot
        }).collect()
    }

    /// Computes `sum(w[basis[i]] * rhs(i))` -- the dot product of an objective
    /// vector with the current basic variable values.
    pub fn eval_at_basis(&self, w: &[T]) -> T
    where
        T: Zero + Copy + Add<Output = T> + Mul<Output = T>,
    {
        let rhs_col = self.rhs_col();
        self.basis.iter().enumerate()
            .map(|(i, &bi)| w[bi] * self.data[(i, rhs_col)])
            .fold(T::zero(), |a, b| a + b)
    }
}

impl<T> Tableau<T>
where
    T: Zero
        + One
        + PartialOrd
        + Clone
        + Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + SubAssign
        + MulAssign,
{
    /// Performs a pivot at (row_idx, col_idx); updates basis and all rows including z-row.
    pub fn pivot(&mut self, row_idx: usize, col_idx: usize) {
        let pivot_element = self.data[(row_idx, col_idx)];
        let inv_pivot = T::one() / pivot_element;

        {
            let mut p_row = self.data.row_mut(row_idx);
            p_row *= inv_pivot;
        }
        let norm = self.data.row(row_idx);

        for i in 0..=self.m {
            if i != row_idx {
                let factor = self.data[(i, col_idx)];
                let mut current = self.data.row_mut(i);
                current.sub_assign_scaled(&norm, factor);
            }
        }

        self.basis[row_idx] = col_idx;
    }
}
