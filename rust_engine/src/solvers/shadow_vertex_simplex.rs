use crate::model::tableau_form::Tableau;
use crate::model::PivotResult;
use crate::solvers::{InitSource, Solution, Solver, Step, Status};
use num_traits::{One, Signed, Zero};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign};

/// Types that support a numerically safe "strictly positive" check for the shadow pivot.
/// For exact types (e.g. `Rational64`) this is `self > 0`; for floats use a small epsilon
/// so that `denom <= 0` is not triggered by rounding error.
pub trait EpsilonThreshold: Zero + PartialOrd + Copy {
    fn is_strictly_positive(self) -> bool;
}

impl EpsilonThreshold for num_rational::Rational64 {
    #[inline]
    fn is_strictly_positive(self) -> bool {
        self > num_rational::Rational64::zero()
    }
}

impl EpsilonThreshold for f64 {
    #[inline]
    fn is_strictly_positive(self) -> bool {
        self > f64::EPSILON
    }
}

impl EpsilonThreshold for f32 {
    #[inline]
    fn is_strictly_positive(self) -> bool {
        self > f32::EPSILON
    }
}

#[derive(Clone, Debug)]
pub struct ShadowSolveResult<T> {
    pub solution: Solution<T>,
    pub history: Vec<Step<T>>,
    pub shadow_points: Vec<(T, T)>,
}

pub struct ShadowVertexSimplexSolver<T> {
    tableau: Option<Tableau<T>>,
    n_vars: usize,
    iteration: usize,
    done: bool,
    last_step: Option<Step<T>>,
    /// Auxiliary objective (coefficients for structural variables).
    d_coeffs: Vec<T>,
    /// Auxiliary objective (coefficients for slack variables).
    d_slack: Vec<T>,
    d_rhs: T,
    /// True objective (stored to restore z-row and for reduced cost computation).
    c_coeffs: Vec<T>,
    c_slack: Vec<T>,
    c_rhs: T,
}

impl<T> ShadowVertexSimplexSolver<T>
where
    T: Zero
        + One
        + Clone
        + Copy
        + PartialOrd
        + Signed
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + SubAssign
        + MulAssign
        + EpsilonThreshold,
{
    /// Creates a new shadow vertex simplex solver with auxiliary objective `d = 0`.
    pub fn new() -> Self {
        Self {
            tableau: None,
            n_vars: 0,
            iteration: 0,
            done: false,
            last_step: None,
            d_coeffs: Vec::new(),
            d_slack: Vec::new(),
            d_rhs: T::zero(),
            c_coeffs: Vec::new(),
            c_slack: Vec::new(),
            c_rhs: T::zero(),
        }
    }

    /// Sets the auxiliary objective `d`. The solver must already be initialized;
    /// call after `init()` and before `find_initial_bfs()` / `step()`.
    /// Lengths must match: `d_coeffs.len() == n structural`, `d_slack.len() == m`.
    pub fn set_auxiliary_objective(
        &mut self,
        d_coeffs: Vec<T>,
        d_slack: Vec<T>,
        d_rhs: T,
    ) {
        self.d_coeffs = d_coeffs;
        self.d_slack = d_slack;
        self.d_rhs = d_rhs;
    }

    /// Returns (d'x, c'x) at the current vertex for plotting the shadow polygon.
    fn current_shadow_point(&self) -> (T, T) {
        let tab = self.tableau.as_ref().expect("Not initialized");
        let n = tab.coefficients.cols;
        let mut d_val = self.d_rhs;
        for (i, &var_idx) in tab.basis.iter().enumerate() {
            let coef = if var_idx < n {
                self.d_coeffs.get(var_idx).copied().unwrap_or(T::zero())
            } else {
                self.d_slack.get(var_idx - n).copied().unwrap_or(T::zero())
            };
            d_val = d_val + coef * tab.rhs[i];
        }
        let c_val = tab.z_rhs.clone();
        (d_val, c_val)
    }

    /// Solves from the given source and returns the solution, full step history,
    /// and points in the (d, c) shadow plane for plotting the shadow polygon.
    pub fn solve_with_shadow_history(
        &mut self,
        source: InitSource<T>,
    ) -> Result<ShadowSolveResult<T>, String>
    where
        T: Default,
    {
        self.init(source);
        self.find_initial_bfs()?;
        let mut shadow_points = vec![self.current_shadow_point()];
        let mut last_step = self.step();
        let mut history = Vec::new();
        while !self.is_done() {
            history.push(last_step.clone());
            shadow_points.push(self.current_shadow_point());
            last_step = self.step();
        }
        history.push(last_step.clone());
        shadow_points.push(self.current_shadow_point());

        let solution = match last_step.status {
            Status::Optimal => Solution {
                x: last_step.primal,
                objective: last_step.objective_value,
                status: Status::Optimal,
            },
            Status::Infeasible => Solution {
                x: vec![],
                objective: T::default(),
                status: Status::Infeasible,
            },
            Status::Unbounded => Solution {
                x: vec![],
                objective: T::default(),
                status: Status::Unbounded,
            },
            Status::InProgress => return Err(self.handle_error("Solver stopped prematurely")),
        };

        Ok(ShadowSolveResult {
            solution,
            history,
            shadow_points,
        })
    }

    fn reduced_costs(
        tableau: &Tableau<T>,
        n: usize,
        w_coeffs: &[T],
        w_slack: &[T],
    ) -> Vec<T>
    where
        T: Zero + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
    {
        let m = tableau.rows();
        let num_cols = n + m;
        let mut r = Vec::with_capacity(num_cols);

        for j in 0..num_cols {
            let w_j = if j < n {
                w_coeffs.get(j).copied().unwrap_or(T::zero())
            } else {
                w_slack.get(j - n).copied().unwrap_or(T::zero())
            };

            let mut dot = T::zero();
            for (i, &var_idx) in tableau.basis.iter().enumerate() {
                let w_bi = if var_idx < n {
                    w_coeffs.get(var_idx).copied().unwrap_or(T::zero())
                } else {
                    w_slack.get(var_idx - n).copied().unwrap_or(T::zero())
                };
                dot = dot + w_bi * tableau[(i, j)];
            }
            r.push(w_j - dot);
        }
        r
    }

    /// Shadow vertex pivot rule: choose entering column using the parametric objective
    /// r(λ) = (1-λ)r_d + λ r_c. Find the smallest λ in (0, 1] at which some r_j(λ) becomes negative.
    ///
    /// To stay on the shadow (edge of the projection), we require:
    /// - r_d_j ≥ 0: current basis must be optimal for the auxiliary objective d; if r_d_j < 0
    ///   we have already passed the breakpoint for that variable.
    /// - r_c_j < 0: only consider variables that improve the true objective c.
    /// - denom = r_d_j - r_c_j strictly positive (using EpsilonThreshold for numeric safety with floats).
    /// λ_j = r_d_j / (r_d_j - r_c_j). Choose j with minimum λ_j in (0, 1].
    fn find_shadow_pivot_col(
        _tableau: &Tableau<T>,
        _n: usize,
        r_d: &[T],
        r_c: &[T],
    ) -> Option<usize> {
        let one = T::one();
        let mut best_col = None;
        let mut best_lambda: Option<T> = None;

        for j in 0..r_d.len() {
            let r_d_j = r_d[j];
            let r_c_j = r_c[j];

            // Feasibility for d: basis must be optimal for the auxiliary objective.
            if r_d_j < T::zero() {
                continue;
            }
            // Must improve the true objective eventually.
            if r_c_j >= T::zero() {
                continue;
            }
            let denom = r_d_j - r_c_j;
            if !denom.is_strictly_positive() {
                continue;
            }
            let lambda_j = r_d_j / denom;
            // Only consider λ in (0, 1]: first breakpoint toward c.
            if lambda_j <= T::zero() || lambda_j > one {
                continue;
            }

            if best_lambda.is_none() || lambda_j < best_lambda.unwrap() {
                best_lambda = Some(lambda_j);
                best_col = Some(j);
            }
        }

        // If no parametric candidate, fall back to standard rule: any j with r_c_j < 0
        if best_col.is_some() {
            return best_col;
        }
        for (j, &r_c_j) in r_c.iter().enumerate() {
            if r_c_j < T::zero() {
                return Some(j);
            }
        }
        None
    }

    fn try_pivot_step(&self) -> PivotResult {
        let tab = self.tableau.as_ref().expect("Not initialized");
        let n = tab.coefficients.cols;

        let r_d = Self::reduced_costs(tab, n, &self.d_coeffs, &self.d_slack);
        let r_c: Vec<T> = tab
            .z_coeffs
            .iter()
            .copied()
            .chain(tab.z_slack.iter().copied())
            .collect();

        let col = match Self::find_shadow_pivot_col(tab, n, &r_d, &r_c) {
            Some(c) => c,
            None => return PivotResult::Optimal,
        };

        match tab.ratio_test(col) {
            Some(row) => PivotResult::Pivot(row, col),
            None => PivotResult::Unbounded,
        }
    }
}

impl<T> Default for ShadowVertexSimplexSolver<T>
where
    T: Zero
        + One
        + Clone
        + Copy
        + PartialOrd
        + Signed
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + SubAssign
        + MulAssign
        + EpsilonThreshold,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Solver<T> for ShadowVertexSimplexSolver<T>
where
    T: Zero
        + One
        + Clone
        + Copy
        + PartialOrd
        + Signed
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + SubAssign
        + MulAssign
        + EpsilonThreshold
        + Default,
{
    type Error = String;

    fn init(&mut self, source: InitSource<T>) {
        let (n_vars, tableau) = source.into_tableau_and_n_vars();
        self.n_vars = n_vars;
        self.c_coeffs = tableau.z_coeffs.clone();
        self.c_slack = tableau.z_slack.clone();
        self.c_rhs = tableau.z_rhs;
        if self.d_coeffs.is_empty() {
            self.d_coeffs = vec![T::zero(); tableau.z_coeffs.len()];

            // let n = tableau.z_coeffs.len();
            // self.d_coeffs = vec![T::zero(); n];
            // if let Some(last) = self.d_coeffs.last_mut() {
            //     *last = T::one();
            // }

            self.d_slack = vec![T::zero(); tableau.z_slack.len()];
            self.d_rhs = T::zero();
        }
        self.tableau = Some(tableau);
        self.iteration = 0;
        self.done = false;
        self.last_step = None;
    }

    fn find_initial_bfs(&mut self) -> Result<bool, Self::Error> {
        if self
            .tableau
            .as_ref()
            .map_or(false, |t| t.has_negative_rhs())
        {
            return Err("Infeasible: initial tableau has negative RHS".to_string());
        }
        Ok(true)
    }

    fn is_done(&self) -> bool {
        self.done
    }

    fn step(&mut self) -> Step<T> {
        let status = match self.try_pivot_step() {
            PivotResult::Pivot(row, col) => {
                self.tableau.as_mut().expect("Not initialized").pivot(row, col);
                self.iteration += 1;
                Status::InProgress
            }
            PivotResult::Optimal => {
                self.done = true;
                Status::Optimal
            }
            PivotResult::Unbounded => {
                self.done = true;
                Status::Unbounded
            }
        };

        let tab = self.tableau.as_ref().expect("Not initialized");
        let step = Step {
            iteration: self.iteration,
            primal: tab.current_vertex(self.n_vars),
            objective_value: tab.z_rhs.clone(),
            status,
        };
        self.last_step = Some(step.clone());
        step
    }

    fn last_step(&self) -> Option<&Step<T>> {
        self.last_step.as_ref()
    }

    fn handle_error(&self, msg: &str) -> Self::Error {
        msg.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Goal, Problem, Relation};
    use num_rational::Rational64;

    fn rational(n: i64, d: i64) -> Rational64 {
        Rational64::new(n, d)
    }

    #[test]
    fn shadow_vertex_solves_simple_lp() {
        let mut prob = Problem::new(vec![rational(3, 1), rational(2, 1)], Goal::Max);
        prob.add_constraint(vec![rational(1, 1), rational(1, 1)], Relation::LessEqual, rational(4, 1));
        prob.add_constraint(vec![rational(2, 1), rational(1, 1)], Relation::LessEqual, rational(5, 1));

        let mut solver = ShadowVertexSimplexSolver::new();
        let sol = solver
            .solve(InitSource::Problem(prob))
            .expect("solve");
        assert_eq!(sol.status, Status::Optimal);
        assert_eq!(sol.x.len(), 2);
        assert_eq!(sol.objective, rational(9, 1));
    }

    #[test]
    fn shadow_vertex_with_d_zero_matches_standard_behavior() {
        let mut prob = Problem::new(vec![rational(1, 1), rational(1, 1)], Goal::Min);
        prob.add_constraint(vec![rational(1, 1), rational(0, 1)], Relation::LessEqual, rational(5, 1));
        prob.add_constraint(vec![rational(0, 1), rational(1, 1)], Relation::LessEqual, rational(5, 1));

        let mut solver = ShadowVertexSimplexSolver::new();
        let sol = solver
            .solve(InitSource::Problem(prob.clone()))
            .expect("solve");
        assert_eq!(sol.status, Status::Optimal);
        assert_eq!(sol.objective, rational(0, 1)); // min x+y, x,y>=0, x<=5, y<=5 -> 0 at (0,0)
    }
}
