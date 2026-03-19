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
    /// Auxiliary objective coefficients (length n+m, structural then slack).
    d: Vec<T>,
    d_rhs: T,
    /// True objective coefficients (stored for z-row restoration and reduced costs).
    c: Vec<T>,
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
    pub fn new() -> Self {
        Self {
            tableau: None,
            n_vars: 0,
            iteration: 0,
            done: false,
            last_step: None,
            d: Vec::new(),
            d_rhs: T::zero(),
            c: Vec::new(),
            c_rhs: T::zero(),
        }
    }

    /// Sets the auxiliary objective `d`. Call after `init()` and before `find_initial_bfs()`.
    /// `d_coeffs` has length n (structural), `d_slack` has length m.
    /// They are concatenated into a single vector internally.
    pub fn set_auxiliary_objective(
        &mut self,
        d_coeffs: Vec<T>,
        d_slack: Vec<T>,
        d_rhs: T,
    ) {
        self.d = d_coeffs.into_iter().chain(d_slack).collect();
        self.d_rhs = d_rhs;
    }

    /// Returns (d'x, c'x) at the current vertex for plotting the shadow polygon.
    fn current_shadow_point(&self) -> (T, T) {
        let tab = self.tableau.as_ref().unwrap();
        (self.d_rhs + tab.eval_at_basis(&self.d), tab.z_rhs())
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

        let initial = self.current_step();
        let mut prev_primal = initial.primal.clone();
        let mut history = vec![initial];
        let mut shadow_points = vec![self.current_shadow_point()];

        let mut last_step;
        loop {
            last_step = self.step();
            if self.is_done() {
                break;
            }
            if last_step.primal != prev_primal {
                prev_primal = last_step.primal.clone();
                history.push(last_step.clone());
                shadow_points.push(self.current_shadow_point());
            }
        }

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

    /// Shadow vertex pivot rule: parametric objective w(lambda) = (1-lambda)d + lambda*c.
    ///
    /// `r_d[j]` = bar_d_j  (standard reduced cost for d)
    /// `r_c[j]` = -bar_c_j (z-row entry for c; negative means c-improving)
    ///
    /// The parametric reduced cost is:
    ///   bar_w_j(lambda) = (1-lambda) bar_d_j + lambda bar_c_j
    ///
    /// A variable j becomes a pivot candidate when bar_w_j crosses from
    /// <= 0 to > 0 as lambda increases.
    fn find_shadow_pivot_col(r_d: &[T], r_c: &[T]) -> Option<usize> {
        let mut best_col = None;
        let mut best_lambda: Option<T> = None;
        let mut must_enter_col: Option<usize> = None;
        let mut must_enter_rc: Option<T> = None;

        for j in 0..r_d.len() {
            if r_c[j] >= T::zero() {
                continue;
            }

            let denom = r_d[j] + r_c[j];

            if (-denom).is_strictly_positive() {
                let lambda_j = r_d[j] / denom;

                if best_lambda.is_none() || lambda_j < best_lambda.unwrap() {
                    best_lambda = Some(lambda_j);
                    best_col = Some(j);
                }
            } else if r_d[j].is_strictly_positive() {
                if must_enter_rc.is_none() || r_c[j] < must_enter_rc.unwrap() {
                    must_enter_rc = Some(r_c[j]);
                    must_enter_col = Some(j);
                }
            }
        }

        if must_enter_col.is_some() {
            return must_enter_col;
        }
        best_col
    }

    fn try_pivot_step(&self) -> PivotResult {
        let tab = self.tableau.as_ref().unwrap();

        let r_d = tab.reduced_costs(&self.d);
        let r_c = tab.z_row_vars();

        let col = match Self::find_shadow_pivot_col(&r_d, &r_c) {
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

        self.c = tableau.z_row_vars();
        self.c_rhs = tableau.z_rhs();

        if self.d.is_empty() {
            self.d = vec![T::zero(); tableau.num_vars()];
            self.d_rhs = T::zero();
        }
        self.tableau = Some(tableau);
        self.iteration = 0;
        self.done = false;
        self.last_step = None;
    }

    fn find_initial_bfs(&mut self) -> Result<bool, Self::Error> {
        if self.tableau.as_ref().map_or(false, |t| t.has_negative_rhs()) {
            return Err("Infeasible: initial tableau has negative RHS".to_string());
        }

        // Phase I: install -d as z-row and pivot to a d-optimal BFS.
        let neg_d: Vec<T> = self.d.iter().map(|&x| -x).collect();
        self.tableau.as_mut().unwrap().set_z_row(&neg_d, T::zero());

        let max_phase1_iters = 50_000;
        for _ in 0..max_phase1_iters {
            match self.tableau.as_ref().unwrap().find_pivot_indices() {
                PivotResult::Optimal => break,
                PivotResult::Unbounded => {
                    return Err("Unbounded auxiliary objective d in Phase I".into());
                }
                PivotResult::Pivot(row, col) => {
                    self.tableau.as_mut().unwrap().pivot(row, col);
                }
            }
        }

        // Restore the true c z-row for the current basis.
        let tab = self.tableau.as_mut().unwrap();
        let r_c = tab.reduced_costs(&self.c);
        let z_rhs = self.c_rhs - tab.eval_at_basis(&self.c);
        tab.set_z_row(&r_c, z_rhs);

        Ok(true)
    }

    fn is_done(&self) -> bool {
        self.done
    }

    fn current_step(&self) -> Step<T> {
        let tab = self.tableau.as_ref().unwrap();
        Step {
            iteration: self.iteration,
            primal: tab.current_vertex(self.n_vars),
            objective_value: tab.z_rhs(),
            status: if self.done { Status::Optimal } else { Status::InProgress },
        }
    }

    fn step(&mut self) -> Step<T> {
        let status = match self.try_pivot_step() {
            PivotResult::Pivot(row, col) => {
                self.tableau.as_mut().unwrap().pivot(row, col);
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

        let tab = self.tableau.as_ref().unwrap();
        let step = Step {
            iteration: self.iteration,
            primal: tab.current_vertex(self.n_vars),
            objective_value: tab.z_rhs(),
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
        assert_eq!(sol.objective, rational(0, 1));
    }

    #[test]
    fn shadow_vertex_3d_cube_traces_boundary() {
        let mut prob = Problem::new(
            vec![rational(1, 1), rational(1, 1), rational(1, 1)],
            Goal::Max,
        );
        for i in 0..3 {
            let mut row = vec![rational(0, 1); 3];
            row[i] = rational(1, 1);
            prob.add_constraint(row.clone(), Relation::LessEqual, rational(1, 1));
            prob.add_constraint(row, Relation::GreaterEqual, rational(0, 1));
        }

        let mut solver = ShadowVertexSimplexSolver::new();
        solver.set_auxiliary_objective(
            vec![rational(0, 1), rational(0, 1), rational(-1, 1)],
            vec![rational(0, 1); 6],
            rational(0, 1),
        );
        let result = solver
            .solve_with_shadow_history(InitSource::Problem(prob))
            .expect("solve");

        assert_eq!(result.solution.status, Status::Optimal);
        assert_eq!(result.solution.objective, rational(3, 1));

        assert_eq!(result.history[0].primal, vec![rational(0, 1); 3]);
        let last = &result.history.last().unwrap().primal;
        assert_eq!(*last, vec![rational(1, 1); 3]);
    }

    #[test]
    fn shadow_vertex_random_d_traces_boundary() {
        let mut prob = Problem::new(
            vec![rational(1, 1), rational(1, 1), rational(1, 1)],
            Goal::Max,
        );
        for i in 0..3 {
            let mut row = vec![rational(0, 1); 3];
            row[i] = rational(1, 1);
            prob.add_constraint(row.clone(), Relation::LessEqual, rational(1, 1));
            prob.add_constraint(row, Relation::GreaterEqual, rational(0, 1));
        }

        let mut solver = ShadowVertexSimplexSolver::new();
        solver.set_auxiliary_objective(
            vec![rational(3, 1), rational(-2, 1), rational(1, 1)],
            vec![rational(0, 1); 6],
            rational(0, 1),
        );
        let result = solver
            .solve_with_shadow_history(InitSource::Problem(prob))
            .expect("solve");

        assert_eq!(result.solution.status, Status::Optimal);
        assert_eq!(result.solution.objective, rational(3, 1));

        let (d0, _c0) = &result.shadow_points[0];
        assert_eq!(*d0, rational(4, 1), "Phase I should start at d-optimal (d'x=4)");

        for w in result.shadow_points.windows(2) {
            assert!(
                w[0].0 >= w[1].0,
                "d-values should be non-increasing along the path: {:?} -> {:?}",
                w[0], w[1]
            );
        }
    }

    #[test]
    fn shadow_vertex_klee_minty_avoids_exponential_path() {
        let c = vec![rational(4, 1), rational(2, 1), rational(1, 1)];
        let mut prob = Problem::new(c, Goal::Max);
        prob.add_constraint(
            vec![rational(1, 1), rational(0, 1), rational(0, 1)],
            Relation::LessEqual,
            rational(1, 1),
        );
        prob.add_constraint(
            vec![rational(2, 1), rational(1, 1), rational(0, 1)],
            Relation::LessEqual,
            rational(5, 1),
        );
        prob.add_constraint(
            vec![rational(4, 1), rational(2, 1), rational(1, 1)],
            Relation::LessEqual,
            rational(25, 1),
        );
        for i in 0..3 {
            let mut row = vec![rational(0, 1); 3];
            row[i] = rational(1, 1);
            prob.add_constraint(row, Relation::GreaterEqual, rational(0, 1));
        }

        let mut solver = ShadowVertexSimplexSolver::new();
        solver.set_auxiliary_objective(
            vec![rational(0, 1), rational(0, 1), rational(-1, 1)],
            vec![rational(0, 1); 6],
            rational(0, 1),
        );
        let result = solver
            .solve_with_shadow_history(InitSource::Problem(prob))
            .expect("solve");

        assert_eq!(result.solution.status, Status::Optimal);
        assert_eq!(result.solution.objective, rational(25, 1));
        assert!(
            result.history.len() <= 5,
            "Expected <= 5 steps, got {}",
            result.history.len()
        );
    }

    #[test]
    fn shadow_vertex_perturbed_3d_cube() {
        let mut prob = Problem::new(
            vec![rational(1,1), rational(1,1), rational(1,1)],
            Goal::Max,
        );
        prob.add_constraint(
            vec![rational(1,1), rational(0,1), rational(0,1)],
            Relation::GreaterEqual, rational(0,1));
        prob.add_constraint(
            vec![rational(35,32), rational(1,32), rational(3,32)],
            Relation::LessEqual, rational(35,32));
        prob.add_constraint(
            vec![rational(0,1), rational(1,1), rational(0,1)],
            Relation::GreaterEqual, rational(0,1));
        prob.add_constraint(
            vec![rational(3,32), rational(29,32), rational(3,32)],
            Relation::LessEqual, rational(32,32));
        prob.add_constraint(
            vec![rational(0,1), rational(0,1), rational(1,1)],
            Relation::GreaterEqual, rational(0,1));
        prob.add_constraint(
            vec![rational(0,1), rational(1,32), rational(32,32)],
            Relation::LessEqual, rational(35,32));

        let mut solver = ShadowVertexSimplexSolver::new();
        solver.set_auxiliary_objective(
            vec![rational(0,1), rational(0,1), rational(-1,1)],
            vec![rational(0,1); 6],
            rational(0,1),
        );
        let result = solver
            .solve_with_shadow_history(InitSource::Problem(prob.clone()))
            .expect("solve");

        let mut std_solver = crate::solvers::SimplexSolver::new();
        std_solver.init(InitSource::Problem(prob));
        std_solver.find_initial_bfs().unwrap();
        let std_last = loop {
            let s = std_solver.step();
            if std_solver.is_done() { break s; }
        };

        assert_eq!(result.solution.status, Status::Optimal);
        assert_eq!(
            result.solution.objective, std_last.objective_value,
            "Shadow vertex should match standard simplex"
        );
    }
}
