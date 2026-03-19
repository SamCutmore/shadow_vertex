use std::collections::HashSet;

use crate::model::tableau_form::Tableau;
use crate::model::PivotResult;
use crate::solvers::{InitSource, Solver, Step, Status};
use num_traits::{Signed, Zero, FromPrimitive};
use std::ops::{AddAssign, Div, MulAssign, SubAssign};

/// Simplex solver (Dantzig pivot rule) with cycling detection.
pub struct SimplexSolver<T> {
    tableau: Option<Tableau<T>>,
    iteration: usize,
    n_vars: usize,
    done: bool,
    last_step: Option<Step<T>>,
    prev_primal: Option<Vec<T>>,
    seen_bases: HashSet<Vec<usize>>,
}

impl<T> SimplexSolver<T>
where
    T: Zero
        + Signed
        + Clone
        + Copy
        + FromPrimitive
        + AddAssign
        + SubAssign
        + MulAssign
        + Div<Output = T>
        + PartialOrd,
{
    pub fn new() -> Self {
        Self {
            tableau: None,
            iteration: 0,
            n_vars: 0,
            done: false,
            last_step: None,
            prev_primal: None,
            seen_bases: HashSet::new(),
        }
    }
}

impl<T> Default for SimplexSolver<T>
where
    T: Zero
        + Signed
        + Clone
        + Copy
        + FromPrimitive
        + AddAssign
        + SubAssign
        + MulAssign
        + Div<Output = T>
        + PartialOrd,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Solver<T> for SimplexSolver<T>
where
    T: Zero
        + Signed
        + Clone
        + Copy
        + FromPrimitive
        + AddAssign
        + SubAssign
        + MulAssign
        + Div<Output = T>
        + PartialOrd
        + Default,
{
    type Error = String;

    fn init(&mut self, source: InitSource<T>) {
        let (n_vars, tableau) = source.into_tableau_and_n_vars();
        self.n_vars = n_vars;
        self.tableau = Some(tableau);
        self.iteration = 0;
        self.done = false;
        self.last_step = None;
        self.prev_primal = None;
        self.seen_bases = HashSet::new();
    }

    fn find_initial_bfs(&mut self) -> Result<bool, Self::Error> {
        if self
            .tableau
            .as_ref()
            .map_or(false, |t| t.has_negative_rhs())
        {
            return Err("Infeasible: initial tableau has negative RHS".to_string());
        }
        let tab = self.tableau.as_ref().unwrap();
        self.seen_bases.insert(tab.basis.clone());
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
            is_degenerate: false,
            entering_var: None,
            leaving_var: None,
        }
    }

    fn step(&mut self) -> Step<T> {
        let tab = self.tableau.as_mut().unwrap();

        let (status, entering, leaving) = match tab.find_pivot_indices() {
            PivotResult::Pivot(row, col) => {
                let leaving_var = tab.basis[row];
                tab.pivot(row, col);
                self.iteration += 1;

                if self.seen_bases.contains(&tab.basis) {
                    self.done = true;
                    (Status::Cycling, Some(col), Some(leaving_var))
                } else {
                    self.seen_bases.insert(tab.basis.clone());
                    (Status::InProgress, Some(col), Some(leaving_var))
                }
            }
            PivotResult::Optimal => {
                self.done = true;
                (Status::Optimal, None, None)
            }
            PivotResult::Unbounded => {
                self.done = true;
                (Status::Unbounded, None, None)
            }
        };

        let tab = self.tableau.as_ref().unwrap();
        let primal = tab.current_vertex(self.n_vars);
        let is_degenerate = self
            .prev_primal
            .as_ref()
            .map_or(false, |prev| *prev == primal);
        self.prev_primal = Some(primal.clone());

        let step = Step {
            iteration: self.iteration,
            primal,
            objective_value: tab.z_rhs(),
            status,
            is_degenerate,
            entering_var: entering,
            leaving_var: leaving,
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
