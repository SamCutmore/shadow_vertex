use crate::model::tableau_form::Tableau;
use crate::model::PivotResult;
use crate::solvers::{InitSource, Solver, Step, Status};
use num_traits::{Signed, Zero, FromPrimitive};
use std::ops::{AddAssign, Div, MulAssign, SubAssign};

/// Simplex solver using Bland's rule (smallest-index pivot) to avoid cycling.
pub struct BlandSimplexSolver<T> {
    tableau: Option<Tableau<T>>,
    iteration: usize,
    n_vars: usize,
    done: bool,
    last_step: Option<Step<T>>,
    prev_primal: Option<Vec<T>>,
}

impl<T> BlandSimplexSolver<T>
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
        }
    }
}

impl<T> Default for BlandSimplexSolver<T>
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

impl<T> Solver<T> for BlandSimplexSolver<T>
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

        let (status, entering, leaving) = match tab.find_pivot_indices_bland() {
            PivotResult::Pivot(row, col) => {
                let leaving_var = tab.basis[row];
                tab.pivot(row, col);
                self.iteration += 1;
                (Status::InProgress, Some(col), Some(leaving_var))
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
