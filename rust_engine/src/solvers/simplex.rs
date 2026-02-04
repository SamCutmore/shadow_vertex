use crate::model::tableau_form::Tableau;
use crate::model::PivotResult;
use crate::solvers::{InitSource, Solver, Step, Status};
use num_traits::{Signed, Zero, FromPrimitive};
use std::ops::{AddAssign, Div, MulAssign, SubAssign};

/// Simplex solver state: tableau, iteration count, and last step.
pub struct SimplexSolver<T> {
    tableau: Option<Tableau<T>>,
    iteration: usize,
    n_vars: usize,
    done: bool,
    last_step: Option<Step<T>>,
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
    /// Creates a new simplex solver.
    pub fn new() -> Self {
        Self {
            tableau: None,
            iteration: 0,
            n_vars: 0,
            done: false,
            last_step: None,
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
    }

    /// Checks initial tableau for negative RHS; returns Err if infeasible.
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
        let tab = self.tableau.as_mut().expect("Not initialized");

        let status = match tab.find_pivot_indices() {
            PivotResult::Pivot(row, col) => {
                tab.pivot(row, col);
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
