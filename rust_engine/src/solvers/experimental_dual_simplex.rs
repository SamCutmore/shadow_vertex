use crate::model::tableau_form::Tableau;
use crate::model::PivotResult;
use crate::solvers::{InitSource, Solver, Step, Status};
use num_traits::{Signed, Zero, FromPrimitive};
use std::ops::{AddAssign, Div, Mul, MulAssign, Neg, SubAssign};

enum Phase {
    OptimizeD,
    OptimizeC,
}

enum PivotOutcome {
    Pivoted,
    Optimal,
    Unbounded,
}

/// tableau, phase (d then c), and stored objective c.
pub struct TwoPhaseSimplexSolver<T> {
    tableau: Option<Tableau<T>>,
    n_vars: usize,
    iteration: usize,
    done: bool,
    last_step: Option<Step<T>>,
    phase: Phase,
    c_coeffs: Vec<T>,
    c_slack: Vec<T>,
    c_rhs: T,
}

impl<T> TwoPhaseSimplexSolver<T>
where
    T: Zero
        + Signed
        + Clone
        + Copy
        + FromPrimitive
        + AddAssign
        + SubAssign
        + Mul<Output = T>
        + MulAssign
        + Div<Output = T>
        + Neg<Output = T>
        + PartialOrd,
{
    pub fn new() -> Self {
        Self {
            tableau: None,
            n_vars: 0,
            iteration: 0,
            done: false,
            last_step: None,
            phase: Phase::OptimizeD,
            c_coeffs: Vec::new(),
            c_slack: Vec::new(),
            c_rhs: T::zero(),
        }
    }

    /// Sets z-row to d = -c for the first phase.
    fn set_z_to_d(&mut self) {
        let tab = self.tableau.as_mut().expect("tableau");
        for i in 0..tab.z_coeffs.len() {
            tab.z_coeffs[i] = -self.c_coeffs[i];
        }
        for i in 0..tab.z_slack.len() {
            tab.z_slack[i] = -self.c_slack[i];
        }
        tab.z_rhs = -self.c_rhs;
    }

    /// Restores z-row to the reduced-cost row for c at the current basis, and z_rhs to (c'x + c_rhs) at the BFS.
    /// Initializes z_rhs to c_rhs so the constant part of the objective is preserved; uses row-based sub_assign_scaled.
    fn set_z_to_c(&mut self) {
        let tab = self.tableau.as_mut().expect("tableau");
        let n = self.c_coeffs.len();
        let m = tab.rows();
        let c_b: Vec<T> = tab
            .basis
            .iter()
            .map(|&var_idx| {
                if var_idx < n {
                    self.c_coeffs[var_idx]
                } else {
                    self.c_slack[var_idx - n]
                }
            })
            .collect();

        tab.z_coeffs.clone_from(&self.c_coeffs);
        tab.z_slack.clone_from(&self.c_slack);
        tab.z_rhs = self.c_rhs;

        let constraint_rows: Vec<_> = (0..m).map(|i| tab.row(i)).collect();
        for (i, row_i) in constraint_rows.iter().enumerate() {
            tab.z_row_mut().sub_assign_scaled(row_i, c_b[i]);
        }
        tab.z_rhs = -tab.z_rhs;
        tab.z_rhs += self.c_rhs;
        tab.z_rhs += self.c_rhs;
    }

    /// Tries one pivot using Bland's rule to avoid cycling; returns Pivoted, Optimal, or Unbounded.
    fn try_pivot_step(&mut self) -> PivotOutcome {
        let tab = self.tableau.as_mut().expect("Not initialized");
        match tab.find_pivot_indices_bland() {
            PivotResult::Pivot(row, col) => {
                tab.pivot(row, col);
                self.iteration += 1;
                PivotOutcome::Pivoted
            }
            PivotResult::Optimal => PivotOutcome::Optimal,
            PivotResult::Unbounded => PivotOutcome::Unbounded,
        }
    }
}

impl<T> Default for TwoPhaseSimplexSolver<T>
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
        + Neg<Output = T>
        + PartialOrd,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Solver<T> for TwoPhaseSimplexSolver<T>
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
        + Neg<Output = T>
        + PartialOrd
        + Default,
{
    type Error = String;

    fn init(&mut self, source: InitSource<T>) {
        let (n_vars, tab) = source.into_tableau_and_n_vars();
        self.n_vars = n_vars;
        self.c_coeffs = tab.z_coeffs.clone();
        self.c_slack = tab.z_slack.clone();
        self.c_rhs = tab.z_rhs;
        self.tableau = Some(tab);
        self.iteration = 0;
        self.done = false;
        self.last_step = None;
        self.phase = Phase::OptimizeD;
        self.set_z_to_d();
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

    /// Performs one step of the path (d-phase or c-phase).
    fn step(&mut self) -> Step<T> {
        let status = match self.phase {
            Phase::OptimizeD => match self.try_pivot_step() {
                PivotOutcome::Pivoted => Status::InProgress,
                PivotOutcome::Optimal | PivotOutcome::Unbounded => {
                    self.set_z_to_c();
                    self.phase = Phase::OptimizeC;
                    Status::InProgress
                }
            },
            Phase::OptimizeC => match self.try_pivot_step() {
                PivotOutcome::Pivoted => Status::InProgress,
                PivotOutcome::Optimal => {
                    self.done = true;
                    Status::Optimal
                }
                PivotOutcome::Unbounded => {
                    self.done = true;
                    Status::Unbounded
                }
            },
        };

        let tab = self.tableau.as_mut().expect("Not initialized");
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
