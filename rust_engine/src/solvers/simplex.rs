use crate::model::Problem;
use crate::model::tableau_form::Tableau;
use crate::solvers::{Solver, Step, Status};
use num_traits::{Signed, Zero, FromPrimitive};
use std::ops::{AddAssign, SubAssign, MulAssign, Div};

pub struct SimplexSolver<T> {
    tableau: Option<Tableau<T>>,
    iteration: usize,
    n_vars: usize,
    done: bool,
}

impl<T> SimplexSolver<T> 
where 
    T: Zero + Signed + Clone + Copy + FromPrimitive + AddAssign + SubAssign + MulAssign + Div<Output = T> + PartialOrd
{
    pub fn new() -> Self {
        Self { tableau: None, iteration: 0, n_vars: 0, done: false }
    }
}

impl<T> Solver<T> for SimplexSolver<T>
where 
    T: Zero + Signed + Clone + Copy + FromPrimitive + AddAssign + SubAssign + MulAssign + Div<Output = T> + PartialOrd
{
    type Error = String;

    fn init(&mut self, problem: Problem<T>) {
        self.n_vars = problem.objective.len();
        self.tableau = Some(problem.to_tableau());
        self.iteration = 0;
        self.done = false;
    }

    fn is_done(&self) -> bool {
        self.done
    }

    fn step(&mut self) -> Step<T> {
        let tab = self.tableau.as_mut().expect("Not initialized");
        
        let status = match tab.find_pivot_indices() {
            Some((row, col)) => {
                tab.pivot(row, col);
                self.iteration += 1;
                Status::InProgress
            },
            None => {
                self.done = true;
                Status::Optimal
            }
        };

        Step {
            iteration: self.iteration,
            vertex: tab.current_vertex(self.n_vars),
            objective_value: tab.z_rhs.clone(),
            status,
        }
    }

    fn handle_error(&self, msg: &str) -> Self::Error {
        msg.to_string()
    }
}