use crate::model::{Problem, StandardForm, Tableau};
use num_traits::{One, Zero};
use std::ops::Neg;

/// Input for solver init: a Problem or a StandardForm.
#[derive(Clone)]
pub enum InitSource<T> {
    Problem(Problem<T>),
    StandardForm(StandardForm<T>),
}

impl<T> InitSource<T>
where
    T: Clone + Copy + Default + PartialOrd + One + Zero + Neg<Output = T>,
{
    /// Builds tableau and number of original variables from this source.
    pub fn into_tableau_and_n_vars(self) -> (usize, Tableau<T>) {
        match self {
            InitSource::Problem(p) => {
                let n_vars = p.objective.len();
                let tableau = p.into_tableau_form();
                (n_vars, tableau)
            }
            InitSource::StandardForm(sf) => {
                let n_vars = sf.n_vars();
                let tableau = sf.into_tableau();
                (n_vars, tableau)
            }
        }
    }
}

/// One solver step: iteration index, primal point, objective value, status.
#[derive(Clone, Debug)]
pub struct Step<T> {
    pub iteration: usize,
    pub primal: Vec<T>,
    pub objective_value: T,
    pub status: Status,
}

/// Final solution: primal x, objective value, status.
#[derive(Clone, Debug)]
pub struct Solution<T> {
    pub x: Vec<T>,
    pub objective: T,
    pub status: Status,
}

/// Solver termination status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    InProgress,
    Optimal,
    Infeasible,
    Unbounded,
}

impl Default for Status {
    fn default() -> Self {
        Status::InProgress
    }
}

/// Solver trait: init, find_initial_bfs(), step(), last_step(), solve().
pub trait Solver<T> {
    type Error;

    /// Loads problem or standard form; does not run phase I or II.
    fn init(&mut self, source: InitSource<T>);

    /// Ensures a feasible basis; call after init() before step(). Returns Err if infeasible.
    fn find_initial_bfs(&mut self) -> Result<bool, Self::Error> {
        Ok(true)
    }

    /// Performs one iteration from the current basis.
    fn step(&mut self) -> Step<T>;
    fn is_done(&self) -> bool;

    /// Returns the last step produced, if any.
    fn last_step(&self) -> Option<&Step<T>> {
        None
    }

    /// Runs to completion: init, find_initial_bfs(), then step until done.
    fn solve(&mut self, source: InitSource<T>) -> Result<Solution<T>, Self::Error>
    where
        T: Default,
    {
        self.init(source);
        self.find_initial_bfs()?;
        let mut last_step = self.step();
        while !self.is_done() {
            last_step = self.step();
        }
        match last_step.status {
            Status::Optimal => Ok(Solution {
                x: last_step.primal,
                objective: last_step.objective_value,
                status: Status::Optimal,
            }),
            Status::Infeasible => Ok(Solution {
                x: vec![],
                objective: T::default(),
                status: Status::Infeasible,
            }),
            Status::Unbounded => Ok(Solution {
                x: vec![],
                objective: T::default(),
                status: Status::Unbounded,
            }),
            Status::InProgress => Err(self.handle_error("Solver stopped prematurely")),
        }
    }

    fn handle_error(&self, msg: &str) -> Self::Error;
}
