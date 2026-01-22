pub trait Solver<T> {
    type Error;

    fn init(&mut self, problem: crate::model::Problem<T>);
    fn step(&mut self) -> Step<T>;
    fn is_done(&self) -> bool;

    fn solve(&mut self, problem: crate::model::Problem<T>) -> Result<Solution<T>, Self::Error> {
        self.init(problem);
        let mut last_step = self.step();
        
        while !self.is_done() {
            last_step = self.step();
        }

        match last_step.status {
            Status::Optimal => Ok(Solution {
                x: last_step.vertex,
                objective: last_step.objective_value,
                status: Status::Optimal,
            }),
            Status::Infeasible => Ok(Solution { status: Status::Infeasible, ..Default::default() }),
            Status::Unbounded => Ok(Solution { status: Status::Unbounded, ..Default::default() }),
            Status::InProgress => Err(self.handle_error("Solver stopped prematurely")),
        }
    }
    
    fn handle_error(&self, msg: &str) -> Self::Error;
}

pub struct Step<T> {
    pub iteration: usize,
    pub vertex: Vec<T>,
    pub objective_value: T,
    pub status: Status,
}

#[derive(Default)]
pub struct Solution<T> {
    pub x: Vec<T>,
    pub objective: T,
    pub status: Status,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    InProgress,
    Optimal,
    Infeasible,
    Unbounded,
}