use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use num_rational::Rational64;

pub mod linalg;
pub mod model;
pub mod solvers;

use crate::model::{Problem, Goal, Relation};
use crate::solvers::{
    InitSource, TwoPhaseSimplexSolver, SimplexSolver, Solution, Status, Step, Solver,
};

/// Converts a Python value to Rational64 (int, float, or (num, den) tuple).
fn py_to_rational(value: &Bound<'_, PyAny>) -> PyResult<Rational64> {
    if let Ok((n, d)) = value.extract::<(i64, i64)>() {
        if d == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Rational denominator must not be zero",
            ));
        }
        return Ok(Rational64::new(n, d));
    }
    if let Ok(i) = value.extract::<i64>() {
        return Ok(Rational64::from_integer(i));
    }
    if let Ok(f) = value.extract::<f64>() {
        const SCALE: f64 = 1e12;
        let n = (f * SCALE).round() as i64;
        return Ok(Rational64::new(n, SCALE as i64));
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Expected int, float, or (numerator, denominator) tuple",
    ))
}

fn to_rational_vec(list: &Bound<'_, PyList>) -> PyResult<Vec<Rational64>> {
    list.iter()
        .map(|item| py_to_rational(&item))
        .collect()
}

/// Converts Rational64 to f64 for Python-facing attributes.
fn rational_to_f64(r: Rational64) -> f64 {
    *r.numer() as f64 / *r.denom() as f64
}

fn status_to_str(s: Status) -> &'static str {
    match s {
        Status::InProgress => "in_progress",
        Status::Optimal => "optimal",
        Status::Infeasible => "infeasible",
        Status::Unbounded => "unbounded",
    }
}

#[pyclass]
pub struct PyProblem {
    pub(crate) inner: Problem<Rational64>,
}

impl PyProblem {
    /// Returns the underlying problem for solver init.
    pub fn inner(&self) -> &Problem<Rational64> {
        &self.inner
    }
}

#[pymethods]
impl PyProblem {
    #[new]
    #[pyo3(signature = (objective, goal="max"))]
    pub fn new(objective: &Bound<'_, PyList>, goal: &str) -> PyResult<Self> {
        let g = match goal.to_lowercase().as_str() {
            "min" => Goal::Min,
            _ => Goal::Max,
        };
        Ok(PyProblem {
            inner: Problem::new(to_rational_vec(objective)?, g),
        })
    }

    pub fn add_constraint(
        &mut self,
        coeffs: &Bound<'_, PyList>,
        rel: &str,
        rhs: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let r = match rel {
            "<=" | "leq" => Relation::LessEqual,
            ">=" | "geq" => Relation::GreaterEqual,
            "=" | "==" | "eq" => Relation::Equal,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown relation '{}'; use '<=', '>=', or '='",
                    rel
                )));
            }
        };
        self.inner
            .add_constraint(to_rational_vec(coeffs)?, r, py_to_rational(rhs)?);
        Ok(())
    }

    pub fn to_tableau(&self) -> PyTableau {
        PyTableau {
            inner: self.inner.clone().into_tableau_form(),
        }
    }

    pub fn __str__(&self) -> String {
        format!("{}", self.inner) 
    }

    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

/// One solver step: primal point, objective value, and status.
#[pyclass]
pub struct PyStep {
    #[pyo3(get)]
    pub iteration: usize,
    #[pyo3(get)]
    pub primal: Vec<f64>,
    #[pyo3(get)]
    pub objective_value: f64,
    #[pyo3(get)]
    pub status: String,
}

/// Final solution: primal, objective, and status.
#[pyclass]
pub struct PySolution {
    #[pyo3(get)]
    pub x: Vec<f64>,
    #[pyo3(get)]
    pub objective: f64,
    #[pyo3(get)]
    pub status: String,
}

#[pyclass]
pub struct PyTableau {
    pub inner: crate::model::Tableau<Rational64>,
}

#[pymethods]
impl PyTableau {
    pub fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    pub fn __repr__(&self) -> String {
        self.__str__()
    }

    pub fn num_rows(&self) -> usize { self.inner.rows() }
    pub fn num_cols(&self) -> usize { self.inner.cols() }
}

/// Simplex solver. init(problem) then step(), or solve() / solve_with_history() to run to completion.
#[pyclass]
pub struct PySimplexSolver {
    inner: SimplexSolver<Rational64>,
    initialized: bool,
}

#[pymethods]
impl PySimplexSolver {
    #[new]
    pub fn new() -> Self {
        PySimplexSolver {
            inner: SimplexSolver::new(),
            initialized: false,
        }
    }

    /// Loads the problem; then call find_initial_bfs() and step(), or solve() / solve_with_history().
    pub fn init(&mut self, problem: &PyProblem) -> PyResult<()> {
        self.inner
            .init(InitSource::Problem(problem.inner().clone()));
        self.initialized = true;
        Ok(())
    }

    /// Ensures a feasible basis; returns Err if infeasible.
    pub fn find_initial_bfs(&mut self) -> PyResult<()> {
        self.inner
            .find_initial_bfs()
            .map(|_| ())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }

    /// Performs one iteration and returns the resulting step.
    pub fn step(&mut self) -> PyResult<PyStep> {
        if !self.initialized {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Solver not initialized; call init(problem) first",
            ));
        }
        let step = self.inner.step();
        Ok(step_to_py(step))
    }

    /// Returns the last step produced, or None.
    pub fn last_step(&self) -> Option<PyStep> {
        self.inner
            .last_step()
            .map(|s: &Step<Rational64>| step_to_py(s.clone()))
    }

    /// Returns true when the solver has reached a terminal status.
    pub fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    /// Runs to completion and returns the final solution.
    pub fn solve(&mut self, problem: &PyProblem) -> PyResult<PySolution> {
        self.initialized = true;
        run_solve(&mut self.inner, InitSource::Problem(problem.inner().clone()))
    }

    /// Run to completion and return (solution, list of steps visited).
    pub fn solve_with_history(&mut self, problem: &PyProblem) -> PyResult<(PySolution, Vec<PyStep>)> {
        self.initialized = true;
        run_solve_with_history(&mut self.inner, InitSource::Problem(problem.inner().clone()))
    }
}

/// Two-phase (dual / shadow vertex) simplex solver.
#[pyclass]
pub struct PyTwoPhaseSimplexSolver {
    inner: TwoPhaseSimplexSolver<Rational64>,
    initialized: bool,
}

#[pymethods]
impl PyTwoPhaseSimplexSolver {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: TwoPhaseSimplexSolver::new(),
            initialized: false,
        }
    }

    pub fn init(&mut self, problem: &PyProblem) -> PyResult<()> {
        self.inner
            .init(InitSource::Problem(problem.inner().clone()));
        self.initialized = true;
        Ok(())
    }

    pub fn find_initial_bfs(&mut self) -> PyResult<()> {
        self.inner
            .find_initial_bfs()
            .map(|_| ())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }

    pub fn step(&mut self) -> PyResult<PyStep> {
        if !self.initialized {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Solver not initialized; call init(problem) first",
            ));
        }
        Ok(step_to_py(self.inner.step()))
    }

    pub fn last_step(&self) -> Option<PyStep> {
        self.inner
            .last_step()
            .map(|s: &Step<Rational64>| step_to_py(s.clone()))
    }

    pub fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    /// Runs to completion and returns the final solution.
    pub fn solve(&mut self, problem: &PyProblem) -> PyResult<PySolution> {
        self.initialized = true;
        run_solve(&mut self.inner, InitSource::Problem(problem.inner().clone()))
    }

    /// Run to completion and return (solution, list of steps visited).
    pub fn solve_with_history(&mut self, problem: &PyProblem) -> PyResult<(PySolution, Vec<PyStep>)> {
        self.initialized = true;
        run_solve_with_history(&mut self.inner, InitSource::Problem(problem.inner().clone()))
    }
}

fn step_to_py(s: Step<Rational64>) -> PyStep {
    PyStep {
        iteration: s.iteration,
        primal: s.primal.iter().copied().map(rational_to_f64).collect(),
        objective_value: rational_to_f64(s.objective_value),
        status: status_to_str(s.status).to_string(),
    }
}

fn solution_to_py(s: Solution<Rational64>) -> PySolution {
    PySolution {
        x: s.x.iter().copied().map(rational_to_f64).collect(),
        objective: rational_to_f64(s.objective),
        status: status_to_str(s.status).to_string(),
    }
}

fn run_solve<S>(solver: &mut S, source: InitSource<Rational64>) -> PyResult<PySolution>
where
    S: Solver<Rational64, Error = String>,
{
    solver.init(source);
    solver.find_initial_bfs().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let mut last = solver.step();
    while !solver.is_done() {
        last = solver.step();
    }
    let sol = match last.status {
        Status::Optimal => Solution { x: last.primal, objective: last.objective_value, status: last.status },
        Status::Infeasible | Status::Unbounded => Solution { x: vec![], objective: Rational64::default(), status: last.status },
        Status::InProgress => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Solver stopped prematurely")),
    };
    Ok(solution_to_py(sol))
}

fn run_solve_with_history<S>(solver: &mut S, source: InitSource<Rational64>) -> PyResult<(PySolution, Vec<PyStep>)>
where
    S: Solver<Rational64, Error = String>,
{
    solver.init(source);
    solver.find_initial_bfs().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let mut history = Vec::new();
    let mut last = solver.step();
    while !solver.is_done() {
        history.push(step_to_py(last.clone()));
        last = solver.step();
    }
    history.push(step_to_py(last.clone()));
    let sol = match last.status {
        Status::Optimal => Solution { x: last.primal, objective: last.objective_value, status: last.status },
        Status::Infeasible | Status::Unbounded => Solution { x: vec![], objective: Rational64::default(), status: last.status },
        Status::InProgress => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Solver stopped prematurely")),
    };
    Ok((solution_to_py(sol), history))
}

#[pymodule]
fn linprog_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProblem>()?;
    m.add_class::<PyTableau>()?;
    m.add_class::<PyStep>()?;
    m.add_class::<PySolution>()?;
    m.add_class::<PySimplexSolver>()?;
    m.add_class::<PyTwoPhaseSimplexSolver>()?;
    Ok(())
}
