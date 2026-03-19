use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use num_rational::Rational64;

pub mod linalg;
pub mod model;
pub mod solvers;

use crate::model::{Problem, Goal, Relation};
use crate::solvers::{
    BlandSimplexSolver, CyclingProneSolver, InitSource, ShadowVertexSimplexSolver,
    SimplexSolver, Solution, SolveStats, Status, Step, Solver,
};

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

fn rational_to_f64(r: Rational64) -> f64 {
    *r.numer() as f64 / *r.denom() as f64
}

fn status_to_str(s: Status) -> &'static str {
    match s {
        Status::InProgress => "in_progress",
        Status::Optimal => "optimal",
        Status::Infeasible => "infeasible",
        Status::Unbounded => "unbounded",
        Status::Cycling => "cycling",
    }
}

#[pyclass]
pub struct PyProblem {
    pub(crate) inner: Problem<Rational64>,
}

impl PyProblem {
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
    #[pyo3(get)]
    pub is_degenerate: bool,
    #[pyo3(get)]
    pub entering_var: Option<usize>,
    #[pyo3(get)]
    pub leaving_var: Option<usize>,
}

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
#[derive(Clone)]
pub struct PySolveStats {
    #[pyo3(get)]
    pub total_pivots: usize,
    #[pyo3(get)]
    pub degenerate_pivots: usize,
    #[pyo3(get)]
    pub path_length: usize,
    #[pyo3(get)]
    pub cycling_detected: bool,
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

// ---------------------------------------------------------------------------
// Simplex solver (Dantzig rule, with cycling detection)
// ---------------------------------------------------------------------------

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
        let step = self.inner.step();
        Ok(step_to_py(step))
    }

    pub fn last_step(&self) -> Option<PyStep> {
        self.inner
            .last_step()
            .map(|s: &Step<Rational64>| step_to_py(s.clone()))
    }

    pub fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    pub fn solve(&mut self, problem: &PyProblem) -> PyResult<PySolution> {
        self.initialized = true;
        run_solve(&mut self.inner, InitSource::Problem(problem.inner().clone()))
    }

    pub fn solve_with_history(&mut self, problem: &PyProblem) -> PyResult<(PySolution, Vec<PyStep>, PySolveStats)> {
        self.initialized = true;
        run_solve_with_history(&mut self.inner, InitSource::Problem(problem.inner().clone()))
    }
}

// ---------------------------------------------------------------------------
// Bland's rule simplex solver
// ---------------------------------------------------------------------------

#[pyclass]
pub struct PyBlandSimplexSolver {
    inner: BlandSimplexSolver<Rational64>,
    initialized: bool,
}

#[pymethods]
impl PyBlandSimplexSolver {
    #[new]
    pub fn new() -> Self {
        PyBlandSimplexSolver {
            inner: BlandSimplexSolver::new(),
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
        let step = self.inner.step();
        Ok(step_to_py(step))
    }

    pub fn last_step(&self) -> Option<PyStep> {
        self.inner
            .last_step()
            .map(|s: &Step<Rational64>| step_to_py(s.clone()))
    }

    pub fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    pub fn solve(&mut self, problem: &PyProblem) -> PyResult<PySolution> {
        self.initialized = true;
        run_solve(&mut self.inner, InitSource::Problem(problem.inner().clone()))
    }

    pub fn solve_with_history(&mut self, problem: &PyProblem) -> PyResult<(PySolution, Vec<PyStep>, PySolveStats)> {
        self.initialized = true;
        run_solve_with_history(&mut self.inner, InitSource::Problem(problem.inner().clone()))
    }
}

// ---------------------------------------------------------------------------
// Cycling-prone simplex solver (largest-index entering + largest-basis leaving)
// ---------------------------------------------------------------------------

#[pyclass]
pub struct PyCyclingProneSolver {
    inner: CyclingProneSolver<Rational64>,
    initialized: bool,
}

#[pymethods]
impl PyCyclingProneSolver {
    #[new]
    pub fn new() -> Self {
        PyCyclingProneSolver {
            inner: CyclingProneSolver::new(),
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
        let step = self.inner.step();
        Ok(step_to_py(step))
    }

    pub fn last_step(&self) -> Option<PyStep> {
        self.inner
            .last_step()
            .map(|s: &Step<Rational64>| step_to_py(s.clone()))
    }

    pub fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    pub fn solve(&mut self, problem: &PyProblem) -> PyResult<PySolution> {
        self.initialized = true;
        run_solve(&mut self.inner, InitSource::Problem(problem.inner().clone()))
    }

    pub fn solve_with_history(&mut self, problem: &PyProblem) -> PyResult<(PySolution, Vec<PyStep>, PySolveStats)> {
        self.initialized = true;
        run_solve_with_history(&mut self.inner, InitSource::Problem(problem.inner().clone()))
    }
}

// ---------------------------------------------------------------------------
// Shadow vertex simplex solver
// ---------------------------------------------------------------------------

#[pyclass]
pub struct PyShadowVertexSimplexSolver {
    inner: ShadowVertexSimplexSolver<Rational64>,
    initialized: bool,
}

#[pymethods]
impl PyShadowVertexSimplexSolver {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: ShadowVertexSimplexSolver::new(),
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

    pub fn solve(&mut self, problem: &PyProblem) -> PyResult<PySolution> {
        self.initialized = true;
        run_solve(&mut self.inner, InitSource::Problem(problem.inner().clone()))
    }

    pub fn solve_with_history(&mut self, problem: &PyProblem) -> PyResult<(PySolution, Vec<PyStep>, PySolveStats)> {
        self.initialized = true;
        run_solve_with_history(&mut self.inner, InitSource::Problem(problem.inner().clone()))
    }

    pub fn set_auxiliary_objective(
        &mut self,
        d_coeffs: &Bound<'_, PyList>,
        d_slack: &Bound<'_, PyList>,
        d_rhs: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.inner.set_auxiliary_objective(
            to_rational_vec(d_coeffs)?,
            to_rational_vec(d_slack)?,
            py_to_rational(d_rhs)?,
        );
        Ok(())
    }

    pub fn solve_with_shadow_history(
        &mut self,
        problem: &PyProblem,
    ) -> PyResult<(PySolution, Vec<PyStep>, Vec<(f64, f64)>, PySolveStats)> {
        self.initialized = true;
        let result = self
            .inner
            .solve_with_shadow_history(InitSource::Problem(problem.inner().clone()))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

        let mut stats = SolveStats::default();
        let history_steps: Vec<PyStep> = result
            .history
            .iter()
            .map(|s: &Step<Rational64>| {
                stats.total_pivots = s.iteration;
                if s.is_degenerate {
                    stats.degenerate_pivots += 1;
                }
                step_to_py(s.clone())
            })
            .collect();
        stats.path_length = history_steps.len();
        stats.cycling_detected = result.solution.status == Status::Cycling;
        if let Some(last) = result.history.last() {
            stats.total_pivots = last.iteration;
        }

        let solution = solution_to_py(result.solution);
        let shadow_points: Vec<(f64, f64)> = result
            .shadow_points
            .iter()
            .map(|(d, c)| (rational_to_f64(*d), rational_to_f64(*c)))
            .collect();
        Ok((solution, history_steps, shadow_points, stats_to_py(&stats)))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn step_to_py(s: Step<Rational64>) -> PyStep {
    PyStep {
        iteration: s.iteration,
        primal: s.primal.iter().copied().map(rational_to_f64).collect(),
        objective_value: rational_to_f64(s.objective_value),
        status: status_to_str(s.status).to_string(),
        is_degenerate: s.is_degenerate,
        entering_var: s.entering_var,
        leaving_var: s.leaving_var,
    }
}

fn solution_to_py(s: Solution<Rational64>) -> PySolution {
    PySolution {
        x: s.x.iter().copied().map(rational_to_f64).collect(),
        objective: rational_to_f64(s.objective),
        status: status_to_str(s.status).to_string(),
    }
}

fn stats_to_py(s: &SolveStats) -> PySolveStats {
    PySolveStats {
        total_pivots: s.total_pivots,
        degenerate_pivots: s.degenerate_pivots,
        path_length: s.path_length,
        cycling_detected: s.cycling_detected,
    }
}

fn run_solve<S>(solver: &mut S, source: InitSource<Rational64>) -> PyResult<PySolution>
where
    S: Solver<Rational64, Error = String>,
{
    solver.init(source);
    solver.find_initial_bfs().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    let last = loop {
        let s = solver.step();
        if solver.is_done() {
            break s;
        }
    };
    let sol = match last.status {
        Status::Optimal | Status::Cycling => Solution { x: last.primal, objective: last.objective_value, status: last.status },
        Status::Infeasible | Status::Unbounded => Solution { x: vec![], objective: Rational64::default(), status: last.status },
        Status::InProgress => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Solver stopped prematurely")),
    };
    Ok(solution_to_py(sol))
}

fn run_solve_with_history<S>(solver: &mut S, source: InitSource<Rational64>) -> PyResult<(PySolution, Vec<PyStep>, PySolveStats)>
where
    S: Solver<Rational64, Error = String>,
{
    solver.init(source);
    solver.find_initial_bfs().map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    let initial = solver.current_step();
    let mut prev_primal = initial.primal.clone();
    let mut history = vec![step_to_py(initial)];

    let mut stats = SolveStats::default();

    let mut last;
    loop {
        last = solver.step();
        stats.total_pivots += 1;
        if last.is_degenerate {
            stats.degenerate_pivots += 1;
        }
        if solver.is_done() {
            break;
        }
        if last.primal != prev_primal {
            prev_primal = last.primal.clone();
            history.push(step_to_py(last.clone()));
        }
    }

    stats.path_length = history.len();
    stats.cycling_detected = last.status == Status::Cycling;

    let sol = match last.status {
        Status::Optimal | Status::Cycling => {
            Solution { x: last.primal, objective: last.objective_value, status: last.status }
        }
        Status::Infeasible | Status::Unbounded => Solution { x: vec![], objective: Rational64::default(), status: last.status },
        Status::InProgress => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Solver stopped prematurely")),
    };
    Ok((solution_to_py(sol), history, stats_to_py(&stats)))
}

#[pymodule]
fn linprog_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProblem>()?;
    m.add_class::<PyTableau>()?;
    m.add_class::<PyStep>()?;
    m.add_class::<PySolution>()?;
    m.add_class::<PySolveStats>()?;
    m.add_class::<PySimplexSolver>()?;
    m.add_class::<PyBlandSimplexSolver>()?;
    m.add_class::<PyCyclingProneSolver>()?;
    m.add_class::<PyShadowVertexSimplexSolver>()?;
    Ok(())
}
