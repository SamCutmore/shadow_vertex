use pyo3::prelude::*;
use pyo3::types::PyList;
use num_rational::Rational64;

pub mod linalg;
pub mod model;

use crate::model::{Problem, Goal, Relation};

fn to_rational_vec(list: &Bound<'_, PyList>) -> Vec<Rational64> {
    list.iter()
        .map(|item| {
            if let Ok((n, d)) = item.extract::<(i64, i64)>() {
                Rational64::new(n, d)
            } 
            else {
                Rational64::new(item.extract::<i64>().unwrap_or(0), 1)
            }
        })
        .collect()
}

#[pyclass]
pub struct PyProblem {
    inner: Problem<Rational64>,
}

#[pymethods]
impl PyProblem {
    #[new]
    #[pyo3(signature = (objective, goal="max"))]
    pub fn new(objective: &Bound<'_, PyList>, goal: &str) -> Self {
        let g = match goal.to_lowercase().as_str() {
            "min" => Goal::Min,
            _ => Goal::Max,
        };
        PyProblem {
            inner: Problem::new(to_rational_vec(objective), g),
        }
    }

    pub fn add_constraint(&mut self, coeffs: &Bound<'_, PyList>, rel: &str, rhs: Bound<'_, PyAny>) {
        let r = match rel {
            "<=" => Relation::LessEqual,
            ">=" => Relation::GreaterEqual,
            _ => Relation::Equal,
        };

        let rhs_rational = if let Ok((n, d)) = rhs.extract::<(i64, i64)>() {
            Rational64::new(n, d)
        } else {
            Rational64::new(rhs.extract::<i64>().unwrap_or(0), 1)
        };

        self.inner.add_constraint(to_rational_vec(coeffs), r, rhs_rational);
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

#[pymodule]
fn linprog_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProblem>()?;
    m.add_class::<PyTableau>()?;
    Ok(())
}