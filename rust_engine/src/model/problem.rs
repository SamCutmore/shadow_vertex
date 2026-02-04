use super::Goal;
use crate::model::{StandardForm, Tableau};
use crate::linalg::Matrix;
use std::ops::Neg;
use num_traits::{One, Zero};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Relation {
    LessEqual,
    GreaterEqual,
    Equal,
}

#[derive(Debug, Clone)]
pub struct Constraint<T> {
    pub coefficients: Vec<T>,
    pub relation: Relation,
    pub rhs: T,
}

impl<T> Constraint<T>
where
    T: Clone + Copy + Default + PartialOrd + std::ops::Neg<Output = T>,
{
    pub fn normalise(mut self) -> Self {
        if self.rhs < T::default() {
            self.coefficients.iter_mut().for_each(|v| *v = -*v);
            self.rhs = -self.rhs;
            self.relation = match self.relation {
                Relation::LessEqual => Relation::GreaterEqual,
                Relation::GreaterEqual => Relation::LessEqual,
                Relation::Equal => Relation::Equal,
            };
        }
        self
    }
}

#[derive(Debug, Clone)]
pub struct Problem<T> {
    pub constraints: Vec<Constraint<T>>,
    pub objective: Vec<T>,
    pub goal: Goal,
}

impl<T> Problem<T> {
    pub fn new(objective: Vec<T>, goal: Goal) -> Self {
        Self {
            objective,
            goal,
            constraints: Vec::new(),
        }
    }

    pub fn add_constraint(&mut self, coefficients: Vec<T>, relation: Relation, rhs: T) {
        self.constraints.push(Constraint {
            coefficients,
            relation,
            rhs,
        });
    }
}

impl<T> Problem<T>
where
    T: Clone + Copy + Default + PartialOrd + One + Zero + Neg<Output = T>,
{
    pub fn to_tableau(&self) -> Tableau<T> {
        self.clone().into_tableau_form()
    }

    pub fn into_standard_form(self) -> StandardForm<T> {
        let one = T::one();
        let neg_one = -T::one();
        let zero = T::zero();

        let surplus_slack = self.constraints.iter().filter(|c| c.relation != Relation::Equal).count();
        let total_cols = self.objective.len() + surplus_slack;

        let mut a_matrix: Matrix<T> = Matrix::with_capacity(self.constraints.len(), total_cols);
        let mut b_vec: Vec<T> = Vec::new();
        let mut slack_indices: Vec<usize> = Vec::with_capacity(surplus_slack);
        let mut slack_index = self.objective.len();

        for constraint in self.constraints {
            let normalised = constraint.normalise();
            let mut row_data: Vec<T> = Vec::with_capacity(total_cols);
            row_data.extend(normalised.coefficients);
            row_data.resize(total_cols, zero.clone());

            match normalised.relation {
                Relation::LessEqual => {
                    row_data[slack_index] = one.clone();
                    slack_indices.push(slack_index);
                },
                Relation::GreaterEqual => {
                    row_data[slack_index] = neg_one.clone();
                    slack_indices.push(slack_index);
                },
                Relation::Equal => {}
            }

            if normalised.relation != Relation::Equal {
                slack_index += 1;
            }

            b_vec.push(normalised.rhs);
            a_matrix.push_row(&row_data);
        }

        let mut c_vec = vec![zero.clone(); total_cols];

        for (i, val) in self.objective.into_iter().enumerate() {
            c_vec[i] = if self.goal == Goal::Max {
                -val
            } else {
                val
            };
        }

        StandardForm {
            a: a_matrix,
            b: b_vec,
            c: c_vec,
            goal: self.goal,
            slack_indices,
        }
    }

    pub fn into_tableau_form(self) -> Tableau<T> {
        let one = T::one();
        let neg_one = -T::one();
        let zero = T::zero();

        let m = self.constraints.len();
        let n = self.objective.len();

        let mut a_matrix = Matrix::with_capacity(m, n);
        let mut s_matrix = Matrix::with_capacity(m, m);
        let mut b_vec = Vec::with_capacity(m);
        
        let mut basis = Vec::with_capacity(m);
        let nonbasis: Vec<usize> = (0..n).collect();

        let mut z_coeffs = Vec::with_capacity(n);
        for val in self.objective {
            z_coeffs.push(if self.goal == Goal::Max {
                -val
            } else {
                val
            });
        }
        let z_slack = vec![zero.clone(); m];
        let z_rhs = zero.clone();

        for (i, constraint) in self.constraints.into_iter().enumerate() {
            let normalised = constraint.normalise();
            
            a_matrix.push_row(&normalised.coefficients);
            b_vec.push(normalised.rhs);

            let mut slack_row = vec![zero.clone(); m];
            match normalised.relation {
                Relation::LessEqual => {
                    slack_row[i] = one.clone();
                    basis.push(n + i);
                },
                Relation::GreaterEqual => {
                    slack_row[i] = neg_one.clone();
                    basis.push(n + i);
                },
                Relation::Equal => {
                    basis.push(n + i); 
                }
            }
            s_matrix.push_row(&slack_row);
        }

        Tableau {
            coefficients: a_matrix,
            slack: s_matrix,
            rhs: b_vec,
            basis,
            nonbasis,
            z_coeffs,
            z_slack,
            z_rhs,
        }
    }
}