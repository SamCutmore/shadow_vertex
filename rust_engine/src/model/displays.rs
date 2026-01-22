use std::fmt;
use num_rational::Rational64;
use num_traits::{Zero, Signed}; 

use crate::model::{Goal};
use crate::model::problem::{Problem, Relation};
use crate::model::tableau_form::Tableau;

fn format_rational(r: Rational64) -> String {
    if *r.denom() == 1 {
        format!("{}", r.numer())
    } else {
        format!("{}/{}", r.numer(), r.denom())
    }
}

impl fmt::Display for Problem<Rational64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let goal_str = match self.goal {
            Goal::Max => "Max",
            Goal::Min => "Min",
        };

        write!(f, "{} Z = ", goal_str)?;
        writeln!(f, "{}", format_expression(&self.objective))?;

        writeln!(f, "\nSubject to:")?;
        for c in &self.constraints {
            let rel = match c.relation {
                Relation::LessEqual => "<=",
                Relation::GreaterEqual => ">=",
                Relation::Equal => "=",
            };
            writeln!(f, "  {} {} {}", format_expression(&c.coefficients), rel, format_rational(c.rhs))?;
        }
        let vars: Vec<String> = (0..self.objective.len()).map(|i| format!("x{}", i)).collect();
        writeln!(f, "  where  {}, ... >= 0", vars.join(", "))?;
        Ok(())
    }
}

fn format_expression(coeffs: &[Rational64]) -> String {
    let mut parts = Vec::new();
    for (i, &coeff) in coeffs.iter().enumerate() {
        if coeff.is_zero() { continue; }
        
        let abs_c = coeff.abs();
        let term = if abs_c.is_integer() && *abs_c.numer() == 1 {
            format!("x{}", i)
        } else {
            format!("{}x{}", format_rational(abs_c), i)
        };

        if parts.is_empty() {
            parts.push(if coeff.is_negative() { format!("-{}", term) } else { term });
        } else {
            parts.push(format!(" {} {}", if coeff.is_negative() { "-" } else { "+" }, term));
        }
    }
    parts.concat()
}

impl fmt::Display for Tableau<Rational64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.coefficients.cols;
        let m = self.slack.cols;
        let rule_len = 10 + (n * 9) + (m * 9) + 10;

        writeln!(f, "\nTableau (Basis: {:?})", self.basis)?;
        write!(f, "{:>6} | ", "Basis")?;
        for j in 0..n { write!(f, "x{:<7} ", j)?; }
        write!(f, "| ")?;
        for j in 0..m { write!(f, "s{:<7} ", j)?; }
        writeln!(f, "| {:>8}", "RHS")?;
        writeln!(f, "{}", "-".repeat(rule_len))?;

        for i in 0..self.rows() {
            let label = if self.basis[i] < n { format!("x{}", self.basis[i]) } 
                        else { format!("s{}", self.basis[i] - n) };
            write!(f, "{:>6} | ", label)?;
            for j in 0..n { write!(f, "{:>8} ", format_rational(self.coefficients[(i, j)]))?; }
            write!(f, "| ")?;
            for j in 0..m { write!(f, "{:>8} ", format_rational(self.slack[(i, j)]))?; }
            writeln!(f, "| {:>8}", format_rational(self.rhs[i]))?;
        }

        writeln!(f, "{}", "-".repeat(rule_len))?;
        write!(f, "{:>6} | ", "Z")?;
        for j in 0..n { write!(f, "{:>8} ", format_rational(self.z_coeffs[j]))?; }
        write!(f, "| ")?;
        for j in 0..m { write!(f, "{:>8} ", format_rational(self.z_slack[j]))?; }
        writeln!(f, "| {:>8}", format_rational(self.z_rhs))
    }
}