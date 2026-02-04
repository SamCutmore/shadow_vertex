pub mod problem;
pub mod standard_form;
pub mod tableau_form;
pub mod tableau_operations;
pub mod displays;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Goal {
    Min,
    Max,
}

pub use problem::{Problem, Relation, Constraint};
pub use standard_form::StandardForm;
pub use tableau_form::{Tableau, TableauRow, TableauRowMut};
pub use tableau_operations::PivotResult;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Matrix;
    use crate::model::problem::Relation;
    use num_rational::Rational64;

    fn rational(n: i64) -> Rational64 {
        Rational64::from_integer(n)
    }

    #[test]
    fn test_tableau_row_addition() {
        let mut coefficients = Matrix::new(2, 2);
        coefficients[(0, 0)] = 1; coefficients[(0, 1)] = 2;
        coefficients[(1, 0)] = 3; coefficients[(1, 1)] = 4;

        let mut slack = Matrix::new(2, 2);
        slack[(0, 0)] = 1; slack[(0, 1)] = 0;
        slack[(1, 0)] = 0; slack[(1, 1)] = 1;

        let rhs = vec![10, 20];
        let z_coeffs = vec![0, 0];
        let z_slack = vec![0, 0];
        let z_rhs = 0;

        let tableau = Tableau::new(
            coefficients,
            slack,
            rhs,
            z_coeffs,
            z_slack,
            z_rhs,
        );

        let r0 = tableau.row(0);
        let r1 = tableau.row(1);

        let r2 = r0 + r1 + 1;

        assert_eq!(r2.coefficients.data, vec![5, 7]);
        assert_eq!(r2.slack.data, vec![2, 2]);
        assert_eq!(r2.rhs, 31);
    }

    #[test]
    fn test_tableau_row_mut_addition() {
        let mut coefficients = Matrix::new(2, 2);
        coefficients[(0, 0)] = 1; coefficients[(0, 1)] = 2;
        coefficients[(1, 0)] = 3; coefficients[(1, 1)] = 4;

        let mut slack = Matrix::new(2, 2);
        slack[(0, 0)] = 1; slack[(0, 1)] = 0;
        slack[(1, 0)] = 0; slack[(1, 1)] = 1;

        let rhs = vec![10, 20];
        
        let z_coeffs = vec![0, 0];
        let z_slack = vec![0, 0];
        let z_rhs = 0;

        let mut tableau = Tableau::new(
            coefficients,
            slack,
            rhs,
            z_coeffs,
            z_slack,
            z_rhs,
        );
        
        {
            let row1 = tableau.row(1);
            let mut row0 = tableau.row_mut(0);
            row0 += &row1 + 1;
        }

        let row0 = tableau.row(0);

        assert_eq!(row0.coefficients.data, vec![5, 7]);
        assert_eq!(row0.slack.data, vec![2, 2]);
        assert_eq!(row0.rhs, 31);
    }

    #[test]
    fn test_tableau_row_mut_rational_addition() {
        let mut coefficients = Matrix::<Rational64>::new(2,2);
        coefficients[(0,0)] = Rational64::new(1,1);  coefficients[(0,1)] = Rational64::new(2,1);
        coefficients[(1,0)] = Rational64::new(3,1);  coefficients[(1,1)] = Rational64::new(4,1);

        let mut slack = Matrix::<Rational64>::new(2,2);
        slack[(0,0)] = Rational64::new(1,1);  slack[(0,1)] = Rational64::new(0,1);
        slack[(1,0)] = Rational64::new(0,1);  slack[(1,1)] = Rational64::new(1,1);
        
        let rhs = vec![Rational64::new(20,1), Rational64::new(10,1)];
        
        let z_coeffs = vec![Rational64::default(); 2];
        let z_slack = vec![Rational64::default(); 2];
        let z_rhs = Rational64::default();

        let mut tableau = Tableau::new(
            coefficients,
            slack,
            rhs,
            z_coeffs,
            z_slack,
            z_rhs,
        );

        {
            let row1 = tableau.row(1);
            let mut row0 = tableau.row_mut(0);
            row0 += row1;
        }

        let row0 = tableau.row(0);

        assert_eq!(row0.coefficients.data, vec![Rational64::new(4,1), Rational64::new(6,1)]);
    }

    #[test]
    fn test_tableau_row_mut_rational_multiplcation() {
        let mut coefficients = Matrix::<Rational64>::new(2,2);
        coefficients[(0,0)] = Rational64::new(1,1);  coefficients[(0,1)] = Rational64::new(2,1);
        coefficients[(1,0)] = Rational64::new(3,1);  coefficients[(1,1)] = Rational64::new(4,1);

        let mut slack = Matrix::<Rational64>::new(2,2);
        slack[(0,0)] = Rational64::new(1,1);  slack[(0,1)] = Rational64::new(0,1);
        slack[(1,0)] = Rational64::new(0,1);  slack[(1,1)] = Rational64::new(1,1);
        
        let rhs = vec![Rational64::new(20,1), Rational64::new(10,1)];
        
        let z_coeffs = vec![Rational64::default(); 2];
        let z_slack = vec![Rational64::default(); 2];
        let z_rhs = Rational64::default();

        let mut tableau = Tableau::new(
            coefficients,
            slack,
            rhs,
            z_coeffs,
            z_slack,
            z_rhs,
        );

        {
            let mut row0 = tableau.row_mut(0);
            row0 *= Rational64::new(3,1);
        }

        let row0 = tableau.row(0);

        assert_eq!(row0.coefficients.data, vec![Rational64::new(3,1), Rational64::new(6,1)]);
    }

    #[test]
    fn test_tableau_row_mut_rational_add_mul() {
        let mut coefficients = Matrix::<Rational64>::new(2,2);
        coefficients[(0,0)] = Rational64::new(1,1);  coefficients[(0,1)] = Rational64::new(2,1);
        coefficients[(1,0)] = Rational64::new(3,1);  coefficients[(1,1)] = Rational64::new(4,1);

        let mut slack = Matrix::<Rational64>::new(2,2);
        slack[(0,0)] = Rational64::new(1,1);  slack[(0,1)] = Rational64::new(0,1);
        slack[(1,0)] = Rational64::new(0,1);  slack[(1,1)] = Rational64::new(1,1);
        
        let rhs = vec![Rational64::new(20,1), Rational64::new(10,1)];
        
        let z_coeffs = vec![Rational64::default(); 2];
        let z_slack = vec![Rational64::default(); 2];
        let z_rhs = Rational64::default();

        let mut tableau = Tableau::new(
            coefficients,
            slack,
            rhs,
            z_coeffs,
            z_slack,
            z_rhs,
        );

        {
            let row1 = tableau.row(1);
            let mut row0 = tableau.row_mut(0);
            row0 += row1 * Rational64::new(3,1);
        }

        let row0 = tableau.row(0);

        assert_eq!(row0.coefficients.data, vec![Rational64::new(10,1), Rational64::new(14,1)]);
    }

    #[test]
    fn test_tableau_objective_negation() {
        let mut prob = Problem::new(vec![rational(3), rational(5)], Goal::Max);
        prob.add_constraint(vec![rational(1), rational(1)], Relation::LessEqual, rational(10));
        
        let tableau = prob.into_tableau_form();
        
        // Max 3x + 5y  => Min [-3, -5]
        assert_eq!(tableau.z_coeffs[0], rational(-3));
        assert_eq!(tableau.z_coeffs[1], rational(-5));
    }

    #[test]
    fn test_into_standard_form() {
        // Max  3x + 2y
        //      2x + 1y <= 10
        //      1x + 1y >= 4
        let mut prob = Problem::new(vec![rational(3), rational(2)], Goal::Max);
        prob.add_constraint(vec![rational(2), rational(1)], Relation::LessEqual, rational(10));
        prob.add_constraint(vec![rational(1), rational(1)], Relation::GreaterEqual, rational(4));

        let sf = prob.into_standard_form();

        // 2 decision vars + 2 slacks = 4 columns
        assert_eq!(sf.a.cols, 4);
        assert_eq!(sf.a.rows, 2);

        // Max(3, 2) becomes Min(-3, -2)
        assert_eq!(sf.c[0], rational(-3));
        assert_eq!(sf.c[1], rational(-2));
        assert_eq!(sf.c[2], rational(0));
        assert_eq!(sf.c[3], rational(0)); 

        // Row 2 should have a -1 surplus at index 3
        assert_eq!(sf.a[(1, 0)], rational(1));
        assert_eq!(sf.a[(1, 1)], rational(1));
        assert_eq!(sf.a[(1, 3)], rational(-1)); 
    }

    #[test]
    fn test_into_tableau_form() {
        let mut prob = Problem::new(vec![rational(1), rational(1)], Goal::Min);
        prob.add_constraint(vec![rational(1), rational(0)], Relation::LessEqual, rational(5));
        prob.add_constraint(vec![rational(0), rational(1)], Relation::LessEqual, rational(5));

        let tableau = prob.into_tableau_form();

        // Slack matrix must be m x m (2x2)
        assert_eq!(tableau.slack.rows, 2);
        assert_eq!(tableau.slack.cols, 2);

        // Initial basis [2, 3]
        assert_eq!(tableau.basis, vec![2, 3]);
        
        // Check that the slack matrix is an identity matrix
        assert_eq!(tableau.slack[(0, 0)], rational(1));
        assert_eq!(tableau.slack[(0, 1)], rational(0));
        assert_eq!(tableau.slack[(1, 0)], rational(0));
        assert_eq!(tableau.slack[(1, 1)], rational(1));
    }

    #[test]
    fn test_tableau_mixed_relations() {
        // Objective: x + y
        let mut prob = Problem::new(vec![rational(1), rational(1)], Goal::Max);
        // x + 0y <= 5  (Slack +1)
        prob.add_constraint(vec![rational(1), rational(0)], Relation::LessEqual, rational(5));
        // 0x + y >= 2   (Surplus -1)
        prob.add_constraint(vec![rational(0), rational(1)], Relation::GreaterEqual, rational(2));
        // x + y == 10   (Equality, no slack variable logic)
        prob.add_constraint(vec![rational(1), rational(1)], Relation::Equal, rational(10));

        let tableau = prob.into_tableau_form();

        // Rows = 3
        assert_eq!(tableau.rows(), 3);
        // Cols = 2 (vars) + 3 (slacks) + 1 (rhs) = 6
        assert_eq!(tableau.cols(), 6);

        // Row 0: Slack (+1)
        assert_eq!(tableau.slack[(0, 0)], rational(1));
        assert_eq!(tableau.slack[(0, 1)], rational(0));
        assert_eq!(tableau.slack[(0, 2)], rational(0));

        // Row 1: Surplus (-1)
        assert_eq!(tableau.slack[(1, 0)], rational(0));
        assert_eq!(tableau.slack[(1, 1)], rational(-1));
        assert_eq!(tableau.slack[(1, 2)], rational(0));

        // Row 2: Equality (All zeros)
        assert_eq!(tableau.slack[(2, 0)], rational(0));
        assert_eq!(tableau.slack[(2, 1)], rational(0));
        assert_eq!(tableau.slack[(2, 2)], rational(0));

        // Verify Basis/Non-basis
        // Decision variables are indices 0 and 1.
        // Slack variables are indices 2, 3, and 4.
        assert_eq!(tableau.nonbasis, vec![0, 1]);
        assert_eq!(tableau.basis, vec![2, 3, 4]);

        // RHS
        assert_eq!(tableau.rhs, vec![rational(5), rational(2), rational(10)]);
    }

    #[test]
    fn test_basic_pivot() {
        // Max 3x + 2y
        //     1x + 1y <= 4
        //     2x + 1y <= 5
        let obj = vec![Rational64::new(3, 1), Rational64::new(2, 1)];
        let mut prob = Problem::new(obj, crate::model::Goal::Max);
        prob.add_constraint(vec![Rational64::new(1, 1), Rational64::new(1, 1)], crate::model::Relation::LessEqual, Rational64::new(4, 1));
        prob.add_constraint(vec![Rational64::new(2, 1), Rational64::new(1, 1)], crate::model::Relation::LessEqual, Rational64::new(5, 1));

        let mut tab = prob.into_tableau_form();
        assert_eq!(tab.z_row()[0], Rational64::new(-3, 1));

        tab.pivot(1, 0);

        // Basis row 1 should now be x0
        assert_eq!(tab.basis[1], 0);

        // Row 1: [1, 1/2, 5/2]
        assert_eq!(tab[(1, 0)], Rational64::new(1, 1));
        assert_eq!(tab[(1, 1)], Rational64::new(1, 2));
        assert_eq!(tab.rhs[1], Rational64::new(5, 2));

        // Row 0: [0, 1/2, 3/2]
        assert_eq!(tab[(0, 0)], Rational64::new(0, 1));
        assert_eq!(tab[(0, 1)], Rational64::new(1, 2));
        assert_eq!(tab.rhs[0], Rational64::new(3, 2));

        // Z Row: [0, -1/2, 15/2]
        assert_eq!(tab.z_row()[0], Rational64::new(0, 1));
        assert_eq!(tab.z_row()[1], Rational64::new(-1, 2));
        assert_eq!(tab.z_rhs, Rational64::new(15, 2));

        // current_vertex: basis[0]=2 (s0), basis[1]=0 (x0) -> x0 = rhs[1] = 5/2, x1 = 0
        let vertex = tab.current_vertex(2);
        assert_eq!(vertex[0], Rational64::new(5, 2));
        assert_eq!(vertex[1], Rational64::new(0, 1));
        // Not yet optimal (negative reduced cost on x1)
        assert!(!tab.is_optimal());
    }
}