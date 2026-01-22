pub mod matrix;
pub mod matrix_operations;
pub mod matrix_arithmetic;
pub mod matrix_row_operations;

pub use matrix::{Matrix, Row, RowMut};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_swap() {
        let mut m = Matrix::<i32>::new(2, 2);
        m[(0,0)] =  1;
        m[(1,1)] =  9;
        m.swap_elements(0, 0, 1, 1);
        assert_eq!(m[(0,0)], 9);
        assert_eq!(m[(0,1)], 0);
        assert_eq!(m[(1,0)], 0);
        assert_eq!(m[(1,1)], 1);
    }

    #[test]
    fn test_matrix_swap_rows() {
        let mut m = Matrix::<i32>::new(2, 2);
        m[(0,0)] =  1;
        m[(1,1)] =  9;
        m.swap_rows(0, 1);
        assert_eq!(m[(0,0)], 0);
        assert_eq!(m[(0,1)], 9);
        assert_eq!(m[(1,0)], 1);
        assert_eq!(m[(1,1)], 0);
    }

    #[test]
    fn test_matrix_swap_columns() {
        let mut m = Matrix::<i32>::new(2, 2);
        m[(0,0)] =  1;
        m[(1,1)] =  9;
        m.swap_columns(0, 1);
        assert_eq!(m[(0,0)], 0);
        assert_eq!(m[(0,1)], 1);
        assert_eq!(m[(1,0)], 9);
        assert_eq!(m[(1,1)], 0);
    }

    #[test]
    fn test_matrix_addition() {
        let mut a = Matrix::<i32>::new(2,2);
        let mut b = Matrix::<i32>::new(2,2);
        
        a[(0,0)] = 1; a[(0,1)] = 2;
        a[(1,0)] = 3; a[(1,1)] = 4;
        
        b[(0,0)] = 5; b[(0,1)] = 6;
        b[(1,0)] = 7; b[(1,1)] = 8;

        let c = &a + &b;
        assert_eq!(c[(0,0)], 6);
        assert_eq!(c[(0,1)], 8);
        assert_eq!(c[(1,0)], 10);
        assert_eq!(c[(1,1)], 12);

        let mut a_clone = a.clone();
        a_clone += &b;
        assert_eq!(a_clone[(0,0)], 6);
        assert_eq!(a_clone[(0,1)], 8);
        assert_eq!(a_clone[(1,0)], 10);
        assert_eq!(a_clone[(1,1)], 12);
    }

    #[test]
    fn test_matrix_subtraction() {
        let mut a = Matrix::<i32>::new(2,2);
        let mut b = Matrix::<i32>::new(2,2);

        a[(0,0)] = 5; a[(0,1)] = 6;
        a[(1,0)] = 7; a[(1,1)] = 8;
        
        b[(0,0)] = 1; b[(0,1)] = 2;
        b[(1,0)] = 3; b[(1,1)] = 4;

        let c = &a - &b;
        assert_eq!(c[(0,0)], 4);
        assert_eq!(c[(0,1)], 4);
        assert_eq!(c[(1,0)], 4);
        assert_eq!(c[(1,1)], 4);

        let mut a_clone = a.clone();
        a_clone -= &b;
        assert_eq!(a_clone[(0,0)], 4);
        assert_eq!(a_clone[(0,1)], 4);
        assert_eq!(a_clone[(1,0)], 4);
        assert_eq!(a_clone[(1,1)], 4);
    }

    #[test]
    fn test_hadamard_and_scalar() {
        let mut a = Matrix::<i32>::new(2,2);
        let mut b = Matrix::<i32>::new(2,2);

        a[(0,0)] = 1; a[(0,1)] = 2;
        a[(1,0)] = 3; a[(1,1)] = 4;
        
        b[(0,0)] = 2; b[(0,1)] = 0;
        b[(1,0)] = 1; b[(1,1)] = 2;

        // Hadamard product
        let c = &a * &b;
        assert_eq!(c[(0,0)], 2);
        assert_eq!(c[(0,1)], 0);
        assert_eq!(c[(1,0)], 3);
        assert_eq!(c[(1,1)], 8);

        // In-place Hadamard
        let mut a_clone = a.clone();
        a_clone *= &b;
        assert_eq!(a_clone[(0,0)], 2);
        assert_eq!(a_clone[(0,1)], 0);
        assert_eq!(a_clone[(1,0)], 3);
        assert_eq!(a_clone[(1,1)], 8);

        // Scalar multiplication
        let d = &a * 3;
        assert_eq!(d[(0,0)], 3);
        assert_eq!(d[(0,1)], 6);
        assert_eq!(d[(1,0)], 9);
        assert_eq!(d[(1,1)], 12);

        // In-place scalar multiplication
        let mut a_clone2 = a.clone();
        a_clone2 *= 2;
        assert_eq!(a_clone2[(0,0)], 2);
        assert_eq!(a_clone2[(0,1)], 4);
        assert_eq!(a_clone2[(1,0)], 6);
        assert_eq!(a_clone2[(1,1)], 8);
    }

    #[test]
    fn test_dot_product() {
        let mut a = Matrix::<i32>::new(2,3);
        let mut b = Matrix::<i32>::new(3,2);

        a[(0,0)] = 1; a[(0,1)] = 2; a[(0,2)] = 3;
        a[(1,0)] = 4; a[(1,1)] = 5; a[(1,2)] = 6;

        b[(0,0)] = 7; b[(0,1)] = 8;
        b[(1,0)] = 9; b[(1,1)] = 10;
        b[(2,0)] = 11; b[(2,1)] = 12;

        let c = a.dot(&b);
        assert_eq!(c.rows, 2);
        assert_eq!(c.cols, 2);

        assert_eq!(c[(0,0)], 58);  // 1*7 + 2*9 + 3*11
        assert_eq!(c[(0,1)], 64);  // 1*8 + 2*10 + 3*12
        assert_eq!(c[(1,0)], 139); // 4*7 + 5*9 + 6*11
        assert_eq!(c[(1,1)], 154); // 4*8 + 5*10 + 6*12
    }

    #[test]
    fn test_push_row() {
        let mut m = Matrix::<i32>::new(2, 3);
        m[(0, 0)] = 1; m[(0, 1)] = 2; m[(0, 2)] = 3;
        m[(1, 0)] = 4; m[(1, 1)] = 5; m[(1, 2)] = 6;

        let new_row = [7, 8, 9];
        m.push_row(&new_row);

        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 3);

        // New Row Added
        assert_eq!(m[(2, 0)], 7);
        assert_eq!(m[(2, 1)], 8);
        assert_eq!(m[(2, 2)], 9);

        // original data still intact
        assert_eq!(m[(0, 0)], 1);
        assert_eq!(m[(1, 2)], 6);
    }

    #[test]
    fn test_push_empty_row() {
        let mut m = Matrix::<i32>::new(2, 2);
        m.push_empty_row();

        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 2);

        for c in 0..2 {
            assert_eq!(m[(2, c)], 0);
        }
    }

    #[test]
    fn test_push_column_with_data() {
        let mut m = Matrix::<i32>::new(2, 2);
        m[(0, 0)] = 1; m[(0, 1)] = 2;
        m[(1, 0)] = 3; m[(1, 1)] = 4;

        let new_col = [5, 6];
        m.push_column(Some(&new_col));

        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);

        // original data preserved
        assert_eq!(m[(0, 0)], 1);
        assert_eq!(m[(1, 1)], 4);

        // new column added
        assert_eq!(m[(0, 2)], 5);
        assert_eq!(m[(1, 2)], 6);
    }

    #[test]
    fn test_push_column_default() {
        let mut m = Matrix::<i32>::new(2, 2);
        m.push_column(None);

        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);

        for r in 0..2 {
            assert_eq!(m[(r, 2)], 0);
        }
    }

    #[test]
    fn test_row_arithmetic_chain() {
        let mut m: Matrix<i32> = Matrix::new(3, 2);
        m[(0, 0)] = 1; m[(0, 1)] = 2;
        m[(1, 0)] = 3; m[(1, 1)] = 4;
        m[(2, 0)] = 5; m[(2, 1)] = 6;
        
        let r1 = m.row(1);
        let r3 = m.row(2);
        let mut r0 = m.row_mut(0);
        r0 *= 2;
        r0 += (r1*2) + r3 + 1;

        assert_eq!(m[(0,0)], 14);
        assert_eq!(m[(0,1)], 19);
    }
}