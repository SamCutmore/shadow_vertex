use crate::linalg::Matrix;
use super::Goal;

pub struct StandardForm<T> {
    pub a: Matrix<T>,
    pub b: Vec<T>,
    pub c: Vec<T>,
    pub goal: Goal,
    pub slack_indices: Vec<usize>, 
}

impl<T> StandardForm<T> {
    pub fn new(a: Matrix<T>, b: Vec<T>, c: Vec<T>, goal: Goal, slack_indices: Vec<usize>,) -> Self {
        assert_eq!(a.rows, b.len(), "Matrix A rows must match vector b length");
        assert_eq!(a.cols, c.len(), "Matrix A columns must match vector c length");
        
        StandardForm { a, b, c, goal, slack_indices,}
    }
}
