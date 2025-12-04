use std::ops::{Index, IndexMut};

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

impl<T> Matrix<T> {
    #[inline(always)]
    fn linear_index(&self, r: usize, c: usize) -> usize {
        debug_assert!(r < self.rows && c < self.cols);
        r * self.cols + c
    }

    #[inline(always)]
    fn row_offset(&self, r: usize) -> usize {
        debug_assert!(r < self.rows);
        r * self.cols
    }

    pub fn swap_elements(&mut self, r1: usize, c1: usize, r2: usize, c2: usize) {
        let idx1 = self.linear_index(r1, c1);
        let idx2 = self.linear_index(r2, c2);
        self.data.swap(idx1, idx2);
    }

    pub fn swap_rows(&mut self, r1: usize, r2: usize) {
        let row1 = self.row_offset(r1);
        let row2 = self.row_offset(r2);
        
        for col in 0..self.cols {
            self.data.swap(row1 + col, row2 + col);
        }
    }

    pub fn swap_columns(&mut self, c1: usize, c2: usize) {
        debug_assert!(c1 < self.cols && c2 < self.cols);

        for row in 0..self.rows {
            let row_offset = row * self.cols;
            self.data.swap(row_offset + c1, row_offset + c2);
        }
    }
}

impl<T: Clone + Default> Matrix<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![T::default(); rows * cols],
        }
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (r, c) = index;
        &self.data[self.linear_index(r, c)]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (r, c) = index;
        let idx = self.linear_index(r, c);
        &mut self.data[idx]
    }
}
