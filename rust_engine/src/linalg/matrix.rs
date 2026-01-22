use std::ops::{Index, IndexMut, Range, Deref, DerefMut};

#[derive(Debug, Clone)]
pub struct Matrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<T>,
}

impl<T> Matrix<T> {
    pub fn with_capacity(rows: usize, cols: usize) -> Self {
        Matrix {
            rows: 0,
            cols,
            data: Vec::with_capacity(rows * cols),
        }
    }

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

    #[inline(always)]
    fn row_range(&self, r: usize) -> Range<usize> {
        let start = self.row_offset(r);
        start..start + self.cols
    }

    pub fn swap_elements(&mut self, r1: usize, c1: usize, r2: usize, c2: usize) {
        let idx1 = self.linear_index(r1, c1);
        let idx2 = self.linear_index(r2, c2);
        self.data.swap(idx1, idx2);
    }

    pub fn swap_columns(&mut self, c1: usize, c2: usize) {
        debug_assert!(c1 < self.cols && c2 < self.cols);

        for row in 0..self.rows {
            let row_offset = row * self.cols;
            self.data.swap(row_offset + c1, row_offset + c2);
        }
    }

    pub fn swap_rows(&mut self, r1: usize, r2: usize) {
        if r1 == r2 { return; }
        let (r1, r2) = if r1 > r2 { (r2, r1) } else { (r1, r2) };
        
        let range1 = self.row_range(r1);
        let range2 = self.row_range(r2);
        
        let (left, right) = self.data.split_at_mut(range2.start);
        let row1 = &mut left[range1];
        let row2 = &mut right[..self.cols];
        
        row1.swap_with_slice(row2);
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

    pub fn push_row(&mut self, new_row: &[T]) {
        assert_eq!(new_row.len(), self.cols, "Row length must match matrix dimensions");
        self.data.extend_from_slice(new_row);
        self.rows += 1;
    }

    pub fn push_empty_row(&mut self) {
        self.data.extend((0..self.cols).map(|_| T::default()));
        self.rows += 1;
    }

    // This is not memory efficient, should implement splciing here.
    // Splicing may be computationally expensive due to memory shifting.
    pub fn push_column(&mut self, new_col: Option<&[T]>) {
        assert!(new_col.map_or(true, |col| col.len() == self.rows), "Column length must match matrix dimensions");

        let mut new_data = Vec::with_capacity((self.cols + 1) * self.rows);

        for r in 0..self.rows {
            let row_start = r * self.cols;
            let row_end = row_start + self.cols;
            new_data.extend_from_slice(&self.data[row_start..row_end]);

            let val = if let Some(col) = new_col { col[r].clone() } else { T::default() };
            new_data.push(val);
        }

        self.cols += 1;
        self.data = new_data;
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

#[derive(Debug, Clone)]
pub struct Row<T> {
    pub data: Vec<T>,
}

#[derive(Debug)]
pub struct RowMut<'a, T> {
    pub data: &'a mut [T],
}

impl<T: Clone> Matrix<T> {
    pub fn row(&self, r: usize) -> Row<T> {
        let range = self.row_range(r);
        Row { data: self.data[range].to_owned() }
    }

    pub fn row_mut(&mut self, r: usize) -> RowMut<'_, T> {
        let range = self.row_range(r);
        RowMut { data: &mut self.data[range] }
    }
}

impl<T> Deref for Row<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for Row<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<'a, T> Deref for RowMut<'a, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a, T> DerefMut for RowMut<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}

impl<T> Index<usize> for Row<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for Row<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<'a, T> Index<usize> for RowMut<'a, T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<'a, T> IndexMut<usize> for RowMut<'a, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}