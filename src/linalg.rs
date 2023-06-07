use std::{ops::*, process::Output};

#[derive(Debug, Clone)]
pub struct Matrix {
    // M rows, N columns
    m: usize,
    n: usize,
    arr: Vec<f64>,
}

// AI generated code
impl Matrix {
    pub fn new(m: usize, n: usize) -> Matrix {
        Matrix {
            m,
            n,
            arr: vec![0.0; m * n],
        }
    }

    pub fn from(m: usize, n: usize, arr: &[f64]) -> Matrix {
        Matrix {
            m,
            n,
            arr: arr.to_vec(),
        }
    }

    pub fn randomize(&self) -> Matrix {
        let mut result = Matrix::new(self.m, self.n);
        for i in 0..self.m {
            for j in 0..self.n {
                result[(i, j)] = rand::random();
            }
        }
        result
    }

    pub fn size(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.n, self.m);
        for i in 0..self.m {
            for j in 0..self.n {
                result[(j, i)] = self[(i, j)];
            }
        }
        result
    }

    pub fn map(&self, f: fn(f64) -> f64) -> Matrix {
        let mut result = Matrix::new(self.m, self.n);
        for i in 0..self.m {
            for j in 0..self.n {
                result[(i, j)] = f(self[(i, j)]);
            }
        }
        result
    }

    pub fn get_row(&self, i: usize) -> Matrix {
        let mut result = Matrix::new(1, self.n);
        for j in 0..self.n {
            result[(0, j)] = self[(i, j)];
        }
        result
    }

    pub fn rows(&self) -> usize {
        self.m
    }

    pub fn cols(&self) -> usize {
        self.n
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        let mut result = Matrix::new(self.m, self.n);
        for i in 0..self.m {
            for j in 0..self.n {
                result[(i, j)] = self[(i, j)] + other[(i, j)];
            }
        }
        result
    }

    pub fn sub(&self, other: &Matrix) -> Matrix {
        let mut result = Matrix::new(self.m, self.n);
        for i in 0..self.m {
            for j in 0..self.n {
                result[(i, j)] = self[(i, j)] - other[(i, j)];
            }
        }
        result
    }

    pub fn scale(&self, scalar: f64) -> Matrix {
        let mut result = Matrix::new(self.m, self.n);
        for i in 0..self.m {
            for j in 0..self.n {
                result[(i, j)] = self[(i, j)] * scalar;
            }
        }
        result
    }

    pub fn dot(&self, other: &Matrix) -> Matrix {
        if other.rows() != self.cols() {
            panic!(
                "Matrix dimensions do not match: {}x{} * {}x{}",
                self.m, self.n, other.m, other.n
            );
        }
        let mut result = Matrix::new(self.m, other.n);
        for i in 0..self.m {
            for j in 0..other.n {
                let mut sum = 0.0;
                for k in 0..self.n {
                    sum += self[(i, k)] * other[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        result
    }

    pub fn hadamard(&self, other: &Matrix) -> Matrix {
        if self.size() != other.size() {
            panic!(
                "Matrix dimensions do not match: {}x{} * {}x{}",
                self.m, self.n, other.m, other.n
            );
        }
        let mut result = Matrix::new(self.m, self.n);
        for i in 0..self.m {
            for j in 0..self.n {
                result[(i, j)] = self[(i, j)] * other[(i, j)];
            }
        }
        result
    }

    pub fn normalize(&self) -> Matrix {
        let mut result = Matrix::new(self.m, self.n);
        for i in 0..self.n {
            let mut min = std::f64::INFINITY;
            let mut max = std::f64::NEG_INFINITY;
            for j in 0..self.m {
                if self[(j, i)] < min {
                    min = self[(j, i)];
                }
                if self[(j, i)] > max {
                    max = self[(j, i)];
                }
            }

            let range = max - min;
            for j in 0..self.m {
                result[(j, i)] = (self[(j, i)] - min) / range;
            }
        }
        result
    }

    pub fn slice_x(&self, range: std::ops::Range<usize>) -> Matrix {
        let mut result = Matrix::new(self.m, range.len());
        for i in 0..self.m {
            for j in range.clone() {
                result[(i, j - range.start)] = self[(i, j)];
            }
        }
        result
    }

    pub fn slice_y(&self, range: std::ops::Range<usize>) -> Matrix {
        let mut result = Matrix::new(range.len(), self.n);
        for i in range.clone() {
            for j in 0..self.n {
                result[(i - range.start, j)] = self[(i, j)];
            }
        }
        result
    }

    pub fn clear(&mut self) {
        for i in 0..self.m {
            for j in 0..self.n {
                self[(i, j)] = 0.0;
            }
        }
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &f64 {
        let (i, j) = index;
        &self.arr[i * self.n + j]
    }
}

impl Index<usize> for Matrix {
    type Output = [f64];

    fn index(&self, index: usize) -> &Self::Output {
        &self.arr[self.m * index..self.m * (index + 1)]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut f64 {
        let (i, j) = index;
        &mut self.arr[i * self.n + j]
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Matrix) -> bool {
        if self.m != other.m || self.n != other.n {
            return false;
        }
        for i in 0..self.m {
            for j in 0..self.n {
                if self[(i, j)] != other[(i, j)] {
                    return false;
                }
            }
        }
        true
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        for i in 0..self.m {
            for j in 0..self.n {
                write!(f, "{:.5} ", self[(i, j)])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
