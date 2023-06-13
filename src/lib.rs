use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;



fn multiply_matrices(matrix1: &[Vec<f64>], matrix2: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows_a = matrix1.len();
    let cols_b = matrix2[0].len();

    (0..rows_a).into_par_iter().map(|indx| {
        (0..cols_b).map(|j| {
            let mut sum = 0.0;
            for k in 0..matrix2.len() {
                sum += matrix1[indx][k] * matrix2[k][j];
            }
            sum
        }).collect()
    }).collect()
}


#[pyclass]
#[derive(Debug)]
struct RsMatrix {
    data: Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
}

#[pymethods]
impl RsMatrix {
    /// Creates a new matrix from a 2D array of floats.
    #[new]
    fn new(data: Vec<Vec<f64>>) -> Result<Self, PyErr> {
        // check if all rows have the same length
        let row_len = data[0].len();
        if !data.iter().all(|row| row.len() == row_len) {
            return Err(PyValueError::new_err("All rows must have the same length"));
        }
        let rows = data.len();
        let cols = row_len;
        Ok(Self { data, rows, cols })
    }

    /// Calculates the dot product of two matrices.
    ///
    /// # Arguments
    ///
    /// * `other` - The second matrix.
    #[pyo3(text_signature = "($self, other)")]
    fn mul(&self, other: &Self) -> Result<Self, PyErr> {
        if self.cols != other.rows {
            return Err(PyValueError::new_err(format!(
                "Invalid matrix dimensions. \
                The number of columns of the first matrix should be equal to the number of rows \
                of the second matrix. Got {} and {} instead.",
                self.cols, other.rows,
            )));
        }

        let mut result = vec![vec![0.0; other.cols]; self.rows];

        for row_a in 0..self.rows {
            for col_b in 0..other.cols {
                for i in 0..self.cols {
                    result[row_a][col_b] += self.data[row_a][i] * other.data[i][col_b];
                }
            }
        }

        Ok(Self::new(result)?)
    }

    fn mul_par(&self, other: &Self) -> Result<Self, PyErr> {
        if self.cols != other.rows {
                return Err(PyValueError::new_err(format!(
                    "Invalid matrix dimensions. \
                    The number of columns of the first matrix should be equal to the number of rows \
                    of the second matrix. Got {} and {} instead.",
                self.cols, other.rows,
            )));
        }

        let result = multiply_matrices(&self.data, &other.data);

        Ok(Self::new(result)?)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn mtrx_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RsMatrix>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::RsMatrix;
    use rstest::rstest;

    #[rstest]
    #[case(
        vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ],
        vec![
            vec![7.0, 8.0],
            vec![9.0, 10.0],
            vec![11.0, 12.0],
        ],
        vec![
            vec![58.0, 64.0],
            vec![139.0, 154.0],
        ]
    )]
    #[case(
        vec![
            vec![3.0, 6.0, 9.0],
            vec![2.0, 4.0, 6.0],
        ],
        vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ],
        vec![
            vec![66.0, 84.0],
            vec![44.0, 56.0],
        ]
    )]
    fn test_matrix_mul(
        #[case] a: Vec<Vec<f64>>,
        #[case] b: Vec<Vec<f64>>,
        #[case] expected: Vec<Vec<f64>>,
    ) {
        let a = RsMatrix::new(a).unwrap();
        let b = RsMatrix::new(b).unwrap();
        let result = a.mul(&b);
        let result_par = a.mul_par(&b);

        assert_eq!(*result.unwrap().data, expected);
        assert_eq!(*result_par.unwrap().data, expected);
    }

    #[test]
    fn test_wrong_dimensions() {
        let a = RsMatrix::new(vec![vec![1.0, 2.0, 3.0]]).unwrap();
        let b = RsMatrix::new(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]).unwrap();

        let result = a.mul(&b);
        assert!(result.is_err());
    }
}
