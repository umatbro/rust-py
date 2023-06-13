use num_cpus;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;
use std::thread;

fn multiply_matrices(matrix1: &[Vec<f64>], matrix2: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows_a = matrix1.len();
    let cols_b = matrix2[0].len();

    (0..rows_a)
        .into_par_iter()
        .map(|indx| {
            (0..cols_b)
                .map(|j| {
                    let mut sum = 0.0;
                    for k in 0..matrix2.len() {
                        sum += matrix1[indx][k] * matrix2[k][j];
                    }
                    sum
                })
                .collect()
        })
        .collect()
}

fn multiply_matrices_raw_no_rayon(
    matrix1: Arc<Vec<Vec<f64>>>,
    matrix2: Arc<Vec<Vec<f64>>>,
) -> Vec<Vec<f64>> {
    let num_threads = num_cpus::get();
    let chunk_size = (matrix1.len() as f32 / num_threads as f32).ceil() as usize;
    let mut handles = Vec::with_capacity(num_threads);

    for th_num in 0..num_threads {
        let chunk_start = th_num * chunk_size;
        if chunk_start >= matrix1.len() {
            break;
        }
        let is_last_iteration = th_num == num_threads - 1;
        let chunk_end = if is_last_iteration {
            matrix1.len()
        } else {
            chunk_start + chunk_size
        };

        let m1 = Arc::clone(&matrix1);
        let m2 = Arc::clone(&matrix2);

        let handle = thread::spawn(move || {
            let mut local_result = vec![vec![0.0; m2[0].len()]; chunk_end - chunk_start];
            for row_a in chunk_start..chunk_end {
                for col_b in 0..m2[0].len() {
                    let mut sum = 0.0;
                    for i in 0..m2.len() {
                        sum += m1[row_a][i] * m2[i][col_b];
                    }
                    local_result[row_a - chunk_start][col_b] = sum;
                }
            }

            local_result
        });
        handles.push(handle);
    }

    let mut result = Vec::with_capacity(matrix1.len());
    for handle in handles {
        let local_result = handle.join().unwrap();
        result.extend(local_result);
    }

    result
}

#[pyclass]
#[derive(Debug)]
struct RsMatrix {
    data: Arc<Vec<Vec<f64>>>,
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
        Ok(Self {
            data: Arc::new(data),
            rows,
            cols,
        })
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

        // let result = (0..self.rows).map(|row_a_num|{
        //     (0..other.cols).map(|col_b_num|{
        //         (0..self.cols).map(|i|{
        //             self.data[row_a_num][i] * other.data[i][col_b_num]
        //         }).sum()
        //     }).collect()
        // }).collect();

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

        // let result = multiply_matrices(&self.data, &other.data);
        let result =
            multiply_matrices_raw_no_rayon(Arc::clone(&self.data), Arc::clone(&other.data));

        Ok(Self::new(result)?)
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.data)
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
        let b = RsMatrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let result = a.mul(&b);
        assert!(result.is_err());
    }
}
