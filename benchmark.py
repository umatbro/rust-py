import random
import time

from matrix import PyMatrix
from mtrx_rs import RsMatrix

ROW_COUNT_A = 100
COL_COUNT_A = 100
ROW_COUNT_B = 100
COL_COUNT_B = 100
NUM_TESTS = 100


def generate_test_matrix(rows: int, cols: int) -> list[list[float]]:
    return [[random.random() for _ in range(cols)] for _ in range(rows)]


def setup_benchmark() -> list[tuple[list[list[float]], list[list[float]]]]:
    """
    Generate matrices for benchmarking.
    :return: list of tuples of matrices
    """
    matrices: list[tuple[list[list[float]], list[list[float]]]] = []
    for iter_num in range(NUM_TESTS):
        matrices.append(
            (
                generate_test_matrix(ROW_COUNT_A, COL_COUNT_A),
                generate_test_matrix(ROW_COUNT_B, COL_COUNT_B),
            )
        )

    return matrices


def benchmark_python(test_data: list[tuple[list[list[float]], list[list[float]]]]):
    for a, b in test_data:
        PyMatrix(a).mul(PyMatrix(b))


def benchmark_rs(test_data: list[tuple[list[list[float]], list[list[float]]]]):
    for a, b in test_data:
        RsMatrix(a).mul(RsMatrix(b))


def benchmark_rs_parallel(test_data: list[tuple[list[list[float]], list[list[float]]]]):
    for a, b in test_data:
        RsMatrix(a).mul_par(RsMatrix(b))


def time_execution(func, *args, **kwargs) -> float:
    start = time.time()
    func(*args, **kwargs)
    end = time.time()
    return end - start


if __name__ == "__main__":
    print(f"Running {NUM_TESTS} tests of {ROW_COUNT_A}x{COL_COUNT_A} * {ROW_COUNT_B}x{COL_COUNT_B} matrix multiplication")
    setup = setup_benchmark()
    print("Starting benchmark")
    python_time = time_execution(benchmark_python, setup)
    print(f"Python implementation total: {python_time}s")
    print(f"Python implementation per iteration: {python_time / NUM_TESTS}s")
    rs_time = time_execution(benchmark_rs, setup)
    print(f"Rust implementation total: {rs_time}s")
    print(f"Rust implementation per iteration: {rs_time / NUM_TESTS}s")

    rs_time = time_execution(benchmark_rs_parallel, setup)
    print(f"Rust implementation (parallel) total: {rs_time}s")
    print(f"Rust implementation (parallel) per iteration: {rs_time / NUM_TESTS}s")