from __future__ import annotations


class PyMatrix:
    def __init__(self, data: list[list[float]]):
        if not all(len(row) == len(data[0]) for row in data):
            raise ValueError("Invalid matrix dimensions. All rows must have the same length.")
        self.rows = len(data)
        self.cols = len(data[0])
        self.data = data

    def mul(self, other: PyMatrix):
        if self.cols != other.rows:
            raise ValueError(
                "Invalid matrix dimensions. "
                "The number of columns of the first matrix must be equal to the number of rows of the second matrix. "
                f"Got {self.cols} and {other.rows} instead."
            )

        result = PyMatrix([[0 for _ in range(other.cols)] for _ in range(self.rows)])

        for row_self in range(self.rows):
            for col_other in range(other.cols):
                for i in range(self.cols):
                    result.data[row_self][col_other] += self.data[row_self][i] * other.data[i][col_other]

        return result
