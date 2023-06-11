import pytest
from matrix import PyMatrix


@pytest.mark.parametrize(
    ["a", "b", "expected"],
    [
        (
            PyMatrix([
                [1, 2, 3],
                [4, 5, 6],
            ]),
            PyMatrix([
                [7, 8],
                [9, 10],
                [11, 12],
            ]),
            PyMatrix([
                [58, 64],
                [139, 154],
            ]),
        ),
        (
            PyMatrix([
                [3, 6, 9],
                [2, 4, 6],
            ]),
            PyMatrix([
                [1, 2],
                [3, 4],
                [5, 6],
            ]),
            PyMatrix([
                [66, 84],
                [44, 56],
            ]),
        )
    ]
)
def test_python_matrix_mul(a, b, expected):
    result = a.mul(b)
    assert result.data == expected.data
