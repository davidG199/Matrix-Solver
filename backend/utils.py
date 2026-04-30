"""Funciones reutilizables para validar, normalizar y reducir matrices.

Este modulo centraliza la limpieza de entradas, la construccion de matrices
aumentadas y la reduccion por filas usada por las operaciones algebraicas.
"""

from __future__ import annotations

from typing import Any

import numpy as np

EPSILON = 1e-10


def _is_numeric(value: Any) -> bool:
    """Return whether a value is a real numeric scalar.

    Args:
        value: Any candidate value that should be inspected.

    Returns:
        True when the value is an int, float or numpy numeric type, excluding
        booleans.
    """
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _coerce_number(value: Any, context: str) -> float:
    """Convert a value to float after validating that it is numeric.

    Args:
        value: The value to convert.
        context: Human readable label used in validation errors.

    Returns:
        The numeric value converted to float.

    Raises:
        ValueError: If the value is not numeric or is not finite.
    """
    if not _is_numeric(value):
        raise ValueError(f"{context} debe contener solo numeros reales.")

    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{context} contiene valores no finitos.")
    return number


def normalize_matrix(matrix: Any, name: str = "matriz") -> np.ndarray:
    """Normalize a matrix-like input into a 2D numpy array of floats.

    Args:
        matrix: Nested lists or an existing numpy array.
        name: Label used in validation errors to identify the input.

    Returns:
        A rectangular 2D ndarray with dtype float.

    Raises:
        ValueError: If the input is empty, jagged, non numeric or not 2D.

    Process:
        1. Accept numpy arrays directly when they are already 2D.
        2. Accept nested lists/tuples as matrix rows.
        3. Validate every row length and every scalar value.
        4. Convert the data to a float array ready for algebraic operations.
    """
    if isinstance(matrix, np.ndarray):
        data = np.asarray(matrix, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"{name} debe ser bidimensional.")
        if data.size == 0 or 0 in data.shape:
            raise ValueError(f"{name} no puede estar vacia.")
        if not np.all(np.isfinite(data)):
            raise ValueError(f"{name} contiene valores no finitos.")
        return data

    if not isinstance(matrix, (list, tuple)) or len(matrix) == 0:
        raise ValueError(f"{name} debe contener al menos una fila.")

    rows = []
    expected_columns = None
    for row_index, row in enumerate(matrix, start=1):
        if not isinstance(row, (list, tuple, np.ndarray)) or len(row) == 0:
            raise ValueError(f"{name} fila {row_index} debe ser una lista no vacia de numeros.")

        current_row = [_coerce_number(value, f"{name} fila {row_index}") for value in row]
        if expected_columns is None:
            expected_columns = len(current_row)
        elif len(current_row) != expected_columns:
            raise ValueError(
                f"{name} debe ser rectangular: todas las filas deben tener {expected_columns} valores."
            )
        rows.append(current_row)

    data = np.asarray(rows, dtype=float)
    if data.ndim != 2 or data.size == 0:
        raise ValueError(f"{name} debe ser una matriz valida.")
    return data


def normalize_vector(vector: Any, expected_size: int | None = None, name: str = "vector") -> np.ndarray:
    """Normalize a vector-like input into a 1D numpy array of floats.

    Args:
        vector: A list, tuple, numpy array or a column/row vector equivalent.
        expected_size: Optional exact length that the vector must satisfy.
        name: Label used in validation errors to identify the input.

    Returns:
        A 1D ndarray with dtype float.

    Raises:
        ValueError: If the input is empty, nested in an invalid way or has the
            wrong size.
    """
    if isinstance(vector, np.ndarray):
        data = np.asarray(vector, dtype=float)
    else:
        if not isinstance(vector, (list, tuple)) or len(vector) == 0:
            raise ValueError(f"{name} debe contener al menos un valor.")

        values = []
        for index, value in enumerate(vector, start=1):
            if isinstance(value, (list, tuple, np.ndarray)):
                if len(value) != 1:
                    raise ValueError(f"{name} debe ser un vector, no una matriz.")
                value = value[0]
            values.append(_coerce_number(value, f"{name} posicion {index}"))
        data = np.asarray(values, dtype=float)

    if data.ndim == 2 and 1 in data.shape:
        data = data.reshape(-1)
    elif data.ndim != 1:
        raise ValueError(f"{name} debe ser un vector unidimensional.")

    if data.size == 0:
        raise ValueError(f"{name} no puede estar vacio.")
    if expected_size is not None and data.shape[0] != expected_size:
        raise ValueError(f"{name} debe tener {expected_size} valores.")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{name} contiene valores no finitos.")
    return data.astype(float)


def require_same_shape(matrix_a: np.ndarray, matrix_b: np.ndarray, operation: str) -> None:
    """Ensure that two matrices share the same shape.

    Args:
        matrix_a: First matrix to compare.
        matrix_b: Second matrix to compare.
        operation: Name of the algebraic operation that requires equality.

    Raises:
        ValueError: If both matrices do not have the same dimensions.
    """
    if matrix_a.shape != matrix_b.shape:
        raise ValueError(f"Las matrices deben tener la misma dimension para {operation}.")


def require_multiplicable(matrix_a: np.ndarray, matrix_b: np.ndarray) -> None:
    """Validate the dimension rule for matrix multiplication.

    Args:
        matrix_a: Left matrix.
        matrix_b: Right matrix.

    Raises:
        ValueError: If the number of columns of A does not match the number of
        rows of B.
    """
    if matrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError("No se pueden multiplicar: las columnas de A deben coincidir con las filas de B.")


def require_square(matrix: np.ndarray, name: str = "matriz") -> None:
    """Ensure that a matrix is square.

    Args:
        matrix: Matrix to validate.
        name: Label used in validation errors.

    Raises:
        ValueError: If the matrix is not n x n.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} debe ser cuadrada.")


def clean_matrix(matrix: np.ndarray) -> np.ndarray:
    """Replace tiny floating-point residues with exact zeros.

    Args:
        matrix: Matrix to clean after a numerical operation.

    Returns:
        A copied matrix where values with absolute value lower than EPSILON are
        set to zero.
    """
    data = np.asarray(matrix, dtype=float).copy()
    data[np.abs(data) < EPSILON] = 0.0
    return data


def matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    """Convert a matrix to a JSON-friendly list of lists.

    Args:
        matrix: Numeric matrix to serialize.

    Returns:
        A nested list representation of the matrix.
    """
    return clean_matrix(matrix).tolist()


def vector_to_list(vector: np.ndarray) -> list[float]:
    """Convert a vector to a JSON-friendly flat list.

    Args:
        vector: Numeric vector to serialize.

    Returns:
        A plain Python list of floats.
    """
    return clean_matrix(np.asarray(vector, dtype=float).reshape(-1)).tolist()


def build_augmented_matrix(matrix_a: np.ndarray, vector_b: np.ndarray) -> np.ndarray:
    """Build the augmented matrix [A | b].

    Args:
        matrix_a: Coefficient matrix.
        vector_b: Independent term vector.

    Returns:
        The augmented matrix obtained by appending b as the last column.
    """
    return np.hstack([matrix_a, vector_b.reshape(-1, 1)])


def matrix_rank(matrix: np.ndarray) -> int:
    """Compute the linear algebra rank of a matrix.

    Args:
        matrix: Numeric matrix whose rank will be computed.

    Returns:
        The rank as an integer.
    """
    return int(np.linalg.matrix_rank(np.asarray(matrix, dtype=float)))


def classify_system(matrix_a: np.ndarray, vector_b: np.ndarray) -> tuple[str, int, int, np.ndarray]:
    """Classify a linear system using the ranks of A and [A | b].

    Args:
        matrix_a: Coefficient matrix.
        vector_b: Independent term vector.

    Returns:
        A tuple containing the status, rank of A, rank of the augmented matrix
        and the augmented matrix itself.

    Status values:
        incompatible: no solution.
        compatible_determinado: unique solution.
        compatible_indeterminado: infinitely many solutions.
    """
    augmented = build_augmented_matrix(matrix_a, vector_b)
    rank_a = matrix_rank(matrix_a)
    rank_augmented = matrix_rank(augmented)

    if rank_a != rank_augmented:
        status = "incompatible"
    elif rank_a == matrix_a.shape[1]:
        status = "compatible_determinado"
    else:
        status = "compatible_indeterminado"

    return status, rank_a, rank_augmented, augmented


def _row_reduce(matrix: np.ndarray, reduce_above: bool, pivot_columns: int | None = None) -> tuple[np.ndarray, list[int]]:
    """Reduce a matrix using Gaussian elimination with partial pivoting.

    Args:
        matrix: Matrix to reduce.
        reduce_above: When True, also eliminate entries above each pivot to
            produce reduced row echelon form.
        pivot_columns: Optional limit for the number of columns to inspect as
            pivots.

    Returns:
        A tuple with the reduced matrix and the list of pivot columns found.

    Process:
        1. Search the best pivot in the current column using the largest
           absolute value available.
        2. Swap rows when necessary so the pivot is placed on the active row.
        3. Normalize the pivot row so the pivot becomes 1.
        4. Eliminate values below the pivot.
        5. If requested, eliminate values above the pivot as well.
    """
    work = np.asarray(matrix, dtype=float).copy()
    rows, cols = work.shape
    pivot_row = 0
    pivot_columns_found: list[int] = []
    max_pivot_columns = cols if pivot_columns is None else min(pivot_columns, cols)

    for pivot_col in range(max_pivot_columns):
        if pivot_row >= rows:
            break

        # Select the strongest available pivot in the current column.
        pivot_index = pivot_row + int(np.argmax(np.abs(work[pivot_row:, pivot_col])))
        pivot_value = work[pivot_index, pivot_col]
        if abs(pivot_value) < EPSILON:
            continue

        # Move the pivot row into position when another row has the best pivot.
        if pivot_index != pivot_row:
            work[[pivot_row, pivot_index]] = work[[pivot_index, pivot_row]]

        # Normalize the pivot row so the pivot becomes 1.
        pivot_value = work[pivot_row, pivot_col]
        work[pivot_row] = work[pivot_row] / pivot_value

        # Eliminate every entry below the pivot.
        for row in range(pivot_row + 1, rows):
            factor = work[row, pivot_col]
            if abs(factor) > EPSILON:
                work[row] = work[row] - factor * work[pivot_row]

        # For reduced row echelon form, clear the entries above the pivot too.
        if reduce_above:
            for row in range(pivot_row):
                factor = work[row, pivot_col]
                if abs(factor) > EPSILON:
                    work[row] = work[row] - factor * work[pivot_row]

        pivot_columns_found.append(pivot_col)
        pivot_row += 1

    return clean_matrix(work), pivot_columns_found


def row_echelon_form(matrix: np.ndarray, pivot_columns: int | None = None) -> tuple[np.ndarray, list[int]]:
    """Return the row echelon form of a matrix.

    Args:
        matrix: Matrix to reduce.
        pivot_columns: Optional limit for pivot search.

    Returns:
        The echelon form and the list of pivot columns found.
    """
    return _row_reduce(matrix, reduce_above=False, pivot_columns=pivot_columns)


def reduced_row_echelon_form(matrix: np.ndarray, pivot_columns: int | None = None) -> tuple[np.ndarray, list[int]]:
    """Return the reduced row echelon form of a matrix.

    Args:
        matrix: Matrix to reduce.
        pivot_columns: Optional limit for pivot search.

    Returns:
        The reduced echelon form and the list of pivot columns found.
    """
    return _row_reduce(matrix, reduce_above=True, pivot_columns=pivot_columns)


def extract_solution_from_rref(reduced_matrix: np.ndarray, variable_count: int) -> np.ndarray:
    """Extract a solution vector from a reduced augmented matrix.

    Args:
        reduced_matrix: Augmented matrix already in reduced row echelon form.
        variable_count: Number of variables in the linear system.

    Returns:
        A vector with the solution values ordered by variable index.
    """
    solution = np.zeros(variable_count, dtype=float)

    for row in range(reduced_matrix.shape[0]):
        coefficients = reduced_matrix[row, :variable_count]
        pivot_candidates = np.where(np.abs(coefficients) > EPSILON)[0]
        if pivot_candidates.size == 0:
            continue

        pivot_column = int(pivot_candidates[0])
        solution[pivot_column] = reduced_matrix[row, -1]

    return clean_matrix(solution)