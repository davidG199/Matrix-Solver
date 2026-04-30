from __future__ import annotations

from typing import Any, Dict

import numpy as np

from utils import (
	build_augmented_matrix,
	classify_system,
	extract_solution_from_rref,
	matrix_to_list,
	normalize_matrix,
	normalize_vector,
	reduced_row_echelon_form,
	require_multiplicable,
	require_same_shape,
	require_square,
	row_echelon_form,
	vector_to_list,
)


def sum_matrices(matrix_a: Any, matrix_b: Any) -> Dict[str, Any]:
	"""Validate two matrices and return their sum.

	Args:
		matrix_a: Left matrix input.
		matrix_b: Right matrix input.

	Returns:
		A dictionary with the operation name and the resulting matrix.

	Raises:
		ValueError: If the matrices are not numeric or do not share the same
		shape.

	Process:
		1. Normalize both matrices into float arrays.
		2. Validate that both shapes are identical.
		3. Add element by element.
		4. Convert the result to a JSON-friendly list.
	"""
	left = normalize_matrix(matrix_a, "matriz A")
	right = normalize_matrix(matrix_b, "matriz B")
	require_same_shape(left, right, "suma")
	return {"operation": "suma", "result": matrix_to_list(left + right)}


def subtract_matrices(matrix_a: Any, matrix_b: Any) -> Dict[str, Any]:
	"""Validate two matrices and return their difference.

	Args:
		matrix_a: Minuend matrix.
		matrix_b: Subtrahend matrix.

	Returns:
		A dictionary with the operation name and the resulting matrix.

	Raises:
		ValueError: If the matrices are not numeric or do not share the same
		shape.

	Process:
		1. Normalize both matrices.
		2. Confirm that the dimensions match.
		3. Subtract element by element.
		4. Serialize the result for API responses.
	"""
	left = normalize_matrix(matrix_a, "matriz A")
	right = normalize_matrix(matrix_b, "matriz B")
	require_same_shape(left, right, "resta")
	return {"operation": "resta", "result": matrix_to_list(left - right)}


def multiply_matrices(matrix_a: Any, matrix_b: Any) -> Dict[str, Any]:
	"""Validate two matrices and return their product.

	Args:
		matrix_a: Left matrix.
		matrix_b: Right matrix.

	Returns:
		A dictionary with the operation name and the product matrix.

	Raises:
		ValueError: If the matrices cannot be multiplied under the rule
		columns(A) = rows(B).

	Process:
		1. Normalize both matrices.
		2. Validate dimension compatibility.
		3. Multiply using numpy.matmul.
		4. Convert the result to lists.
	"""
	left = normalize_matrix(matrix_a, "matriz A")
	right = normalize_matrix(matrix_b, "matriz B")
	require_multiplicable(left, right)
	return {"operation": "multiplicacion", "result": matrix_to_list(np.matmul(left, right))}


def transpose_matrix(matrix: Any) -> Dict[str, Any]:
	"""Return the transpose of a validated matrix.

	Args:
		matrix: Matrix to transpose.

	Returns:
		A dictionary with the operation name and the transposed matrix.

	Process:
		1. Normalize the matrix.
		2. Swap rows and columns with the transpose operator.
		3. Serialize the result for the API.
	"""
	value = normalize_matrix(matrix, "matriz")
	return {"operation": "traspuesta", "result": matrix_to_list(value.T)}


def determinant(matrix: Any) -> Dict[str, Any]:
	"""Compute the determinant using the method that fits the matrix size.

	Args:
		matrix: Square matrix whose determinant will be calculated.

	Returns:
		A dictionary with the operation name, the method used and the scalar
		determinant value.

	Raises:
		ValueError: If the matrix is not square.

	Process:
		1. Normalize and validate the matrix.
		2. Use direct formulas for 1x1 and 2x2 matrices.
		3. Use Sarrus for 3x3 matrices.
		4. Use elimination with partial pivoting for larger matrices.
	"""
	value = normalize_matrix(matrix, "matriz")
	require_square(value, "matriz")

	size = value.shape[0]
	if size == 1:
		det_value = float(value[0, 0])
		method = "directo_1x1"
	elif size == 2:
		det_value = float(value[0, 0] * value[1, 1] - value[0, 1] * value[1, 0])
		method = "directo_2x2"
	elif size == 3:
		det_value = float(
			value[0, 0] * value[1, 1] * value[2, 2]
			+ value[0, 1] * value[1, 2] * value[2, 0]
			+ value[0, 2] * value[1, 0] * value[2, 1]
			- value[0, 2] * value[1, 1] * value[2, 0]
			- value[0, 0] * value[1, 2] * value[2, 1]
			- value[0, 1] * value[1, 0] * value[2, 2]
		)
		method = "sarrus_3x3"
	else:
		# For larger matrices, compute the determinant through elimination.
		work = value.astype(float).copy()
		sign = 1.0
		det_value = 1.0

		for pivot_col in range(size):
			# Use partial pivoting to improve numerical stability.
			pivot_row = pivot_col + int(np.argmax(np.abs(work[pivot_col:, pivot_col])))
			pivot_value = work[pivot_row, pivot_col]
			if abs(pivot_value) < 1e-10:
				return {"operation": "determinante", "method": "eliminacion", "result": 0.0}

			# Every row swap flips the determinant sign.
			if pivot_row != pivot_col:
				work[[pivot_col, pivot_row]] = work[[pivot_row, pivot_col]]
				sign *= -1.0

			# Accumulate pivot values before eliminating the rows below.
			pivot_value = work[pivot_col, pivot_col]
			det_value *= pivot_value

			for row in range(pivot_col + 1, size):
				factor = work[row, pivot_col] / pivot_value
				work[row, pivot_col:] = work[row, pivot_col:] - factor * work[pivot_col, pivot_col:]

		det_value *= sign
		method = "eliminacion"

	return {"operation": "determinante", "method": method, "result": float(det_value)}


def inverse(matrix: Any) -> Dict[str, Any]:
	"""Compute the inverse using the most appropriate method for the size.

	Args:
		matrix: Square matrix to invert.

	Returns:
		A dictionary with the operation name, the method used and the inverse
		matrix.

	Raises:
		ValueError: If the matrix is not square or if it is singular.

	Process:
		1. Normalize and validate the matrix.
		2. Reject singular matrices early by checking the determinant.
		3. Use direct formulas for 1x1 and 2x2 matrices.
		4. For larger matrices, build [A | I] and reduce it with Gauss-Jordan.
	"""
	value = normalize_matrix(matrix, "matriz")
	require_square(value, "matriz")

	size = value.shape[0]
	det_data = np.linalg.det(value)
	if abs(det_data) < 1e-10:
		raise ValueError("La matriz no tiene inversa porque su determinante es 0.")

	if size == 1:
		result = np.array([[1.0 / value[0, 0]]], dtype=float)
		method = "directo_1x1"
	elif size == 2:
		det_value = value[0, 0] * value[1, 1] - value[0, 1] * value[1, 0]
		if abs(det_value) < 1e-10:
			raise ValueError("La matriz no tiene inversa porque su determinante es 0.")
		result = (1.0 / det_value) * np.array(
			[[value[1, 1], -value[0, 1]], [-value[1, 0], value[0, 0]]],
			dtype=float,
		)
		method = "directo_2x2"
	else:
		# Augment with the identity matrix to isolate the inverse on the right.
		augmented = np.hstack([value, np.eye(size, dtype=float)])
		reduced, _ = reduced_row_echelon_form(augmented, pivot_columns=size)
		left = reduced[:, :size]
		if not np.allclose(left, np.eye(size), atol=1e-8):
			raise ValueError("La matriz no tiene inversa.")
		result = reduced[:, size:]
		method = "gauss_jordan"

	return {"operation": "inversa", "method": method, "result": matrix_to_list(result)}


def _validate_system(matrix_a: Any, vector_b: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Normalize a system A x = b and build its augmented matrix.

	Args:
		matrix_a: Coefficient matrix.
		vector_b: Independent term vector.

	Returns:
		A tuple containing the normalized A, the normalized b and the
		augmented matrix [A | b].
	"""
	left = normalize_matrix(matrix_a, "matriz A")
	right = normalize_vector(vector_b, left.shape[0], "vector b")
	augmented = build_augmented_matrix(left, right)
	return left, right, augmented


def gauss(matrix_a: Any, vector_b: Any) -> Dict[str, Any]:
	"""Solve or analyze a linear system using Gaussian elimination.

	Args:
		matrix_a: Coefficient matrix.
		vector_b: Independent term vector.

	Returns:
		A dictionary with the operation name, system status, ranks, echelon
		matrix and, when the solution is unique, the solution vector.

	Process:
		1. Normalize the system.
		2. Build the augmented matrix [A | b].
		3. Reduce it to row echelon form with pivots normalized to 1.
		4. Compare ranks to classify the system.
		5. If the solution is unique, recover it from the reduced form.
	"""
	left, right, augmented = _validate_system(matrix_a, vector_b)
	echelon, _ = row_echelon_form(augmented, pivot_columns=left.shape[1])
	status, rank_a, rank_augmented, _ = classify_system(left, right)

	result: Dict[str, Any] = {
		"operation": "gauss",
		"status": status,
		"rank_a": rank_a,
		"rank_augmented": rank_augmented,
		"matrix": matrix_to_list(echelon),
	}

	if status == "compatible_determinado":
		reduced, _ = reduced_row_echelon_form(augmented, pivot_columns=left.shape[1])
		result["solution"] = vector_to_list(extract_solution_from_rref(reduced, left.shape[1]))
	else:
		result["solution"] = None

	return result


def gauss_jordan(matrix_a: Any, vector_b: Any) -> Dict[str, Any]:
	"""Solve or analyze a linear system using Gauss-Jordan elimination.

	Args:
		matrix_a: Coefficient matrix.
		vector_b: Independent term vector.

	Returns:
		A dictionary with the operation name, system status, ranks, reduced
		matrix and, when possible, the solution vector.

	Process:
		1. Normalize the system.
		2. Build the augmented matrix [A | b].
		3. Reduce it to reduced row echelon form.
		4. Classify the system by comparing ranks.
		5. Read the solution directly from the reduced matrix when it is unique.
	"""
	left, right, augmented = _validate_system(matrix_a, vector_b)
	reduced, _ = reduced_row_echelon_form(augmented, pivot_columns=left.shape[1])
	status, rank_a, rank_augmented, _ = classify_system(left, right)

	result: Dict[str, Any] = {
		"operation": "gauss_jordan",
		"status": status,
		"rank_a": rank_a,
		"rank_augmented": rank_augmented,
		"matrix": matrix_to_list(reduced),
	}

	if status == "compatible_determinado":
		result["solution"] = vector_to_list(extract_solution_from_rref(reduced, left.shape[1]))
	else:
		result["solution"] = None

	return result


def cramer(matrix_a: Any, vector_b: Any) -> Dict[str, Any]:
	"""Solve a linear system with Cramer's rule.

	Args:
		matrix_a: Coefficient matrix.
		vector_b: Independent term vector.

	Returns:
		A dictionary with the operation name, determinant of A and the
		solution vector.

	Raises:
		ValueError: If A is not square or if det(A) is zero.

	Process:
		1. Normalize A and b.
		2. Verify that A is square and invertible.
		3. Replace each column of A by b one at a time.
		4. Compute x_i = det(A_i) / det(A) for every variable.
	"""
	left, right, _ = _validate_system(matrix_a, vector_b)
	require_square(left, "matriz A")

	determinant_a = float(np.linalg.det(left))
	if abs(determinant_a) < 1e-10:
		raise ValueError("El metodo de Cramer requiere que det(A) sea distinto de 0.")

	solutions = []
	for column in range(left.shape[1]):
		modified = left.copy()
		modified[:, column] = right
		determinant_modified = float(np.linalg.det(modified))
		solutions.append(determinant_modified / determinant_a)

	return {
		"operation": "cramer",
		"determinant": determinant_a,
		"solution": vector_to_list(np.asarray(solutions, dtype=float)),
	}
