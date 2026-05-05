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


"""Valida dos matrices y devuelve su suma.

	Args:
		matrix_a: Entrada de la matriz izquierda.
		matrix_b: Entrada de la matriz derecha.

	Returns:
		Un diccionario con el nombre de la operación y la matriz resultante.

	Raises:
		ValueError: Si las matrices no son numéricas o no comparten la misma
		forma.

	Proceso:
		1. Normaliza ambas matrices en arrays de float.
		2. Valida que ambas formas sean idénticas.
		3. Suma elemento por elemento.
		4. Convierte el resultado a una lista amigable con JSON.
"""
def sum_matrices(matrix_a: Any, matrix_b: Any) -> Dict[str, Any]:
	left = normalize_matrix(matrix_a, "matriz A")
	right = normalize_matrix(matrix_b, "matriz B")
	require_same_shape(left, right, "suma")
	# La suma de matrices es una operación directa que se puede realizar con el operador + en numpy.
	# se agarra la primera matriz normalizada y se le suma la segunda matriz normalizada, pero primero se validan que tengan la misma forma
	return {"operation": "suma", "result": matrix_to_list(left + right)}


"""Valida dos matrices y devuelve su diferencia.

	Args:
		matrix_a: Matriz minuendo.
		matrix_b: Matriz sustraendo.

	Returns:
		Un diccionario con el nombre de la operación y la matriz resultante.

	Raises:
		ValueError: Si las matrices no son numéricas o no comparten la misma
		forma.

	Process:
		1. Normaliza ambas matrices.
		2. Confirma que las dimensiones sean iguales.
		3. Resta elemento por elemento de cada matriz.
		4. Serializa el resultado para las respuestas de la API.
"""
def subtract_matrices(matrix_a: Any, matrix_b: Any) -> Dict[str, Any]:
	left = normalize_matrix(matrix_a, "matriz A")
	right = normalize_matrix(matrix_b, "matriz B")
	require_same_shape(left, right, "resta")
	# La resta de matrices es una operación directa que se puede realizar con el operador - en numpy.
	# se agarra la primera matriz normalizada y se le resta la segunda matriz normalizada
	return {"operation": "resta", "result": matrix_to_list(left - right)}


"""Valida dos matrices y devuelve su producto.

	Args:
		matrix_a: Matriz izquierda.
		matrix_b: Matriz derecha.

	Returns:
		Un diccionario con el nombre de la operación y la matriz producto.

	Raises:
		ValueError: Si las matrices no pueden ser multiplicadas bajo la regla
		columnas(A) = filas(B).

	Proceso:
		1. Normaliza ambas matrices.
		2. Valida compatibilidad de dimensiones.
		3. Multiplica usando numpy.matmul.
		4. Convierte el resultado a listas.
"""
def multiply_matrices(matrix_a: Any, matrix_b: Any) -> Dict[str, Any]:
	left = normalize_matrix(matrix_a, "matriz A")
	right = normalize_matrix(matrix_b, "matriz B")
	require_multiplicable(left, right)
	return {"operation": "multiplicacion", "result": matrix_to_list(np.matmul(left, right))}


"""Devuelve la traspuesta de una matriz validada.

	Args:
		matrix: Matriz a trasponerion.

	Returns:
		Un diccionario con el nombre de la operación y la matriz traspuesta.

	Proceso:
		1. Normaliza la matriz.
		2. Intercambia filas y columnas con el operador traspuesta.
		3. Serializa el resultado para la API.
	"""
def transpose_matrix(matrix: Any) -> Dict[str, Any]:
	value = normalize_matrix(matrix, "matriz")
	return {"operation": "traspuesta", "result": matrix_to_list(value.T)}


"""Calcula el determinante usando el método que se ajusta al tamaño de la matriz.

	Args:
		matrix: Matriz cuadrada cuyo determinante será calculado.

	Returns:
		Un diccionario con el nombre de la operación, el método utilizado y el
		valor escalar del determinante.

	Raises:
		ValueError: Si la matriz no es cuadrada.

	Proceso:
		1. Normaliza y valida la matriz.
		2. Usa fórmulas directas para matrices 1x1 y 2x2.
		3. Usa Sarrus para matrices 3x3.
		4. Usa eliminación con pivoteo parcial para matrices más grandes.
	"""
def determinant(matrix: Any) -> Dict[str, Any]:
	print(matrix)
	value = normalize_matrix(matrix, "matriz") # Normaliza la entrada a un array de numpy 2D de floats.
	require_square(value, "matriz") # Verifica que la matriz es cuadrada, de lo contrario lanza un ValueError.

	size = value.shape[0] # Obtiene el tamaño de la matriz (número de filas o columnas, ya que es cuadrada).
	if size == 1:
		det_value = float(value[0, 0]) # Para una matriz 1x1, el determinante es simplemente el valor del único elemento.
		method = "directo 1x1"
	elif size == 2:
		# Para una matriz 2x2, el determinante se calcula con la fórmula ad - bc.
		det_value = float(value[0, 0] * value[1, 1] - value[0, 1] * value[1, 0])
		method = "directo ad - bc"
	elif size == 3:
		# Para una matriz 3x3, se puede usar la regla de Sarrus para calcular el determinante.
		det_value = float(
			value[0, 0] * value[1, 1] * value[2, 2]
			+ value[0, 1] * value[1, 2] * value[2, 0]
			+ value[0, 2] * value[1, 0] * value[2, 1]
			- value[0, 2] * value[1, 1] * value[2, 0]
			- value[0, 0] * value[1, 2] * value[2, 1]
			- value[0, 1] * value[1, 0] * value[2, 2]
		)
		method = "sarrus 3x3"
	else:
		# Para matrices más grandes, calcula el determinante a través de eliminación.
		# Este metodo transforma la matriz a forma triangular superior y luego calcula su determinante con el producto de su diagonal principal
		work = value.astype(float).copy()
		sign = 1.0
		det_value = 1.0

		for pivot_col in range(size):
			#Busca el pivote mas grande
			pivot_row = pivot_col + int(np.argmax(np.abs(work[pivot_col:, pivot_col])))
			pivot_value = work[pivot_row, pivot_col]
			# Si no hay pivote valido entonces determinante = 0 (matriz singular)
			if abs(pivot_value) < 1e-10:
				return {"operation": "determinante", "method": "eliminacion", "result": 0.0}

			# Intercambio de filas si es necesario (se invierte el signo)
			if pivot_row != pivot_col:
				work[[pivot_col, pivot_row]] = work[[pivot_row, pivot_col]]
				sign *= -1.0 #cada intercambio multiplica det por -1

			# Acumula los valores pivote antes de eliminar las filas de abajo.
			pivot_value = work[pivot_col, pivot_col]
			det_value *= pivot_value

			# Elimina hacia abajo a forma triangular
			for row in range(pivot_col + 1, size):
				factor = work[row, pivot_col] / pivot_value
				work[row, pivot_col:] = work[row, pivot_col:] - factor * work[pivot_col, pivot_col:]

		# aplica el signo
		det_value *= sign
		method = "eliminacion"

	return {"operation": "determinante", "method": method, "result": float(det_value)}


"""Calcula la inversa usando el método más apropiado para el tamaño.

	Args:
		matrix: Matriz cuadrada a invertir.

	Returns:
		Un diccionario con el nombre de la operación, el método utilizado y la
		matriz inversa.

	Raises:
		ValueError: Si la matriz no es cuadrada o si es singular.

	Proceso:
		1. Normaliza y valida la matriz.
		2. Rechaza matrices singulares temprano comprobando el determinante.
		3. Usa fórmulas directas para matrices 1x1 y 2x2.
		4. Para matrices más grandes, construye [A | I] y la reduce con Gauss-Jordan.
	"""
def inverse(matrix: Any) -> Dict[str, Any]:
	value = normalize_matrix(matrix, "matriz")
	require_square(value, "matriz")

	size = value.shape[0]
	det_data = np.linalg.det(value)
	if abs(det_data) < 1e-10:
		raise ValueError("La matriz no tiene inversa porque su determinante es 0.")

	if size == 1:
		result = np.array([[1.0 / value[0, 0]]], dtype=float)
		method = "directo 1x1"
	elif size == 2:
		det_value = value[0, 0] * value[1, 1] - value[0, 1] * value[1, 0]
		if abs(det_value) < 1e-10:
			raise ValueError("La matriz no tiene inversa porque su determinante es 0.")
		result = (1.0 / det_value) * np.array(
			[[value[1, 1], -value[0, 1]], [-value[1, 0], value[0, 0]]],
			dtype=float,
		)
		method = "directo 2x2"
	else:
		# Aumenta con la matriz identidad para aislar la inversa a la derecha.
		# value = matriz ingresada por el usuario y normalizada -- np.eye(size) = matriz identidad del mismo tamaño que value
		augmented = np.hstack([value, np.eye(size, dtype=float)]) 
		#reducimos la matriz a forma escalonada reducida
		reduced, _ = reduced_row_echelon_form(augmented, pivot_columns=size)
		# Después de la reducción, la parte izquierda debería ser la identidad y la parte derecha será la inversa.
		left = reduced[:, :size]
		if not np.allclose(left, np.eye(size), atol=1e-8):
			raise ValueError("La matriz no tiene inversa.")
		result = reduced[:, size:]
		method = "gauss jordan"

	return {"operation": "inversa", "method": method, "result": matrix_to_list(result)}


"""Normaliza un sistema A x = b y construye su matriz aumentada.

	Args:
		matrix_a: Matriz de coeficientes.
		vector_b: Vector de término independiente.

	Returns:
		Una tupla conteniendo la A normalizada, la b normalizada y la matriz
		aumentada [A | b].
	"""
def _validate_system(matrix_a: Any, vector_b: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	left = normalize_matrix(matrix_a, "matriz A")
	right = normalize_vector(vector_b, left.shape[0], "vector b")
	augmented = build_augmented_matrix(left, right)
	return left, right, augmented

"""Resuelve o analiza un sistema lineal usando eliminación de Gauss.

	Args:
		matrix_a: Matriz de coeficientes.
		vector_b: Vector de término independiente.

	Returns:
		Un diccionario con el nombre de la operación, estado del sistema, rangos,
		matriz escalonada y, cuando la solución es única, el vector solución.

	Proceso:
		1. Normaliza el sistema.
		2. Construye la matriz aumentada [A | b].
		3. La reduce a forma escalonada por filas con pivotes normalizados a 1.
		4. Compara rangos para clasificar el sistema.
		5. Si la solución es única, la recupera de la forma reducida.
	"""
def gauss(matrix_a: Any, vector_b: Any) -> Dict[str, Any]:
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

"""Resuelve o analiza un sistema lineal usando eliminación de Gauss-Jordan.

	Args:
		matrix_a: Matriz de coeficientes.
		vector_b: Vector de término independiente.

	Returns:
		Un diccionario con el nombre de la operación, estado del sistema, rangos,
		matriz reducida y, cuando es posible, el vector solución.

	Proceso:
		1. Normaliza el sistema.
		2. Construye la matriz aumentada [A | b].
		3. La reduce a forma escalonada reducida por filas.
		4. Clasifica el sistema comparando rangos.
		5. Lee la solución directamente de la matriz reducida cuando es única.
	"""
def gauss_jordan(matrix_a: Any, vector_b: Any) -> Dict[str, Any]:
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


"""Resuelve un sistema lineal con la regla de Cramer.

	Args:
		matrix_a: Matriz de coeficientes.
		vector_b: Vector de término independiente.

	Returns:
		Un diccionario con el nombre de la operación, determinante de A y el
		vector solución.

	Raises:
		ValueError: Si A no es cuadrada o si det(A) es cero.

	Proceso:
		1. Normaliza A y b.
		2. Verifica que A es cuadrada e invertible.
		3. Reemplaza cada columna de A por b una a la vez.
		4. Calcula x_i = det(A_i) / det(A) para cada variable.
	"""
def cramer(matrix_a: Any, vector_b: Any) -> Dict[str, Any]:
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
