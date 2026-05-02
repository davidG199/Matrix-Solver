"""Funciones reutilizables para validar, normalizar y reducir matrices.

Este modulo centraliza la limpieza de entradas, la construccion de matrices
aumentadas y la reduccion por filas usada por las operaciones algebraicas.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Un umbral para tratar residuos numéricos como ceros exactos.
EPSILON = 1e-10


"""Determina si un valor es un escalar numérico real.

    Args:
        value: Cualquier valor candidato que debe ser inspeccionado.

    Returns:
        True cuando el valor es un int, float o tipo numérico de numpy, excluyendo
        booleanos.
"""
def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


"""Convierte un valor a float después de validar que es numérico.

Args:
    value: El valor a convertir.
    context: Etiqueta legible para humanos usada en mensajes de error.

    Returns:
        El valor numérico convertido a float.

    Raises:
        ValueError: Si el valor no es numérico o no es finito.
    """
def _coerce_number(value: Any, context: str) -> float:
    if not _is_numeric(value):
        raise ValueError(f"{context} debe contener solo numeros reales.")

    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{context} contiene valores no finitos.")
    return number


"""Normaliza una entrada similar a una matriz en un array de numpy 2D de floats.

    Args:
        matrix: Listas anidadas o un array de numpy existente.
        name: Etiqueta usada en mensajes de error para identificar la entrada.

    Returns:
        Un ndarray rectangular 2D con dtype float.

    Raises:
        ValueError: Si la entrada está vacía, es irregular, no es numérica o no es 2D.

    Proceso:
        1. Acepta arrays de numpy directamente cuando ya son 2D.
        2. Acepta listas/tuplas anidadas como filas de la matriz.
        3. Valida la longitud de cada fila y cada valor escalar.
        4. Convierte los datos a un array de float listo para operaciones algebraicas.
"""
def normalize_matrix(matrix: Any, name: str = "matriz") -> np.ndarray:
    if isinstance(matrix, np.ndarray): # Acepta arrays de numpy directamente si ya son 2D.
        data = np.asarray(matrix, dtype=float) # Convierte a float si es necesario.
        if data.ndim != 2: # Requiere que el array de numpy sea bidimensional.
            raise ValueError(f"{name} debe ser bidimensional.")
        if data.size == 0 or 0 in data.shape: # Rechaza matrices vacías o con filas/columnas vacías.
            raise ValueError(f"{name} no puede estar vacia.")
        if not np.all(np.isfinite(data)): # Rechaza valores no finitos como NaN o Inf.
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
    return data # retorna la matriz normalizada como un array de numpy 2D de floats.


"""Normaliza una entrada similar a un vector en un array de numpy 1D de floats.

    Args:
        vector: Una lista, tupla, array de numpy o equivalente a vector columna/fila.
        expected_size: Longitud exacta opcional que el vector debe satisfacer.
        name: Etiqueta usada en mensajes de error para identificar la entrada.

    Returns:
        Un ndarray 1D con dtype float.

    Raises:
        ValueError: Si la entrada está vacía, anidada de forma inválida o tiene
            el tamaño incorrecto.
"""
def normalize_vector(vector: Any, expected_size: int | None = None, name: str = "vector") -> np.ndarray:
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


"""Asegura que dos matrices compartan la misma forma.

    Args:
        matrix_a: Primera matriz a comparar.
        matrix_b: Segunda matriz a comparar.
        operation: Nombre de la operación algebraica que requiere igualdad.

    Raises:
        ValueError: Si ambas matrices no tienen las mismas dimensiones.
"""
def require_same_shape(matrix_a: np.ndarray, matrix_b: np.ndarray, operation: str) -> None:
    if matrix_a.shape != matrix_b.shape:
        raise ValueError(f"Las matrices deben tener la misma dimension para {operation}.")


"""Valida la regla de dimensiones para multiplicación de matrices.

    Args:
        matrix_a: Matriz izquierda.
        matrix_b: Matriz derecha.

    Raises:
        ValueError: Si el número de columnas de A no coincide con el número de
        filas de B.
"""
def require_multiplicable(matrix_a: np.ndarray, matrix_b: np.ndarray) -> None:
    if matrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError("No se pueden multiplicar: las columnas de A deben coincidir con las filas de B.")


"""Asegura que una matriz es cuadrada.

    Args:
        matrix: Matriz a validar.
        name: Etiqueta usada en mensajes de error.

    Raises:
        ValueError: Si la matriz no es n x n.
    """
def require_square(matrix: np.ndarray, name: str = "matriz") -> None:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} debe ser cuadrada.")


"""Reemplaza residuos flotantes minúsculos con ceros exactos.

    Args:
        matrix: Matriz a limpiar después de una operación numérica.

    Returns:
        Una copia de la matriz donde valores con valor absoluto menor que EPSILON
        se establecen en cero.
 """
def clean_matrix(matrix: np.ndarray) -> np.ndarray:
    data = np.asarray(matrix, dtype=float).copy()
    data[np.abs(data) < EPSILON] = 0.0
    return data


"""Convierte una matriz a una lista de listas compatible con JSON.

    Args:
        matrix: Matriz numérica a serializar.

    Returns:
        Una representación de lista anidada de la matriz.
"""
def matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return clean_matrix(matrix).tolist()


"""Convierte un vector a una lista plana compatible con JSON.

    Args:
        vector: Vector numérico a serializar.

    Returns:
        Una lista de Python simple de floats.
    """
def vector_to_list(vector: np.ndarray) -> list[float]:
    return clean_matrix(np.asarray(vector, dtype=float).reshape(-1)).tolist()


"""Construye la matriz aumentada [A | b].

    Args:
        matrix_a: Matriz de coeficientes.
        vector_b: Vector de término independiente.

    Returns:
        La matriz aumentada obtenida añadiendo b como la última columna.
    """
def build_augmented_matrix(matrix_a: np.ndarray, vector_b: np.ndarray) -> np.ndarray:
    return np.hstack([matrix_a, vector_b.reshape(-1, 1)])


"""Calcula el rango de álgebra lineal de una matriz.

    Args:
        matrix: Matriz numérica cuyo rango será calculado.

    Returns:
        El rango como un entero.
    """
def matrix_rank(matrix: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(np.asarray(matrix, dtype=float)))


"""Clasifica un sistema lineal usando los rangos de A y [A | b].

    Args:
        matrix_a: Matriz de coeficientes.
        vector_b: Vector de término independiente.

    Returns:
        Una tupla conteniendo el estado, rango de A, rango de la matriz aumentada
        y la matriz aumentada en sí.

    Valores de estado:
        incompatible: sin solución.
        compatible_determinado: solución única.
        compatible_indeterminado: infinitas soluciones.
    """
def classify_system(matrix_a: np.ndarray, vector_b: np.ndarray) -> tuple[str, int, int, np.ndarray]:
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


"""Reduce una matriz usando eliminación de Gauss con pivoteo parcial.

    Args:
        matrix: Matriz a reducir.
        reduce_above: Cuando es True, también elimina entradas arriba de cada pivote
            para producir forma escalonada reducida.
        pivot_columns: Límite opcional para el número de columnas a inspeccionar como
            pivotes.

    Returns:
        Una tupla con la matriz reducida y la lista de columnas pivote encontradas.

    Proceso:
        1. Busca el mejor pivote en la columna actual usando el valor absoluto más
           grande disponible.
        2. Intercambia filas cuando es necesario para colocar el pivote en la fila activa.
        3. Normaliza la fila pivote para que el pivote sea 1.
        4. Elimina valores debajo del pivote.
        5. Si se solicita, también elimina valores arriba del pivote.
    """
def _row_reduce(matrix: np.ndarray, reduce_above: bool, pivot_columns: int | None = None) -> tuple[np.ndarray, list[int]]:
    work = np.asarray(matrix, dtype=float).copy()
    rows, cols = work.shape
    pivot_row = 0
    pivot_columns_found: list[int] = []
    max_pivot_columns = cols if pivot_columns is None else min(pivot_columns, cols)

    for pivot_col in range(max_pivot_columns):
        if pivot_row >= rows:
            break

        # Selecciona el pivote más fuerte disponible en la columna actual.
        pivot_index = pivot_row + int(np.argmax(np.abs(work[pivot_row:, pivot_col])))
        pivot_value = work[pivot_index, pivot_col]
        if abs(pivot_value) < EPSILON:
            continue

        # Mueve la fila pivote a su posición cuando otra fila tiene el mejor pivote.
        if pivot_index != pivot_row:
            work[[pivot_row, pivot_index]] = work[[pivot_index, pivot_row]]

        # Normaliza la fila pivote para que el pivote sea 1.
        pivot_value = work[pivot_row, pivot_col]
        work[pivot_row] = work[pivot_row] / pivot_value

        # Elimina cada entrada debajo del pivote.
        for row in range(pivot_row + 1, rows):
            factor = work[row, pivot_col]
            if abs(factor) > EPSILON:
                work[row] = work[row] - factor * work[pivot_row]

        # Para forma escalonada reducida, también limpia las entradas arriba del pivote.
        if reduce_above:
            for row in range(pivot_row):
                factor = work[row, pivot_col]
                if abs(factor) > EPSILON:
                    work[row] = work[row] - factor * work[pivot_row]

        pivot_columns_found.append(pivot_col)
        pivot_row += 1

    return clean_matrix(work), pivot_columns_found


"""Devuelve la forma escalonada por filas de una matriz.

    Args:
        matrix: Matriz a reducir.
        pivot_columns: Límite opcional para la búsqueda de pivotes.

    Returns:
        La forma escalonada y la lista de columnas pivote encontradas.
    """
def row_echelon_form(matrix: np.ndarray, pivot_columns: int | None = None) -> tuple[np.ndarray, list[int]]:
    return _row_reduce(matrix, reduce_above=False, pivot_columns=pivot_columns)

"""Devuelve la forma escalonada reducida por filas de una matriz.

    Args:
        matrix: Matriz a reducir.
        pivot_columns: Límite opcional para la búsqueda de pivotes.

    Returns:
        La forma escalonada reducida y la lista de columnas pivote encontradas.
    """
def reduced_row_echelon_form(matrix: np.ndarray, pivot_columns: int | None = None) -> tuple[np.ndarray, list[int]]:
    return _row_reduce(matrix, reduce_above=True, pivot_columns=pivot_columns)


"""Extrae la solución de un sistema ya en forma escalonada reducida.

    Args:
        reduced_matrix: Matriz aumentada ya en forma escalonada reducida.
        variable_count: Número de variables en el sistema lineal.

    Returns:
        Un vector con los valores de solución ordenados por índice de variable.
    """
def extract_solution_from_rref(reduced_matrix: np.ndarray, variable_count: int) -> np.ndarray:
    solution = np.zeros(variable_count, dtype=float)

    for row in range(reduced_matrix.shape[0]):
        coefficients = reduced_matrix[row, :variable_count]
        pivot_candidates = np.where(np.abs(coefficients) > EPSILON)[0]
        if pivot_candidates.size == 0:
            continue

        pivot_column = int(pivot_candidates[0])
        solution[pivot_column] = reduced_matrix[row, -1]

    return clean_matrix(solution)