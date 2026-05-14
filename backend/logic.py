import numpy as np
from fractions import Fraction

def formatear_fraccion(valor):
    if isinstance(valor, Fraction):
        if valor.denominator == 1:
            return str(valor.numerator)
        return f"{valor.numerator}/{valor.denominator}"
    
    try:
        # Redondear un poco para evadir residuos como 0.3333333333333333
        valor_float = round(float(valor), 7)
        # Convertir a fraccion y limitar el denominador
        frac = Fraction(valor_float).limit_denominator(1000)
        if frac.denominator == 1:
            return str(frac.numerator)
        return f"{frac.numerator}/{frac.denominator}"
    except:
        return str(valor)

# esta funcion formatea cada elementos de la matriz para que quede como fraccion y sea mas legible
def formatear_matriz(matriz):
    return [[formatear_fraccion(celda) for celda in fila] for fila in matriz]

def crear_paso(titulo, descripcion, matriz=None):
    paso = {"title": titulo, "description": descripcion}
    if matriz is not None:
        paso["matrix"] = formatear_matriz(matriz)
    return paso

def parse_to_fraction_array(matrix):
    return np.array([[Fraction(str(x)).limit_denominator(1000) for x in row] for row in matrix], dtype=object)

def parse_to_fraction_vector(vector):
    return np.array([Fraction(str(x)).limit_denominator(1000) for x in vector], dtype=object)

def sum_matrices(matrix_a, matrix_b):
    A = np.array(matrix_a, dtype=float)
    B = np.array(matrix_b, dtype=float)
    R = A + B
    pasos = [crear_paso("Resultado de la Suma", "Se suma cada elemento de la matriz A con la correspondiente de la matriz B.", R)]
    return {"operation": "suma", "result": formatear_matriz(R), "steps": pasos}

def subtract_matrices(matrix_a, matrix_b):
    A = np.array(matrix_a, dtype=float)
    B = np.array(matrix_b, dtype=float)
    R = A - B
    pasos = [crear_paso("Resultado de la Resta", "Se resta a cada elemento de la matriz A, el correspondiente en la matriz B.", R)]
    return {"operation": "resta", "result": formatear_matriz(R), "steps": pasos}

def multiply_matrices(matrix_a, matrix_b):
    A = np.array(matrix_a, dtype=float)
    B = np.array(matrix_b, dtype=float)
    R = np.dot(A, B)
    pasos = [crear_paso("Resultado de la Multiplicación", "Se multiplica cada fila de A por cada columna de B.", R)]
    return {"operation": "multiplicacion", "result": formatear_matriz(R), "steps": pasos}

def transpose_matrix(matrix):
    A = np.array(matrix, dtype=float)
    R = np.transpose(A)
    pasos = [crear_paso("Resultado de la Traspuesta", "Las filas pasan a ser columnas y las columnas pasan a ser filas.", R)]
    return {"operation": "traspuesta", "result": formatear_matriz(R), "steps": pasos}

def determinant(matrix):
    A = np.array(matrix, dtype=float)
    d = np.linalg.det(A)
    R = np.array([[d]])
    pasos = [crear_paso("Determinante", "Se calcula el determinante de la matriz cuadrada.", R)]
    return {"operation": "determinante", "result": formatear_matriz(R), "steps": pasos}

def inverse(matrix):
    A = np.array(matrix, dtype=float)
    d = np.linalg.det(A)
    if np.isclose(d, 0):
        raise ValueError("La matriz no tiene inversa (El determinante es cero).")
    R = np.linalg.inv(A)
    pasos = [crear_paso("Matriz Inversa", "Usando el método genérico (1/det * Matriz Adjunta) se obtiene la inversa.", R)]
    return {"operation": "inversa", "result": formatear_matriz(R), "steps": pasos}

def gauss(matrix_a, vector_b):
    A = parse_to_fraction_array(matrix_a)
    b = parse_to_fraction_vector(vector_b).reshape(-1, 1)
    M = np.hstack([A, b])
    
    pasos = [crear_paso("Matriz Aumentada [A | b]", "Representación del sistema de ecuaciones en matriz aumentada.", M)]
    
    filas, cols = M.shape
    rango_pivotes = min(filas, cols - 1)

    for i in range(rango_pivotes):
        # 1. Asegurar que el pivote no sea 0
        if M[i][i] == 0:
            for k in range(i+1, filas):
                if M[k][i] != 0:
                    M[[i, k]] = M[[k, i]]
                    pasos.append(crear_paso("Intercambio de filas", f"Fila {i+1} ↔ Fila {k+1}", np.copy(M)))
                    break
            else:
                continue # No hay pivote en esta columna
                
        # 2. Hacer el pivote = 1
        pivote = M[i][i]
        if pivote != 1 and pivote != 0:
            M[i] = M[i] / pivote
            texto_multiplicador = formatear_fraccion(1/pivote)
            pasos.append(crear_paso(f"Pivote 1 (Fila {i+1})", f"F{i+1} = F{i+1} * ({texto_multiplicador})", np.copy(M)))
            
        # 3. Hacer 0 todo lo que está POR DEBAJO
        for j in range(i+1, filas):
            factor = M[j][i]
            if factor != 0:
                M[j] = M[j] - factor * M[i]
                tfactor = formatear_fraccion(factor)
                pasos.append(crear_paso(f"Ceros debajo (Fila {j+1})", f"F{j+1} = F{j+1} - ({tfactor}) * F{i+1}", np.copy(M)))

    return {"operation": "gauss", "result": formatear_matriz(M), "steps": pasos}

def gauss_jordan(matrix_a, vector_b):
    A = parse_to_fraction_array(matrix_a)
    b = parse_to_fraction_vector(vector_b).reshape(-1, 1)
    M = np.hstack([A, b])
    
    pasos = [crear_paso("Matriz Aumentada [A | b]", "Representación del sistema de ecuaciones en matriz aumentada.", M)]
    
    filas, cols = M.shape
    rango_pivotes = min(filas, cols - 1)

    for i in range(rango_pivotes):
        # 1. Asegurar que el pivote no sea 0
        if M[i][i] == 0:
            for k in range(i+1, filas):
                if M[k][i] != 0:
                    M[[i, k]] = M[[k, i]]
                    pasos.append(crear_paso("Intercambio de filas", f"Fila {i+1} ↔ Fila {k+1}", np.copy(M)))
                    break
            else:
                continue
                
        # 2. Hacer el pivote = 1
        pivote = M[i][i]
        if pivote != 1 and pivote != 0:
            M[i] = M[i] / pivote
            texto_multiplicador = formatear_fraccion(1/pivote)
            pasos.append(crear_paso(f"Pivote 1 (Fila {i+1})", f"F{i+1} = F{i+1} * ({texto_multiplicador})", np.copy(M)))
            
        # 3. Hacer 0 lo que está por ENCIMA y POR DEBAJO
        for j in range(filas):
            if i != j:
                factor = M[j][i]
                if factor != 0:
                    M[j] = M[j] - factor * M[i]
                    tfactor = formatear_fraccion(factor)
                    direccion = "arriba" if j < i else "abajo"
                    pasos.append(crear_paso(f"Ceros a {direccion} (Fila {j+1})", f"F{j+1} = F{j+1} - ({tfactor}) * F{i+1}", np.copy(M)))

    return {"operation": "gauss-jordan", "result": formatear_matriz(M), "steps": pasos}

def cramer(matrix_a, vector_b):
    A = np.array(matrix_a, dtype=float)
    b = np.array(vector_b, dtype=float).reshape(-1, 1)
    
    detA = np.linalg.det(A)
    if np.isclose(detA, 0):
        # ValueError pasará al frontend con status code 400 segun main.py
        raise ValueError("El determinante de la matriz es 0, no se puede usar Cramer ya que no tiene solución única.")
        
    pasos = [crear_paso("Cálculo Determinante General", f"El determinante de A (Δ) dio como resultado: {formatear_fraccion(detA)}")]
    
    filas = A.shape[0]
    X_result = [] # Aquí vamos a guardar un vector columna con las respuestas
    
    for i in range(filas):
        Ai = np.copy(A)
        Ai[:, i] = b.flatten()
        detAi = np.linalg.det(Ai)
        xi = detAi / detA
        X_result.append([xi])
        
        descripcion = (
            f"Columna {i+1} ← vector b. "
            f"Δ_{i+1} = {formatear_fraccion(detAi)}. "
            f"x_{i+1} = {formatear_fraccion(detAi)} / {formatear_fraccion(detA)} = {formatear_fraccion(xi)}"
        )
        pasos.append(crear_paso(f"Variable x{i+1}", descripcion, np.copy(Ai)))
        
    return {
        "operation": "cramer", 
        "result": formatear_matriz(X_result), 
        "steps": pasos,
        "determinant": formatear_fraccion(detA),
        "solution": [formatear_fraccion(x[0]) for x in X_result]
    }
