# Logica de operaciones con matrices

Este documento resume la logica matematica implementada en el backend para operar con matrices. Solo describe la parte algebraica y las validaciones asociadas a cada operacion.

## Convenciones generales

- Todas las entradas se normalizan antes de calcular.
- Las matrices deben llegar como listas de listas o como arreglos numericos equivalentes.
- Los vectores se validan como arreglos unidimensionales.
- No se aceptan valores no numericos, vacios o infinitos.
- Se usa un umbral `EPSILON = 1e-10` para limpiar numeros muy cercanos a cero y evitar ruido numerico.

## Validaciones reutilizables

Las funciones auxiliares de validacion trabajan como filtro previo para cada operacion:

- `normalize_matrix`: verifica que la entrada sea una matriz rectangular y numerica.
- `normalize_vector`: verifica que la entrada sea un vector real de longitud valida.
- `require_same_shape`: exige la misma dimension para suma y resta.
- `require_multiplicable`: exige que las columnas de `A` coincidan con las filas de `B`.
- `require_square`: exige matrices cuadradas para determinante, inversa y Cramer.
- `classify_system`: compara el rango de `A` con el rango de la matriz aumentada `[A|b]`.
- `row_echelon_form` y `reduced_row_echelon_form`: reducen matrices usando eliminacion gaussiana con pivotes normalizados a 1.

## Funciones de utils

Esta seccion explica la logica de cada funcion definida en `backend/utils.py`.

### `EPSILON`

- Es un umbral numerico fijo de `1e-10`.
- Se usa para decidir cuando un valor debe considerarse cero por efecto de redondeo.
- Ayuda a limpiar resultados y evitar ruido numerico en reducciones matriciales.

### `_is_numeric(value)`

- Verifica si un valor pertenece a un tipo numerico real aceptado.
- Acepta enteros, flotantes y tipos numericos de `numpy`.
- Excluye booleanos para evitar que `True` y `False` entren como datos validos.

### `_coerce_number(value, context)`

- Convierte un valor numerico a `float`.
- Si el valor no es numerico o no es finito, lanza un error descriptivo.
- Se usa para validar elemento por elemento cada fila de una matriz o cada posicion de un vector.

### `normalize_matrix(matrix, name)`

- Convierte la entrada en una matriz numerica de `numpy`.
- Acepta listas de listas o arreglos `numpy`.
- Exige que la matriz sea bidimensional, rectangular y no vacia.
- Rechaza filas con longitudes distintas.
- Rechaza valores no numericos, vacios o infinitos.
- Es la puerta de entrada para cualquier operacion sobre matrices.

### `normalize_vector(vector, expected_size, name)`

- Convierte la entrada en un vector numerico unidimensional.
- Acepta listas, tuplas o arreglos `numpy`.
- Si se envia como matriz columna o fila, la aplana a un vector.
- Verifica que tenga la longitud esperada cuando `expected_size` no es `None`.
- Rechaza estructuras bidimensionales reales, salvo el caso de un vector columna/fila equivalente.

### `require_same_shape(matrix_a, matrix_b, operation)`

- Comprueba que dos matrices tengan exactamente la misma forma.
- Si no coinciden, lanza un error indicando que la operacion no es valida.
- Se usa en suma y resta.

### `require_multiplicable(matrix_a, matrix_b)`

- Verifica la regla de compatibilidad para multiplicacion matricial.
- Exige que el numero de columnas de `A` sea igual al numero de filas de `B`.
- Si la condicion falla, la multiplicacion no puede realizarse.

### `require_square(matrix, name)`

- Verifica que una matriz tenga el mismo numero de filas y columnas.
- Se usa en determinante, inversa y Cramer.
- Si la matriz no es cuadrada, la operacion se rechaza.

### `clean_matrix(matrix)`

- Copia una matriz numerica y reemplaza valores muy pequenos por cero.
- Usa `EPSILON` como criterio de limpieza.
- Evita mostrar `-0.0` o residuos numericos irrelevantes despues de las reducciones.

### `matrix_to_list(matrix)`

- Convierte una matriz `numpy` a lista de listas.
- Aplica limpieza numerica antes de serializar.
- Se usa para devolver resultados JSON serializables.

### `vector_to_list(vector)`

- Convierte un vector `numpy` a una lista plana de numeros.
- Limpia residuos numericos antes de serializar.
- Se usa cuando la salida matematica debe enviarse como arreglo simple.

### `build_augmented_matrix(matrix_a, vector_b)`

- Construye la matriz aumentada `[A | b]`.
- Convierte el vector `b` en una columna y la concatena a la derecha de `A`.
- Es la estructura base para Gauss, Gauss-Jordan y la clasificacion del sistema.

### `matrix_rank(matrix)`

- Calcula el rango lineal de una matriz numerica.
- Usa `numpy.linalg.matrix_rank`.
- Sirve para decidir si un sistema tiene una, ninguna o infinitas soluciones.

### `classify_system(matrix_a, vector_b)`

- Calcula la matriz aumentada `[A | b]`.
- Compara `rank(A)` con `rank([A | b])`.
- Clasifica el sistema como:
  - `incompatible`: no tiene solucion.
  - `compatible_determinado`: tiene solucion unica.
  - `compatible_indeterminado`: tiene infinitas soluciones.
- Devuelve tambien ambos rangos y la matriz aumentada para reutilizar esos datos en el endpoint.

### `_row_reduce(matrix, reduce_above, pivot_columns)`

- Es el motor interno de la reduccion por filas.
- Copia la matriz de entrada y recorre columnas buscando pivotes.
- Usa pivote parcial: selecciona el mayor valor absoluto disponible en la columna actual.
- Intercambia filas cuando hace falta para posicionar el pivote.
- Normaliza cada pivote a 1 dividiendo toda la fila.
- Elimina los terminos por debajo del pivote en todos los casos.
- Si `reduce_above` es `True`, tambien elimina los terminos por encima del pivote.
- Devuelve la matriz reducida y la lista de columnas pivote encontradas.

### `row_echelon_form(matrix, pivot_columns)`

- Expone la reduccion hacia forma escalonada.
- Es un envoltorio de `_row_reduce` con `reduce_above=False`.
- Se usa para Gauss.

### `reduced_row_echelon_form(matrix, pivot_columns)`

- Expone la reduccion hacia forma escalonada reducida.
- Es un envoltorio de `_row_reduce` con `reduce_above=True`.
- Se usa para Gauss-Jordan e inversa.

### `extract_solution_from_rref(reduced_matrix, variable_count)`

- Lee una solucion desde una matriz en forma escalonada reducida.
- Busca la primera columna no nula en cada fila para identificar el pivote.
- Toma el valor de la columna independiente como valor de la variable pivote.
- Devuelve un vector con la solucion ordenada por variable.

## Relacion entre utils y las operaciones

- `normalize_matrix` y `normalize_vector` garantizan que ninguna operacion reciba basura numerica.
- `require_same_shape`, `require_multiplicable` y `require_square` aplican las restricciones algebraicas basicas.
- `build_augmented_matrix`, `classify_system`, `row_echelon_form` y `reduced_row_echelon_form` soportan Gauss, Gauss-Jordan y Cramer.
- `clean_matrix`, `matrix_to_list` y `vector_to_list` se encargan de preparar salidas limpias para la API.

## Suma de matrices

### Regla

Dos matrices solo se pueden sumar si tienen exactamente la misma forma.

### Logica

- Se valida que `A.shape == B.shape`.
- Se suma elemento por elemento.
- El resultado conserva la misma dimension que las matrices originales.

### Salida

- Matriz resultado de la suma.

## Resta de matrices

### Regla

Dos matrices solo se pueden restar si tienen exactamente la misma forma.

### Logica

- Se valida que `A.shape == B.shape`.
- Se resta elemento por elemento.
- El resultado conserva la misma dimension que las matrices originales.

### Salida

- Matriz resultado de la resta.

## Multiplicacion de matrices

### Regla

La multiplicacion `A x B` solo existe si el numero de columnas de `A` es igual al numero de filas de `B`.

### Logica

- Se valida la compatibilidad interna `A.cols == B.rows`.
- Se calcula el producto matricial con la regla fila por columna.

### Salida

- Matriz resultado de `A @ B`.

## Matriz traspuesta

### Regla

Toda matriz valida tiene traspuesta.

### Logica

- Se intercambian filas por columnas.
- Si `A` tiene forma `m x n`, su traspuesta tiene forma `n x m`.

### Salida

- Matriz traspuesta `A.T`.

## Determinante

### Regla

Solo se calcula para matrices cuadradas.

### Metodos usados segun el tamano

#### Matriz 1x1

- El determinante es el unico elemento.

#### Matriz 2x2

- Se usa la formula directa:

$$
\det(A) = ad - bc
$$

#### Matriz 3x3

- Se usa la regla de Sarrus.

#### Matriz mayor a 3x3

- Se usa eliminacion gaussiana con pivote parcial.
- Se intercambian filas cuando el pivote lo requiere.
- Cada intercambio cambia el signo del determinante.
- El determinante final es el producto de los pivotes ajustado por el signo de los intercambios.
- Si aparece un pivote cercano a cero, el determinante se considera cero.

### Salida

- Valor escalar del determinante.
- Metodo usado: `directo_1x1`, `directo_2x2`, `sarrus_3x3` o `eliminacion`.

## Inversa

### Regla

Solo existe para matrices cuadradas no singulares.

### Validacion previa

- La matriz debe ser cuadrada.
- El determinante debe ser distinto de cero.

### Metodos usados segun el tamano

#### Matriz 1x1

- La inversa es el reciproco del unico valor.

#### Matriz 2x2

- Se usa la formula clasica con adjunta:

$$
A^{-1} = \frac{1}{ad-bc}
\begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}
$$

#### Matriz mayor a 2x2

- Se construye la matriz aumentada `[A | I]`.
- Se aplica Gauss-Jordan hasta convertir la parte izquierda en la identidad.
- Si la parte izquierda no llega a ser identidad, se rechaza la inversa.

### Salida

- Matriz inversa.
- Metodo usado: `directo_1x1`, `directo_2x2` o `gauss_jordan`.

## Gauss

### Objetivo

Llevar un sistema lineal a forma escalonada para analizar su consistencia y, si tiene solucion unica, extraerla.

### Entrada esperada

- Matriz de coeficientes `A`.
- Vector de terminos independientes `b`.

### Logica

- Se valida que `b` tenga tantas filas como `A`.
- Se forma la matriz aumentada `[A | b]`.
- Se calcula la forma escalonada usando eliminacion hacia abajo.
- Cada pivote se normaliza a 1.
- Se usa `classify_system` para comparar rangos:
  - `rank(A) != rank([A|b])`: sistema incompatible.
  - `rank(A) == rank([A|b]) == numero_de_variables`: sistema compatible determinado.
  - `rank(A) == rank([A|b]) < numero_de_variables`: sistema compatible indeterminado.

### Si la solucion es unica

- Se recalcula la forma reducida.
- Se extrae la solucion leyendo la columna independiente de cada pivote.

### Salida

- Estado del sistema.
- Rangos de `A` y `[A|b]`.
- Matriz escalonada.
- Solucion cuando es unica.

## Gauss-Jordan

### Objetivo

Reducir el sistema lineal a forma escalonada reducida para dejar la solucion despejada con pivotes en 1.

### Logica

- Se valida la entrada igual que en Gauss.
- Se construye la matriz aumentada `[A | b]`.
- Se aplica reduccion hacia abajo y hacia arriba.
- Cada pivote se divide para quedar en 1.
- Se eliminan los terminos por encima y por debajo de cada pivote.

### Resultado esperado

- La parte izquierda queda en forma escalonada reducida.
- Si el sistema es compatible determinado, la columna final contiene la solucion.

### Salida

- Estado del sistema.
- Rangos de `A` y `[A|b]`.
- Matriz reducida.
- Solucion cuando existe una unica solucion.

## Regla de Cramer

### Regla

Solo se aplica si la matriz de coeficientes es cuadrada y `det(A) != 0`.

### Logica

- Se valida que `A` sea cuadrada.
- Se valida que el determinante de `A` sea distinto de cero.
- Para cada variable, se reemplaza la columna correspondiente de `A` por `b`.
- Se calcula:

$$
x_i = \frac{\det(A_i)}{\det(A)}
$$

donde `A_i` es la matriz obtenida al sustituir la columna `i` por `b`.

### Salida

- Determinante de `A`.
- Vector solucion.

## Resumen de restricciones por operacion

- Suma: mismas dimensiones.
- Resta: mismas dimensiones.
- Multiplicacion: columnas de `A` iguales a filas de `B`.
- Traspuesta: sin restriccion adicional.
- Determinante: matriz cuadrada.
- Inversa: matriz cuadrada y no singular.
- Gauss: `A` y `b` compatibles por filas.
- Gauss-Jordan: `A` y `b` compatibles por filas.
- Cramer: matriz cuadrada y no singular.

## Nota de implementacion

La logica numerica real vive en `backend/logic.py` y los filtros reutilizables en `backend/utils.py`. Este documento solo describe el comportamiento matematico esperado por esas funciones.