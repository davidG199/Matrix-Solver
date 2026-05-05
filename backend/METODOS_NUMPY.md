# Métodos de NumPy usados en la lógica del proyecto

Este documento resume los métodos y atributos de NumPy que aparecen en la lógica algebraica del backend y explica qué hacen dentro del proyecto.

## Objetivo en el proyecto

NumPy se usa para:
- normalizar matrices y vectores que llegan desde la API,
- validar dimensiones y forma,
- ejecutar operaciones algebraicas de manera eficiente,
- reducir matrices para Gauss, Gauss-Jordan, determinantes e inversas,
- convertir resultados a formatos compatibles con JSON.

## Métodos y atributos utilizados

### 1. np.asarray
Convierte una entrada a un arreglo de NumPy sin copiar innecesariamente si ya era un arreglo compatible.

Uso en el proyecto:
- transformar listas anidadas en matrices de tipo float,
- transformar vectores en arreglos 1D,
- preparar datos antes de operar con ellos.

Qué hace:
- intenta interpretar la entrada como arreglo,
- permite fijar el tipo de dato con dtype,
- facilita operar con matrices homogéneas.

### 2. np.array
Crea un arreglo nuevo a partir de datos dados.

Uso en el proyecto:
- construir resultados directos, por ejemplo en la inversa 1x1 o 2x2,
- crear matrices identidad y otras estructuras auxiliares.

Qué hace:
- genera un arreglo explícitamente nuevo,
- permite definir el tipo de dato desde el inicio.

### 3. np.hstack
Une arreglos horizontalmente, es decir, por columnas.

Uso en el proyecto:
- construir la matriz aumentada [A | b],
- unir la matriz de coeficientes con el vector de términos independientes,
- armar matrices extendidas para Gauss y Gauss-Jordan.

Qué hace:
- toma dos o más arreglos con la misma cantidad de filas,
- los concatena lado a lado.

### 4. np.eye
Crea una matriz identidad.

Uso en el proyecto:
- calcular la inversa de una matriz usando [A | I],
- comparar que la parte izquierda de la reducción quede como identidad.

Qué hace:
- produce una matriz cuadrada con 1 en la diagonal principal y 0 en el resto.

### 5. np.matmul
Realiza multiplicación matricial.

Uso en el proyecto:
- multiplicación de matrices en la operación multiplicacion,
- cálculo del producto matricial respetando la regla columnas(A) = filas(B).

Qué hace:
- ejecuta el producto lineal entre matrices,
- no hace multiplicación elemento por elemento, sino producto algebraico real.

### 6. np.linalg.det
Calcula el determinante de una matriz cuadrada.

Uso en el proyecto:
- validar si una matriz tiene inversa,
- resolver el método de Cramer,
- apoyar el cálculo de determinantes en algunos flujos de validación.

Qué hace:
- devuelve un número escalar,
- si el determinante es 0, la matriz es singular y no tiene inversa.

### 7. np.linalg.matrix_rank
Calcula el rango de una matriz.

Uso en el proyecto:
- clasificar sistemas lineales,
- comparar el rango de A con el rango de la matriz aumentada [A | b].

Qué hace:
- devuelve cuántas filas o columnas linealmente independientes tiene una matriz.

### 8. np.linalg.norm no se usa
No aparece en la lógica actual del proyecto.

### 9. np.argmax
Devuelve el índice del valor máximo en un arreglo.

Uso en el proyecto:
- elegir el pivote más fuerte en eliminación con pivoteo parcial,
- detectar la fila con el mayor valor absoluto en una columna.

Qué hace:
- devuelve la posición del máximo valor encontrado,
- al aplicarlo sobre abs(matriz[columna:]), ayuda a mejorar estabilidad numérica.

### 10. np.abs
Devuelve el valor absoluto de cada elemento.

Uso en el proyecto:
- comparar pivotes por magnitud,
- detectar residuos numéricos pequeños,
- aplicar umbrales de tolerancia.

Qué hace:
- elimina el signo de números reales,
- en matrices sirve para trabajar con magnitudes.

### 11. np.all
Verifica si todos los elementos de una condición son verdaderos.

Uso en el proyecto:
- comprobar que todos los valores de una matriz son finitos,
- validar que no haya NaN ni Inf.

Qué hace:
- devuelve True si toda la condición se cumple,
- devuelve False si al menos un elemento falla.

### 12. np.isfinite
Detecta si un número es finito.

Uso en el proyecto:
- validación de entradas numéricas,
- rechazo de valores NaN e Inf.

Qué hace:
- devuelve un arreglo booleano o un booleano,
- permite filtrar datos inválidos antes de operar.

### 13. np.allclose
Compara dos arreglos con tolerancia numérica.

Uso en el proyecto:
- verificar que la parte izquierda de la reducción de la inversa sea realmente la identidad,
- aceptar pequeñas diferencias por error de punto flotante.

Qué hace:
- no exige igualdad exacta,
- considera cercanos dos valores si la diferencia es menor que una tolerancia.

### 14. np.where
Busca posiciones donde se cumple una condición.

Uso en el proyecto:
- identificar columnas pivote en la extracción de soluciones desde RREF.

Qué hace:
- devuelve índices de los elementos que cumplen la condición dada.

### 15. np.zeros
Crea un arreglo lleno de ceros.

Uso en el proyecto:
- construir el vector solución inicial en forma reducida,
- asignar luego los valores encontrados en cada variable.

Qué hace:
- genera una base limpia para ir llenando resultados.

### 16. reshape
Atributo/método de los arreglos de NumPy para cambiar la forma de un arreglo.

Uso en el proyecto:
- convertir un vector en columna al construir la matriz aumentada,
- aplanar vectores cuando se serializan.

Qué hace:
- reorganiza los datos sin cambiar sus valores,
- solo cambia la forma dimensional.

### 17. copy
Crea una copia independiente del arreglo.

Uso en el proyecto:
- trabajar sobre una copia al calcular determinantes por eliminación,
- evitar modificar la matriz original durante procesos intermedios.

Qué hace:
- permite alterar el arreglo de trabajo sin afectar la entrada original.

### 18. astype
Cambia el tipo de dato de un arreglo.

Uso en el proyecto:
- asegurar cálculos en float,
- preparar arreglos para operaciones numéricas estables.

Qué hace:
- convierte datos a otro tipo, por ejemplo de int a float.

### 19. T
Propiedad que devuelve la traspuesta de una matriz.

Uso en el proyecto:
- implementar la operación de traspuesta de forma directa.

Qué hace:
- intercambia filas por columnas.

### 20. shape
Atributo que devuelve las dimensiones de un arreglo.

Uso en el proyecto:
- validar compatibilidad entre matrices,
- calcular filas y columnas,
- decidir el comportamiento según el tamaño de la matriz.

Qué hace:
- devuelve una tupla con la forma del arreglo, por ejemplo (filas, columnas).

### 21. size
Atributo que devuelve la cantidad total de elementos.

Uso en el proyecto:
- verificar que una matriz no esté vacía,
- decidir casos base como 1x1 o 2x2 en determinantes e inversas.

Qué hace:
- cuenta todos los valores contenidos en el arreglo.

### 22. tolist
Convierte un arreglo de NumPy en listas nativas de Python.

Uso en el proyecto:
- serializar resultados para la respuesta JSON de la API,
- devolver matrices y vectores en un formato que FastAPI pueda enviar.

Qué hace:
- transforma el arreglo en listas o listas anidadas.

## Cómo encajan en la lógica

En términos simples, el flujo del backend es este:
1. Las entradas llegan como listas desde la API.
2. NumPy las normaliza a arreglos numéricos.
3. Se validan dimensiones, rango, cuadratura o compatibilidad.
4. Se ejecuta la operación algebraica.
5. El resultado se convierte otra vez en listas para JSON.

## Importancia práctica

NumPy no solo acelera los cálculos. En este proyecto también sirve para:
- reducir errores por tipos de dato mezclados,
- manejar matrices con reglas algebraicas claras,
- trabajar con tolerancias numéricas razonables,
- devolver resultados estables y fáciles de consumir desde el frontend.

## Resumen rápido

Los métodos más importantes en este proyecto son:
- np.asarray para normalizar entradas,
- np.hstack para matrices aumentadas,
- np.matmul para multiplicación,
- np.linalg.det para determinantes e inversas,
- np.linalg.matrix_rank para clasificar sistemas,
- np.eye para inversas,
- np.allclose para comparar resultados con tolerancia.

Si quieres, este documento se puede ampliar luego con ejemplos numéricos de cada operación usando matrices de 2x2 o 3x3.