# ¿Cómo funciona NumPy por debajo? (Guía para sustentar)

Para poder explicarle a tu profesora de Álgebra Lineal cómo calcula NumPy operaciones complejas como el determinante (`np.linalg.det`) o la matriz inversa (`np.linalg.inv`), debes saber que NumPy **no hace estos cálculos usando las fórmulas básicas que se enseñan en los primeros semestres** (como Regla de Sarrus o la matriz adjunta/cofactores), sino que utiliza la Descomposición LU a través de librerías altamente optimizadas escritas en C y Fortran (LAPACK y BLAS).

A continuación tienes toda la argumentación teórica y técnica para defender el código.

---

## 1. El Determinante (`np.linalg.det`)

Si la matriz es de 2x2 o 3x3, a nosotros en clase nos enseñan a cruzar diagonales o usar Sarrus. Sin embargo, matemáticamente ese método es $O(n!)$ (factorial), lo que significa que para una matriz de 10x10, Sarrus tardaría muchísimo tiempo en computarse. 

### ¿Qué hace NumPy por detrás?
En lugar de cofactores, NumPy usa un algoritmo llamado **Descomposición LU (Lower-Upper)** que tiene una complejidad de $O(n^3)$. 

1. **Validaciones previas:** NumPy primero verifica internamente que la matriz sea un arreglo bidimensional (2D) y que sea cuadrada (filas == columnas). Si no lo es, arroja el error `LinAlgError`.
2. **Llamada a LAPACK:** Llama a una función interna de la librería LAPACK (generalmente una rutina llamada `_getrf`).
3. **Factorización LU:** Transforma la matriz original $A$ en la multiplicación de tres matrices: $P \cdot A = L \cdot U$
   - $P$ es una matriz de permutación (registra si se cambiaron filas de lugar para evitar dividir por ceros).
   - $L$ es una matriz triangular inferior (ceros por encima de la diagonal) cuyos elementos en la diagonal son 1.
   - $U$ es una matriz triangular superior (ceros por debajo de la diagonal).
4. **Cálculo del determinante:** Una propiedad fundamental de los determinantes es que el determinante de una matriz triangular es simplemente el **producto de la diagonal principal**.
   Entonces, el cálculo se resume a:
   $$ Det(A) = Determinante(P) \times Determinante(L) \times Determinante(U) $$
   - $Det(L)$ es siempre 1 (su diagonal son puros 1).
   - $Det(U)$ es multiplicar los numeritos que quedaron en su diagonal principal.
   - $Det(P)$ es $+1$ o $-1$ dependiendo de si se hicieron un número par o impar de intercambios de filas.
5. **Resultado final:** NumPy simplemente multiplica esos valores escalares. Es un método ultra rápido, numéricamente estable y funciona igual de bien para matrices 2x2 o 1000x1000.

### Ejemplo ilustrativo (Determinante 3x3)
Imaginemos que NumPy recibe esta matriz:
$$ A = \begin{pmatrix} 2 & 1 & 1 \\ 4 & -6 & 0 \\ -2 & 7 & 2 \end{pmatrix} $$

En lugar de cruzar diagonales con la Regla de Sarrus, el motor LAPACK la escalona para formar una matriz Triangular Superior ($U$) mediante operaciones de fila:
1. Elimina el 4 y el -2 de la primera columna usando el 2 de arriba como pivote.
2. Luego usa el nuevo centro para eliminar el elemento de abajo.

Al final, la factorización interna deja una matriz $U$ que se ve así:
$$ U = \begin{pmatrix} 2 & 1 & 1 \\ 0 & -8 & -2 \\ 0 & 0 & 1 \end{pmatrix} $$

Como $L$ sólo tiene $1$s en su diagonal transversal y no hubo que intercambiar filas de orden ($P=1$), a NumPy sólo le falta multiplicar la nueva diagonal principal de $U$:
$$ Det = 2 \times (-8) \times 1 = -16 $$
¡Así de sencillo y sin usar cofactores computa el resultado internamente!

---

## 2. La Matriz Inversa (`np.linalg.inv`)

A mano, nos enseñan que $A^{-1} = \frac{1}{Det(A)} \cdot Adj(A)$. Para computación, calcular la matriz de cofactores es sumamente ineficiente y propenso a errores catastróficos de redondeo.

### ¿Qué hace NumPy por detrás?
NumPy delega esto a la rutina `_getri` (general matrix inverse) de LAPACK, que de nuevo, recae sobre la Descomposición LU.

1. **Validación:** Verifica que sea cuadrada.
2. **Pivoteo Parcial y Factorización LU:** Primero factoriza la matriz en $P \cdot A = L \cdot U$ igual que lo hace para el determinante. (Si al hacer esto descubre que uno de los valores en la diagonal de $U$ es excesivamente cercano a 0, sabe que la matriz es singular y lanza la excepción de que no es invertible).
3. **Sustitución hacia adelante y hacia atrás:** Una vez que tiene $A = P^{-1} \cdot L \cdot U$, el objetivo es encontrar la matriz $X$ tal que $A \cdot X = I$ (donde $I$ es la identidad).
   NumPy resuelve un sistema de ecuaciones múltiples al tiempo:
   - Primero resuelve $L \cdot Y = P \cdot I$ (mediante sustitución progresiva).
   - Luego resuelve $U \cdot X = Y$ (mediante sustitución regresiva).
4. **Respuesta:** La matriz $X$ resultante es $A^{-1}$. Al usar LU, garantiza máxima precisión con decimales flotantes.

---

## 3. Resolución de Sistemas (Si se usara `np.linalg.solve`)

Si le preguntan: _"¿Por qué hacer Gauss-Jordan a mano con un `for` en vez de usar las funciones directas del lenguaje?"_

Respuesta: Porque internamente librerías como NumPy (`np.linalg.solve`) usan métodos ultra-optimizados (de nuevo, factorización LU y LAPACK_gesv) que arrojan el **resultado final de manera instantánea perdiendo toda la traza de los pasos intermedios**. 
Para un contexto educativo donde se necesita ver explícitamente cómo la matriz aumentada $[\ A \,|\, b\ ]$ va sumando, multiplicando e intercambiando filas usando escalares manuales, es obligatorio implementar la iteración escalar clásica. 

### ¿Qué pasa según los tamaños (2x2, 3x3, NxN)?
Para NumPy las dimensiones no cambian la forma de resolverlo. El algoritmo iterativo de LU es el mismo sin importar el tamaño (`N`). La gran diferencia está en cómo la CPU administra la memoria local; las operaciones matriciales de NumPy en C se encargan de agrupar los datos en "bloques" que quepan en la memoria caché del procesador para multiplicar números ridículamente rápido.

---

### Resumen para responder rápido en clase:
- **"¿Qué operaciones hace np.linalg.det?"** 
  *Descompone la matriz en un sistema Triangular Superior e Inferior (Factorización LU). Luego multiplica la diagonal principal resultante, sumando un signo 1 o -1 dependiendo de los cambios de fila. Es un método $O(n^3)$ en vez del factorial de Sarrus.*
- **"¿Y qué hace la inversa internamente?"** 
  *No calcula matrices adjuntas. Aprovecha la misma descomposición LU para resolver un sistema equivalente a $A \cdot X = I$ emparejado con la matriz Identidad.*
- **"¿Por qué usamos NumPy sólo para lo básico y fracciones, y Gaús a mano?"** 
  *NumPy nos asegura operaciones aritméticas base muy sólidas, pero como procesa todo a través de librerías ocultas (C/Fortran), usar su 'Solve' nos habría quitado la opción de extraer los pasos matemáticos para mostrarlos en tablas.*