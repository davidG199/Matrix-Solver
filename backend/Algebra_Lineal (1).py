import numpy as np

mostrar_pasos = True
historial = []

# -------------------------
# INPUT SEGURO
# -------------------------

def pedir_entero(msg):
    while True:
        try:
            return int(input(msg))
        except:
            print("Error: ingresa un entero válido.")

def pedir_fila(cols):
    while True:
        try:
            fila = list(map(float, input().split()))
            if len(fila) != cols:
                print(f"Error: deben ser {cols} valores.")
            else:
                return fila
        except:
            print("Error en la fila.")

def pedir_matriz(nombre):
    f = pedir_entero(f"Filas de {nombre}: ")
    c = pedir_entero(f"Columnas de {nombre}: ")
    M = []
    for i in range(f):
        print(f"Fila {i+1}: ", end="")
        M.append(pedir_fila(c))
    return np.array(M)

def pedir_vector(n):
    while True:
        try:
            v = list(map(float, input("Vector b: ").split()))
            if len(v) != n:
                print(f"Error: deben ser {n} valores.")
            else:
                return np.array(v)
        except:
            print("Error en vector.")

# -------------------------
# UTILIDADES
# -------------------------

def imprimir_matriz(M):
    for fila in M:
        print(["{:.2f}".format(x) for x in fila])
    print()

def vars_n(n):
    return [f"x{i+1}" for i in range(n)]

def guardar_historial(texto):
    historial.append(texto)

def exportar():
    with open("resultado.txt", "w") as f:
        for h in historial:
            f.write(h + "\n")
    print("Exportado a resultado.txt")

# -------------------------
# OPERACIONES
# -------------------------

def suma():
    A = pedir_matriz("A")
    B = pedir_matriz("B")
    if A.shape != B.shape:
        print("Dimensiones incompatibles")
        return
    R = A + B
    imprimir_matriz(R)
    guardar_historial("Suma realizada")

def resta():
    A = pedir_matriz("A")
    B = pedir_matriz("B")
    if A.shape != B.shape:
        print("Dimensiones incompatibles")
        return
    R = A - B
    imprimir_matriz(R)
    guardar_historial("Resta realizada")

def mult():
    A = pedir_matriz("A")
    B = pedir_matriz("B")
    if A.shape[1] != B.shape[0]:
        print("No se pueden multiplicar")
        return
    R = np.dot(A, B)
    imprimir_matriz(R)
    guardar_historial("Multiplicación realizada")

def det():
    A = pedir_matriz("A")
    if A.shape[0] != A.shape[1]:
        print("No es cuadrada")
        return
    d = np.linalg.det(A)
    print("Determinante:", round(d,2))
    guardar_historial(f"Determinante: {d}")

def inv():
    A = pedir_matriz("A")
    if A.shape[0] != A.shape[1]:
        print("No es cuadrada")
        return
    try:
        R = np.linalg.inv(A)
        imprimir_matriz(R)
        guardar_historial("Inversa calculada")
    except:
        print("No tiene inversa")

# -------------------------
# SISTEMAS
# -------------------------

def clasificar(M):
    filas, cols = M.shape
    rank = 0
    for i in range(filas):
        if not np.allclose(M[i,:-1],0):
            rank += 1
        elif not np.isclose(M[i,-1],0):
            return "SI"
    if rank < cols-1:
        return "SCI"
    return "SCD"

def gauss_jordan(A,b):
    global mostrar_pasos
    M = np.hstack([A.astype(float), b.reshape(-1,1)])
    
    if mostrar_pasos:
        print("\nMatriz inicial:")
        imprimir_matriz(M)

    n = len(b)
    for i in range(n):
        if np.isclose(M[i][i],0):
            continue
        
        M[i] = M[i]/M[i][i]
        
        if mostrar_pasos:
            imprimir_matriz(M)

        for j in range(n):
            if i!=j:
                M[j] = M[j] - M[j][i]*M[i]
                if mostrar_pasos:
                    imprimir_matriz(M)
    
    return M

def verificar(A,x,b):
    res = np.dot(A,x)
    if np.allclose(res,b):
        print("Verificación correcta")
    else:
        print("Error en solución")

def sistema():
    A = pedir_matriz("A")
    b = pedir_vector(A.shape[0])
    
    M = gauss_jordan(A,b)
    tipo = clasificar(M)
    
    if tipo=="SI":
        print("Sistema incompatible (sin solución)")
    elif tipo=="SCI":
        print("Sistema compatible indeterminado (infinitas soluciones)")
    else:
        print("Sistema compatible determinado (única solución)")
        x = M[:,-1]
        vars = vars_n(len(x))
        for i in range(len(x)):
            print(f"{vars[i]} = {x[i]:.2f}")
        verificar(A,x,b)

    guardar_historial(f"Sistema resuelto: {tipo}")

# -------------------------
# MENÚ
# -------------------------

def menu():
    global mostrar_pasos
    
    while True:
        print("\n--- MENÚ PRO ---")
        print("1. Suma")
        print("2. Resta")
        print("3. Multiplicación")
        print("4. Determinante")
        print("5. Inversa")
        print("6. Sistema lineal")
        print("7. Toggle pasos")
        print("8. Ver historial")
        print("9. Exportar")
        print("0. Salir")
        
        op = input("Opción: ")
        
        if op=="1": suma()
        elif op=="2": resta()
        elif op=="3": mult()
        elif op=="4": det()
        elif op=="5": inv()
        elif op=="6": sistema()
        elif op=="7":
            mostrar_pasos = not mostrar_pasos
            print("Pasos:", "ON" if mostrar_pasos else "OFF")
        elif op=="8":
            for h in historial:
                print(h)
        elif op=="9":
            exportar()
        elif op=="0":
            break
        else:
            print("Opción inválida")

menu()