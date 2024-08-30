# Función para calcular m0 y m1
def calcular_coeficientes(x, y):
    x_prom = sum(x)/len(x)
    y_prom = sum(y)/len(y)

    # Calculamos la pendiente (m1)
    numerador = sum((xi - x_prom) * (yi - y_prom) for xi, yi in zip(x, y))
    denominador = sum((xi - x_prom) ** 2 for xi in x)
    m1 = numerador / denominador

    # Calculamos m0
    m0 = y_prom - m1 * x_prom

    return m0, m1

# Función para hacer predicciones usando los coeficientes
def prediccion(x, m0, m1):
    return [((m1 * xi) + m0) for xi in x]

# Función para evaluar que tan bueno es el modelo
def evaluar(X, y_true, m0, m1):
    y_pred = prediccion(X, m0, m1)

    # Calcular el Error Cuadrático Medio (MSE)
    errores_cuadrados = [(yi - pred) ** 2 for yi, pred in zip(y_true, y_pred)]
    mse = sum(errores_cuadrados) / len(errores_cuadrados)

    # Calcular el Error Absoluto Medio (MAE)
    errores_absolutos = [abs(yi - pred) for yi, pred in zip(y_true, y_pred)]
    mae = sum(errores_absolutos) / len(errores_absolutos)

    return mse, mae


# EJEMPLO DE USO
# Datos de ejemplo
X_train = [1, 2, 3, 4, 5]
y_train = [3, 4, 5, 7, 11]
X_test = [2, 8, 3, 6, 1]
y_test = [10, 8, 6, 4, 2]

# Calcular coeficientes
m0, m1 = calcular_coeficientes(X_train, y_train)
print(f"Coeficientes calculados: m0 = {m0:.4f}, m1 = {m1:.4f}")

# Hacer predicciones con los datos de prueba
predicciones = prediccion(X_test, m0, m1)
print(f"Predicciones:\n {predicciones}")

# Evaluar el modelo
mse, mae = evaluar(X_test, y_test, m0, m1)
print(f"Evaluación del modelo:\n MSE = {mse:.4f}\n MAE = {mae:.4f}")
