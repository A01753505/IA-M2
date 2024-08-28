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

# Función para hacer predicciones usando el modelo
def prediccion(x, m0, m1):
    return [((m1 * xi) + m0) for xi in x]


# EJEMPLO DE USO
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Coeficientes
m0, m1 = calcular_coeficientes(x, y)

# Predicciones
predicciones = prediccion(x, m0, m1)

print(f'Intercepto (m0): {m0}')
print(f'Pendiente (m1): {m1}')
print(f'Predicciones: {predicciones}')
