# Importación de bibliotecas
import pandas as pd
from sklearn.datasets import load_breast_cancer #type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV #type: ignore
from sklearn.tree import DecisionTreeClassifier #type: ignore
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score #type: ignore
import matplotlib.pyplot as plt
import seaborn as sns #type: ignore

# Cargar el dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# División del dataset en train, validation y test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)  # 60% train, 40% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 20% val, 20% test

# Modelo de árbol de decisión
tree = DecisionTreeClassifier()

# {'criterion': 'gini', # Función para medir calidad de una división [gini, entropy]
#  'splitter': 'best',  # Estrategia para elegir división en cada nodo [best, random]
#  'max_depth': None,   # Profundidad máxima del árbol (int, none)
#  'min_samples_split': 2,  # Mínimo de muestras para dividir un nodo (int, float)
#  'min_samples_leaf': 1,   # Mínimo de muestras en hoja (int, float)
#  'min_weight_fraction_leaf': 0.0, # Mínima fracción de la suma de pesos para estar en hoja (float)
#  'max_features': None,    # Número de características a considerar en cada división (int, float, none, auto, sqrt, log2)
#  'random_state': None,    # Semilla para el generador de números aleatorios (int, none)
#  'max_leaf_nodes': None,  # Máximo de nodos hoja (int, none)
#  'min_impurity_decrease': 0.0,    #Umbral de disminución de impureza para dividir (float)
#  'class_weight': None # Pesos asociados a las clases [dict, balanced, none]
# }

# Hiperparámetros para optimización
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3]
}

# GridSearchCV para optimizar el modelo con validación cruzada de 5 pliegues

# Regresión
# neg_mean_squared_error - minimizar el error cuadrático medio
# neg_mean_absolute_error - minimizar el error absoluto medio
# r2 - evaluar proporcion de varianza en la variable dependiente

# Clasificación
# accuracy - maximizar la cantidad de predicciones correctas
# f1 - balance entre precision y recall
# roc_auc - rendimiento del modelo en términos de capacidad de distinguir entre clases

grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Mejores hiperparámetros encontrados:", grid_search.best_params_)

# Mejor modelo obtenido
best_model = grid_search.best_estimator_

# Predicción del conjunto de validación
y_val_pred = best_model.predict(X_val)

# Evaluación del modelo con el conjunto de validación
print("\nReporte de clasificación (Validation Set):")
print(classification_report(y_val, y_val_pred))

print("\nMatriz de confusión (Validation Set):")
conf_matrix_val = confusion_matrix(y_val, y_val_pred)
sns.heatmap(conf_matrix_val, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# # Predicción del conjunto de prueba
# y_test_pred = best_model.predict(X_test)

# print("\nMatriz de confusión (Test Set):")
# conf_matrix_test = confusion_matrix(y_test, y_test_pred)
# sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# # Métricas de evaluación con el conjunto de prueba
# print("\nMétricas en el conjunto de prueba (Test Set):")
# print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
# print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
# print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
# print(f"F1 Score: {f1_score(y_test, y_test_pred):.4f}")
