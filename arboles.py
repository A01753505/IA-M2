# Importación de bibliotecas
import pandas as pd
from sklearn.datasets import load_breast_cancer #type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV #type: ignore
from sklearn.tree import DecisionTreeClassifier #type: ignore
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score #type: ignore
import matplotlib.pyplot as plt
import seaborn as sns #type: ignore

# Cargar el dataset
# El dataset contiene información relacionada con estudiantes inscritos en títulos de pregrado
# Tiene información del estudiante así como su desempeño académico en los semestres
# El objetivo es predecir si un estudiante abandonó, sigue inscrito o se gradua

df = pd.read_csv('data.csv', delimiter=';')
X = df.drop(columns=['Target'])
y = df['Target']

# División del dataset en train, validation y test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)  # 60% train, 40% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 20% val, 20% test

# Modelo de árbol de decisión
tree = DecisionTreeClassifier()

# Hiperparámetros para optimización
param_grid = {
    'criterion': ['gini', 'entropy'],   # Función para medir calidad de una división
    'max_depth': [2, 3, 4, 5],          # Profundidad máxima del árbol
    'min_samples_split': [2, 4, 6],     # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': [1, 2, 3],      # Mínimo de muestras en una hoja
    'max_features': [None, 10, 20, 30]  # Número de características a considerar en cada división
}

# GridSearchCV para buscar el mejor modelo tomando en cuenta un diccionario de hiperparámetros
# con validación cruzada de 5 pliegues y métrica de evaluación 'accuracy'

# Clasificación
grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='accuracy')  # accuracy - maximizar la cantidad de predicciones correctas
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

# # Métricas de evaluación con el conjunto de prueba
# print("\nMétricas en el conjunto de prueba (Test Set):")
# print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
# print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
# print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
# print(f"F1 Score: {f1_score(y_test, y_test_pred):.4f}")

# print("\nMatriz de confusión (Test Set):")
# conf_matrix_test = confusion_matrix(y_test, y_test_pred)
# sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()
