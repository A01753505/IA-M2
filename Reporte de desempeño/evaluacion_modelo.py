# Importación de bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.tree import DecisionTreeClassifier #type: ignore
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score #type: ignore
import matplotlib.pyplot as plt
import seaborn as sns #type: ignore

# Cargar el dataset
df = pd.read_csv('Reporte de desempeño/datos.csv', delimiter=';')
X = df.drop(columns=['Target'])
y = df['Target']

# División del dataset en train, validation y test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)  # 60% train, 40% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 20% val, 20% test

# Información de los conjuntos de datos
print("\nEjemplo del dataset original")
print(df.head())
print("Tamaño del dataset original:")
print(df.shape)

print("\nEjemplo del conjunto de entrenamiento")
print(X_train.head())
print("Tamaño del conjunto de entrenamiento:")
print(X_train.shape)

print("\nEjemplo del conjunto de validación")
print(X_val.head())
print("Tamaño del conjunto de validación:")
print(X_val.shape)

print("\nEjemplo del conjunto de prueba")
print(X_test.head())
print("Tamaño del conjunto de prueba:")
print(X_test.shape)

# Modelo de árbol de decisión
model = DecisionTreeClassifier(criterion='gini', max_depth=5, max_features=35, min_samples_leaf=2, min_samples_split=4)
model.fit(X_train, y_train)

# Calcular métricas para el conjunto de prueba
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='macro')
test_recall = recall_score(y_test, y_test_pred, average='macro')
test_f1 = f1_score(y_test, y_test_pred, average='macro')

# Métricas de evaluación con el conjunto de prueba
print("\nMétricas en el conjunto de prueba")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")

# Crear un DataFrame con las métricas
test_metrics_df = pd.DataFrame({
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Valor': [test_accuracy, test_precision, test_recall, test_f1]
})

# Visualización de las métricas del conjunto de prueba
plt.figure(figsize=(10, 6))
sns.barplot(x='Métrica', y='Valor', data=test_metrics_df)
plt.title('Métricas de Evaluación del Modelo en el Conjunto de Prueba')
plt.ylim(0, 1)  # Las métricas están en el rango de 0 a 1
plt.show()

# Visualización de la matriz de confusión del conjunto de prueba
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
