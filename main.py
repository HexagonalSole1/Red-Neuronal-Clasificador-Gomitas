#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Red-Neuronal-Gomitas: Clasificador de imágenes de gomitas y otros alimentos
Autor: [Tu Nombre]
Fecha: 2025.04.02
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
from utils.utils import convert_heic_to_jpg, prepare_dataset

# Configuración
NUM_CLASSES = 6  # Ajustar según número de clases actuales
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20
K_FOLDS = 5
LEARNING_RATE = 0.001

def create_model(num_classes):
    """
    Crea un modelo de red neuronal convolucional para clasificación de imágenes
    
    Args:
        num_classes: Número de clases a clasificar
    
    Returns:
        modelo: Modelo de red neuronal compilado
    """
    # Utilizamos una arquitectura basada en MobileNetV2 para eficiencia
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Congelamos las capas base para fine-tuning
    base_model.trainable = False
    
    modelo = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compilamos el modelo
    modelo.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return modelo

def train_with_cross_validation(X, y, num_classes, k_folds=K_FOLDS):
    """
    Entrena el modelo utilizando validación cruzada
    
    Args:
        X: Datos de imágenes
        y: Etiquetas
        num_classes: Número de clases
        k_folds: Número de particiones para validación cruzada
    
    Returns:
        histories: Historiales de entrenamiento
        val_accuracies: Precisiones de validación
        best_model: Mejor modelo entrenado
    """
    # Ajustamos k_folds si hay pocas muestras
    sample_count = X.shape[0]
    if sample_count < k_folds * 2:
        new_k_folds = max(2, sample_count // 2)
        print(f"⚠️ Advertencia: Demasiados folds ({k_folds}) para {sample_count} muestras.")
        print(f"Ajustando a {new_k_folds} folds para evitar errores.")
        k_folds = new_k_folds
    
    # Definimos la validación cruzada
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_no = 1
    histories = []
    val_accuracies = []
    best_accuracy = 0
    best_model = None

    for train_idx, val_idx in kfold.split(X):
        print(f'Entrenando fold {fold_no}/{k_folds}')
        
        # Dividimos los datos
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Creamos y entrenamos el modelo
        model = create_model(num_classes)
        
        # Aumento de datos para mejorar generalización
        data_augmentation = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )
        
        # Entrenamos el modelo
        history = model.fit(
            data_augmentation.flow(X_train, y_train, batch_size=BATCH_SIZE),
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            verbose=1
        )
        
        # Evaluamos el modelo
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f'Fold {fold_no}: Precisión de validación = {val_accuracy:.4f}')
        
        histories.append(history)
        val_accuracies.append(val_accuracy)
        
        # Guardamos el mejor modelo
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model
        
        fold_no += 1
    
    return histories, val_accuracies, best_model

def plot_training_history(histories, k_folds=K_FOLDS):
    """
    Grafica el historial de entrenamiento
    
    Args:
        histories: Lista de historiales de entrenamiento
        k_folds: Número de particiones
    """
    plt.figure(figsize=(12, 8))
    
    for i in range(k_folds):
        plt.subplot(2, 3, i+1)
        plt.plot(histories[i].history['accuracy'], label='train')
        plt.plot(histories[i].history['val_accuracy'], label='validation')
        plt.title(f'Fold {i+1}')
        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/cross_validation_results.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Genera y guarda la matriz de confusión
    
    Args:
        y_true: Etiquetas reales
        y_pred: Predicciones
        class_names: Nombres de las clases
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.savefig('output/confusion_matrix.png')
    plt.close()
    
    # También guardamos un reporte de clasificación
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv('output/classification_report.csv')
    
    return cm

def main():
    # Creamos directorios necesarios
    os.makedirs('data/entrenamiento', exist_ok=True)
    os.makedirs('data/prueba', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Verificamos que exista el directorio raw
    raw_dir = 'data/raw'
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)
        print(f"Se creó el directorio {raw_dir}. Por favor, coloque sus imágenes en subcarpetas por clase.")
        print("Ejemplo: data/raw/gomitas_acidul/, data/raw/dulcigomas/, etc.")
        return
    
    # Convertimos imágenes HEIC a JPG si es necesario
    try:
        convert_heic_to_jpg(raw_dir)
    except Exception as e:
        print(f"Error durante la conversión de HEIC: {e}")
        print("Continuando con las imágenes disponibles...")
    
    # Preparamos el dataset
    X, y, class_names = prepare_dataset('data/raw', IMG_HEIGHT, IMG_WIDTH)
    num_classes = len(class_names)
    
    # Entrenamos con validación cruzada
    histories, val_accuracies, best_model = train_with_cross_validation(X, y, num_classes)
    
    # Graficamos resultados de validación cruzada
    plot_training_history(histories)
    
    # Guardamos el mejor modelo
    best_model.save('models/best_model.h5')
    
    # Evaluamos en todo el conjunto de datos
    y_pred_probs = best_model.predict(X)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y, axis=1)
    
    # Generamos y guardamos la matriz de confusión
    cm = plot_confusion_matrix(y_true, y_pred, class_names)
    
    # Guardamos los nombres de las clases para usar en predicciones futuras
    with open('models/class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    # Imprimimos resumen de resultados
    print("\nResultados del entrenamiento:")
    print(f"Precisión promedio en validación cruzada: {np.mean(val_accuracies):.4f}")
    print(f"Desviación estándar: {np.std(val_accuracies):.4f}")
    
    # Generamos un archivo de resumen con métricas
    with open('output/model_summary.txt', 'w') as f:
        f.write(f"Número de clases: {num_classes}\n")
        f.write(f"Clases: {', '.join(class_names)}\n")
        f.write(f"Tamaño de imagen: {IMG_HEIGHT}x{IMG_WIDTH}\n")
        f.write(f"Épocas: {EPOCHS}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Número de folds: {K_FOLDS}\n")
        f.write(f"Precisión promedio: {np.mean(val_accuracies):.4f}\n")
        f.write(f"Desviación estándar: {np.std(val_accuracies):.4f}\n")

if __name__ == "__main__":
    main()