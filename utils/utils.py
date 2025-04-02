#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilidades para el procesamiento de imágenes y preparación de datos
"""

import os
import numpy as np
import glob
from PIL import Image
import subprocess
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import shutil
import matplotlib.pyplot as plt

def convert_heic_to_jpg(data_dir):
    """
    Convierte imágenes HEIC a JPG utilizando varios métodos alternativos
    
    Args:
        data_dir: Directorio con imágenes HEIC
    """
    print("Buscando imágenes HEIC para convertir...")
    
    # Buscamos todos los archivos HEIC
    heic_files = []
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            heic_files.extend(glob.glob(os.path.join(folder_path, "*.heic")))
            heic_files.extend(glob.glob(os.path.join(folder_path, "*.HEIC")))
    
    if not heic_files:
        print("No se encontraron archivos HEIC para convertir.")
        return
    
    print(f"Se encontraron {len(heic_files)} archivos HEIC.")
    
    # Método 1: Intentar con pillow-heif (más común)
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        
        print("Usando pillow-heif para convertir imágenes...")
        
        for heic_file in heic_files:
            try:
                # Abrimos el archivo HEIC con PIL (ahora soporta HEIC gracias a pillow_heif)
                img = Image.open(heic_file)
                
                # Guardamos como JPG
                jpg_file = os.path.splitext(heic_file)[0] + ".jpg"
                img.save(jpg_file, "JPEG")
                print(f"Convertido: {heic_file} -> {jpg_file}")
                
            except Exception as e:
                print(f"Error al convertir {heic_file} con pillow-heif: {e}")
        
        return
        
    except ImportError:
        print("pillow-heif no disponible, intentando método alternativo...")
    
    # Método 2: Intentar con sips (solo MacOS)
    if sys.platform == 'darwin':  # MacOS
        print("Intentando convertir con sips (herramienta nativa de MacOS)...")
        
        for heic_file in heic_files:
            try:
                jpg_file = os.path.splitext(heic_file)[0] + ".jpg"
                subprocess.run(['sips', '-s', 'format', 'jpeg', heic_file, '--out', jpg_file], 
                              check=True, stdout=subprocess.PIPE)
                print(f"Convertido: {heic_file} -> {jpg_file}")
            except Exception as e:
                print(f"Error al convertir {heic_file} con sips: {e}")
        
        return
    
    # Método 3: Usar ImageMagick si está disponible
    try:
        subprocess.run(['convert', '--version'], check=True, stdout=subprocess.PIPE)
        print("Intentando convertir con ImageMagick...")
        
        for heic_file in heic_files:
            try:
                jpg_file = os.path.splitext(heic_file)[0] + ".jpg"
                subprocess.run(['convert', heic_file, jpg_file], check=True, stdout=subprocess.PIPE)
                print(f"Convertido: {heic_file} -> {jpg_file}")
            except Exception as e:
                print(f"Error al convertir {heic_file} con ImageMagick: {e}")
        
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ImageMagick no disponible...")
    
    print("\n¡IMPORTANTE! No se pudieron convertir los archivos HEIC a JPG.")
    print("Por favor, convierta manualmente los archivos HEIC a JPG antes de continuar.")
    print("También puede usar herramientas como 'sips' en MacOS o aplicaciones como iPhoto, Preview, etc.")

def prepare_dataset(data_dir, img_height, img_width, test_split=0.2, min_samples=5):
    """
    Prepara el conjunto de datos para entrenamiento
    
    Args:
        data_dir: Directorio con las imágenes por clase
        img_height: Altura objetivo de las imágenes
        img_width: Anchura objetivo de las imágenes
        test_split: Proporción para conjunto de prueba
        min_samples: Número mínimo de imágenes requeridas por clase
    
    Returns:
        X: Datos de imágenes
        y: Etiquetas codificadas
        class_names: Nombres de las clases
    """
    X = []  # Datos de imágenes
    y = []  # Etiquetas
    class_names = []  # Nombres de clases
    valid_class_indices = []  # Índices de clases válidas
    
    print("\n=== Preparando conjunto de datos ===")
    
    # Verificamos que exista el directorio
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"No se encontró el directorio {data_dir}")
    
    # Obtenemos las clases (carpetas)
    folders = [f for f in sorted(os.listdir(data_dir)) 
              if os.path.isdir(os.path.join(data_dir, f))]
    
    if not folders:
        raise ValueError(f"No se encontraron carpetas de clases en {data_dir}")
    
    # Creamos directorios principales si no existen
    os.makedirs("data/entrenamiento", exist_ok=True)
    os.makedirs("data/prueba", exist_ok=True)
    
    # Procesamos cada clase
    for idx, folder in enumerate(folders):
        folder_path = os.path.join(data_dir, folder)
        
        print(f"\nProcesando clase: {folder} (índice {idx})")
        
        # Creamos directorios para entrenamiento y prueba
        train_dir = os.path.join("data/entrenamiento", folder)
        test_dir = os.path.join("data/prueba", folder)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Obtenemos todas las imágenes jpg o png
        image_files = []
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        # Verificamos si hay suficientes imágenes
        if len(image_files) < min_samples:
            print(f"⚠️ Advertencia: La clase '{folder}' solo tiene {len(image_files)} imágenes, "
                  f"se requieren al menos {min_samples}. Esta clase será ignorada.")
            continue
        
        print(f"Encontradas {len(image_files)} imágenes")
        
        # Dividimos en entrenamiento y prueba
        if len(image_files) >= 2:  # Necesitamos al menos 2 imágenes para dividir
            # Calculamos el número real de muestras de prueba
            n_test = max(1, int(len(image_files) * test_split))
            n_train = len(image_files) - n_test
            
            print(f"Dividiendo en {n_train} imágenes de entrenamiento y {n_test} de prueba")
            
            # Mezclamos y dividimos manualmente si hay pocas imágenes
            if len(image_files) < 10:
                np.random.seed(42)
                np.random.shuffle(image_files)
                train_files = image_files[:-n_test]
                test_files = image_files[-n_test:]
            else:
                train_files, test_files = train_test_split(
                    image_files, test_size=n_test/len(image_files), random_state=42
                )
        else:
            # Si solo hay una imagen, la usamos para entrenamiento y prueba
            train_files = image_files
            test_files = image_files
            print("⚠️ Advertencia: Solo hay una imagen, se usará para entrenamiento y prueba")
        
        # Copiamos archivos a carpetas de entrenamiento y prueba
        for file in train_files:
            dest = os.path.join(train_dir, os.path.basename(file))
            shutil.copy(file, dest)
        
        for file in test_files:
            dest = os.path.join(test_dir, os.path.basename(file))
            shutil.copy(file, dest)
        
        # Procesamos cada imagen
        class_X = []  # Imágenes de esta clase
        class_y = []  # Etiquetas de esta clase
        
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((img_width, img_height))
                img_array = np.array(img) / 255.0  # Normalización
                
                class_X.append(img_array)
                class_y.append(idx)
                
            except Exception as e:
                print(f"Error procesando {img_path}: {e}")
        
        # Solo añadimos la clase si se procesaron suficientes imágenes
        if len(class_X) >= min_samples:
            X.extend(class_X)
            y.extend(class_y)
            class_names.append(folder)
            valid_class_indices.append(idx)
            print(f"✓ Clase '{folder}' añadida con {len(class_X)} imágenes")
        else:
            print(f"⚠️ Advertencia: No se pudieron procesar suficientes imágenes para '{folder}', "
                  f"se requieren al menos {min_samples}. Esta clase será ignorada.")
    
    # Verificamos si hay suficientes clases
    if len(class_names) == 0:
        raise ValueError("No se encontraron clases con suficientes imágenes para entrenar")
    
    # Ajustamos índices si algunas clases fueron ignoradas
    if len(valid_class_indices) < len(folders):
        # Creamos un mapa de índice original a nuevo índice
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_class_indices)}
        # Remapeamos las etiquetas
        y = [idx_map[label] for label in y]
    
    # Convertimos a arrays de numpy
    X = np.array(X)
    y = np.array(y)
    
    # Codificamos etiquetas en one-hot
    y_encoded = to_categorical(y, num_classes=len(class_names))
    
    print(f"\n✅ Dataset preparado: {X.shape[0]} imágenes en {len(class_names)} clases")
    
    # Visualizamos algunas imágenes de ejemplo por clase
    visualize_examples(X, y, class_names)
    
    return X, y_encoded, class_names

def visualize_examples(X, y, class_names, samples_per_class=3):
    """
    Visualiza ejemplos de cada clase
    
    Args:
        X: Datos de imágenes
        y: Etiquetas (no codificadas)
        class_names: Nombres de las clases
        samples_per_class: Número de ejemplos por clase
    """
    os.makedirs('output', exist_ok=True)
    
    num_classes = len(class_names)
    plt.figure(figsize=(15, 2 * num_classes))
    
    for class_idx in range(num_classes):
        # Obtenemos imágenes de esta clase
        indices = np.where(y == class_idx)[0]
        
        # Si hay suficientes imágenes, mostramos samples_per_class ejemplos
        if len(indices) >= samples_per_class:
            sample_indices = indices[:samples_per_class]
            
            for i, sample_idx in enumerate(sample_indices):
                plt.subplot(num_classes, samples_per_class, class_idx * samples_per_class + i + 1)
                plt.imshow(X[sample_idx])
                plt.title(class_names[class_idx])
                plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('output/class_examples.png')
    plt.close()

def add_new_class(data_dir, raw_dir, model_path, class_names_path):
    """
    Añade una nueva clase al modelo existente
    
    Args:
        data_dir: Directorio con la nueva clase
        raw_dir: Directorio raw original
        model_path: Ruta al modelo guardado
        class_names_path: Ruta a los nombres de clases guardados
    
    Returns:
        True si se añadió correctamente, False en caso contrario
    """
    try:
        # Verificamos que la carpeta exista
        if not os.path.isdir(data_dir):
            print(f"Error: {data_dir} no es un directorio válido")
            return False
        
        # Obtenemos el nombre de la clase (nombre de la carpeta)
        class_name = os.path.basename(data_dir)
        
        # Copiamos las imágenes al directorio raw
        dest_dir = os.path.join(raw_dir, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Copiamos todas las imágenes
        for img_file in glob.glob(os.path.join(data_dir, "*.*")):
            shutil.copy(img_file, dest_dir)
        
        print(f"Nueva clase '{class_name}' añadida. "
              f"Ejecute main.py para reentrenar el modelo con la nueva clase.")
        
        return True
        
    except Exception as e:
        print(f"Error al añadir nueva clase: {e}")
        return False

def predict_image(image_path, model_path, class_names_path, img_height=224, img_width=224):
    """
    Predice la clase de una imagen
    
    Args:
        image_path: Ruta a la imagen
        model_path: Ruta al modelo guardado
        class_names_path: Ruta a los nombres de clases guardados
        img_height: Altura de la imagen
        img_width: Anchura de la imagen
    
    Returns:
        predicted_class: Nombre de la clase predicha
        confidence: Confianza de la predicción
    """
    from tensorflow.keras.models import load_model
    
    try:
        # Cargamos el modelo
        model = load_model(model_path)
        
        # Cargamos los nombres de las clases
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        # Procesamos la imagen
        img = Image.open(image_path).convert('RGB')
        img = img.resize((img_width, img_height))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        
        # Realizamos la predicción
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        return class_names[predicted_idx], confidence
        
    except Exception as e:
        print(f"Error al predecir: {e}")
        return None, 0.0