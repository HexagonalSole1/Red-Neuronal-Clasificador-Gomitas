#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para hacer predicciones con el modelo entrenado
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
from utils.utils import predict_image

def parse_args():
    """
    Analiza los argumentos de línea de comandos
    
    Returns:
        args: Argumentos analizados
    """
    parser = argparse.ArgumentParser(description='Predicción de imágenes con el modelo entrenado')
    parser.add_argument('image_path', help='Ruta a la imagen para predecir')
    parser.add_argument('--model', default='models/best_model.h5', help='Ruta al modelo guardado')
    parser.add_argument('--classes', default='models/class_names.txt', help='Ruta a los nombres de clases')
    return parser.parse_args()

def main():
    """Función principal"""
    args = parse_args()
    
    # Verificamos que la imagen exista
    if not os.path.exists(args.image_path):
        print(f"Error: No se encontró la imagen en {args.image_path}")
        return 1
    
    # Verificamos que el modelo exista
    if not os.path.exists(args.model):
        print(f"Error: No se encontró el modelo en {args.model}")
        return 1
    
    # Verificamos que el archivo de clases exista
    if not os.path.exists(args.classes):
        print(f"Error: No se encontró el archivo de clases en {args.classes}")
        return 1
    
    # Realizamos la predicción
    predicted_class, confidence = predict_image(
        args.image_path, args.model, args.classes
    )
    
    # Mostramos la imagen y la predicción
    img = Image.open(args.image_path).convert('RGB')
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f'Predicción: {predicted_class}\nConfianza: {confidence:.2%}')
    plt.axis('off')
    plt.tight_layout()
    
    # Guardamos la visualización
    output_dir = 'output/predictions'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, 
        f"pred_{os.path.basename(args.image_path).split('.')[0]}.png"
    )
    plt.savefig(output_path)
    
    print(f"\nResultado de la predicción:")
    print(f"Clase: {predicted_class}")
    print(f"Confianza: {confidence:.2%}")
    print(f"Visualización guardada en: {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())