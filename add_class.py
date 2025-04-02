#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para añadir una nueva clase al conjunto de datos
"""

import os
import sys
import argparse
from utils import add_new_class

def parse_args():
    """
    Analiza los argumentos de línea de comandos
    
    Returns:
        args: Argumentos analizados
    """
    parser = argparse.ArgumentParser(description='Añadir una nueva clase al conjunto de datos')
    parser.add_argument('class_dir', help='Directorio con imágenes de la nueva clase')
    parser.add_argument('--raw-dir', default='data/raw', help='Directorio raw')
    return parser.parse_args()

def main():
    """Función principal"""
    args = parse_args()
    
    # Verificamos que el directorio exista
    if not os.path.isdir(args.class_dir):
        print(f"Error: No se encontró el directorio {args.class_dir}")
        return 1
    
    # Verificamos que el directorio raw exista
    if not os.path.isdir(args.raw_dir):
        print(f"Error: No se encontró el directorio raw {args.raw_dir}")
        return 1
    
    # Añadimos la nueva clase
    success = add_new_class(
        args.class_dir,
        args.raw_dir,
        'models/best_model.h5',
        'models/class_names.txt'
    )
    
    if success:
        print("\nPara reentrenar el modelo con la nueva clase:")
        print("1. Ejecute: python main.py")
        print("2. El modelo se actualizará automáticamente para incluir la nueva clase")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())