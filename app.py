#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
App: Punto de entrada para la aplicación de clasificación de gomitas
"""

import argparse
import os
from controller import run_server

def parse_args():
    """
    Analiza los argumentos de línea de comandos
    
    Returns:
        args: Argumentos analizados
    """
    parser = argparse.ArgumentParser(description='Servidor API para el clasificador de gomitas')
    parser.add_argument('--port', type=int, default=5000, help='Puerto del servidor (default: 5000)')
    parser.add_argument('--host', default='0.0.0.0', help='Host del servidor (default: 0.0.0.0)')
    return parser.parse_args()

def main():
    """Función principal"""
    args = parse_args()
    
    # Verificamos si el modelo está entrenado
    if not os.path.exists('models/best_model.h5'):
        print("⚠️ Advertencia: No se encontró el modelo entrenado.")
        print("Por favor, ejecute main.py primero para entrenar el modelo.")
        return 1
    
    # Iniciamos el servidor con los parámetros pasados
    run_server(custom_host=args.host, custom_port=args.port)
    
    return 0

if __name__ == "__main__":
    main()