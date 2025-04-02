#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
App: Punto de entrada para la aplicación de clasificación de gomitas
Incluye servidor API REST y interfaz web
"""

import argparse
import os
import sys
from controller import run_server

def parse_args():
    """
    Analiza los argumentos de línea de comandos
    
    Returns:
        args: Argumentos analizados
    """
    parser = argparse.ArgumentParser(description='Servidor API y Web para el clasificador de gomitas')
    parser.add_argument('--port', type=int, default=5000, help='Puerto del servidor (default: 5000)')
    parser.add_argument('--host', default='0.0.0.0', help='Host del servidor (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='Ejecutar el servidor en modo debug')
    return parser.parse_args()

def check_directories():
    """
    Verifica y crea los directorios necesarios
    """
    required_dirs = [
        'data/raw',
        'data/entrenamiento',
        'data/prueba',
        'models',
        'output',
        'temp_uploads',
        'static',
        'static/css',
        'static/js',
        'static/img',
        'static/uploads',
        'templates'
    ]
    
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
        
    print("✅ Directorios verificados y creados si es necesario")

def check_template_files():
    """
    Verifica si existen los archivos de plantilla HTML
    """
    template_files = [
        'templates/layout.html',
        'templates/index.html',
        'templates/predict.html',
        'templates/result.html',
        'templates/classes.html',
        'templates/about.html'
    ]
    
    missing_files = []
    for file in template_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️ Advertencia: Los siguientes archivos de plantilla no existen:")
        for file in missing_files:
            print(f"  - {file}")
        print("Por favor, asegúrate de que los archivos de plantilla estén en el directorio 'templates'")
        return False
    
    return True

def check_static_files():
    """
    Verifica si existen los archivos estáticos
    """
    static_files = [
        'static/css/style.css',
        'static/js/main.js'
    ]
    
    missing_files = []
    for file in static_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️ Advertencia: Los siguientes archivos estáticos no existen:")
        for file in missing_files:
            print(f"  - {file}")
        print("Por favor, asegúrate de que los archivos estáticos estén en los directorios correspondientes")
        return False
    
    return True

def main():
    """Función principal"""
    args = parse_args()
    
    # Verificamos directorios necesarios
    check_directories()
    
    # Verificamos si el modelo está entrenado
    if not os.path.exists('models/best_model.h5'):
        print("⚠️ Advertencia: No se encontró el modelo entrenado.")
        print("Por favor, ejecute main.py primero para entrenar el modelo.")
        print("El servidor se iniciará, pero no podrá realizar predicciones hasta que entrene el modelo.")
    
    # Verificamos archivos de plantillas y estáticos
    templates_ok = check_template_files()
    static_ok = check_static_files()
    
    if not templates_ok or not static_ok:
        print("\n⚠️ Faltan algunos archivos para la interfaz web.")
        print("La API funcionará correctamente, pero la interfaz web podría tener problemas.")
        response = input("¿Desea continuar de todos modos? (s/n): ")
        if response.lower() != 's':
            print("Operación cancelada por el usuario.")
            return 1
    
    # Mostramos información de inicio
    print("\n" + "="*80)
    print(" Clasificador de Gomitas - Servidor API y Web ".center(80, "="))
    print("="*80)
    print(f"📡 API REST: http://{args.host}:{args.port}/api")
    print(f"🌐 Interfaz Web: http://{args.host}:{args.port}/")
    print(f"🧪 Endpoints API disponibles:")
    print(f"   - GET  /api/info")
    print(f"   - POST /api/predict")
    print(f"   - GET  /api/classes")
    print(f"   - GET  /api/health")
    print(f"   - GET  /api/model_status")
    print("="*80)
    
    # Iniciamos el servidor con los parámetros pasados
    try:
        run_server(custom_host=args.host, custom_port=args.port)
        return 0
    except KeyboardInterrupt:
        print("\n👋 Servidor detenido por el usuario")
        return 0
    except Exception as e:
        print(f"\n❌ Error al iniciar el servidor: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())