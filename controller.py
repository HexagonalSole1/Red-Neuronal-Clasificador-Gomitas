#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controller: API REST para el clasificador de gomitas
Permite la conexión con aplicaciones móviles
"""

import os
import base64
import json
import io
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from utils.utils import predict_image
import tensorflow as tf

# Configuración
MODEL_PATH = 'models/best_model.h5'
CLASS_NAMES_PATH = 'models/class_names.txt'
IMG_HEIGHT = 224
IMG_WIDTH = 224
HOST = '0.0.0.0'  # Escucha en todas las interfaces
PORT = int(os.environ.get('PORT', 5000))
UPLOAD_FOLDER = 'temp_uploads'

# Creamos la aplicación Flask
app = Flask(__name__)

# Aseguramos que exista el directorio temporal para subidas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargamos el modelo una sola vez al inicio
print("Cargando modelo...")
model = None

def load_model_if_needed():
    """Carga el modelo si no está ya cargado"""
    global model
    if model is None:
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Modelo cargado correctamente")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False
    return True

def load_class_names():
    """Carga los nombres de las clases"""
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Error al cargar nombres de clases: {e}")
        return []

@app.route('/api/info', methods=['GET'])
def get_info():
    """Devuelve información sobre el modelo"""
    class_names = load_class_names()
    
    # Leemos datos del resumen si existe
    summary_data = {}
    try:
        if os.path.exists('output/model_summary.txt'):
            with open('output/model_summary.txt', 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        summary_data[key.strip()] = value.strip()
    except Exception as e:
        print(f"Error al leer resumen: {e}")
    
    return jsonify({
        'status': 'ok',
        'model_name': 'Clasificador de Gomitas',
        'classes': class_names,
        'num_classes': len(class_names),
        'image_size': f"{IMG_HEIGHT}x{IMG_WIDTH}",
        'summary': summary_data
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Realiza una predicción con una imagen enviada"""
    if not load_model_if_needed():
        return jsonify({'status': 'error', 'message': 'Error al cargar el modelo'}), 500
    
    try:
        # Verificamos el tipo de contenido
        content_type = request.headers.get('Content-Type', '')
        
        # Caso 1: application/json - imagen en base64
        if 'application/json' in content_type:
            if 'image' not in request.json:
                return jsonify({'status': 'error', 'message': 'No se envió ninguna imagen'}), 400
            
            # Decodificamos la imagen en base64
            image_data = request.json['image']
            # Eliminamos el prefijo 'data:image/jpeg;base64,' si existe
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]
            
            # Convertimos de base64 a bytes
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
        # Caso 2: multipart/form-data - archivo de imagen
        elif 'multipart/form-data' in content_type or request.files:
            if 'file' not in request.files:
                return jsonify({'status': 'error', 'message': 'No se envió ningún archivo'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'status': 'error', 'message': 'No se seleccionó ningún archivo'}), 400
            
            # Leemos la imagen directamente
            img = Image.open(file.stream).convert('RGB')
        
        else:
            return jsonify({
                'status': 'error', 
                'message': 'Tipo de contenido no soportado. Use application/json o multipart/form-data'
            }), 415
        
        # Procesamos la imagen para el modelo
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        
        # Cargamos nombres de clases
        class_names = load_class_names()
        if not class_names:
            return jsonify({'status': 'error', 'message': 'Error al cargar nombres de clases'}), 500
        
        # Realizamos la predicción
        predictions = model.predict(img_array)
        
        # Obtenemos los índices ordenados por confianza (descendente)
        sorted_indices = np.argsort(predictions[0])[::-1]
        
        # Preparamos los resultados
        results = []
        for idx in sorted_indices:
            results.append({
                'class': class_names[idx],
                'confidence': float(predictions[0][idx])
            })
        
        # Guardamos la imagen para diagnósticos si se solicita
        save_image = False
        if 'application/json' in content_type and request.json.get('save_image', False):
            save_image = True
        elif request.form and request.form.get('save_image') in ['true', 'True', '1']:
            save_image = True
            
        if save_image:
            temp_path = os.path.join(UPLOAD_FOLDER, f"predict_{len(os.listdir(UPLOAD_FOLDER))}.jpg")
            img.save(temp_path)
        
        return jsonify({
            'status': 'ok', 
            'prediction': results[0]['class'],
            'confidence': results[0]['confidence'],
            'all_predictions': results
        })
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Devuelve la lista de clases disponibles"""
    class_names = load_class_names()
    return jsonify({
        'status': 'ok',
        'classes': class_names
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Comprueba si el servidor está funcionando"""
    return jsonify({'status': 'ok', 'message': 'El servidor está en funcionamiento'})

@app.route('/api/model_status', methods=['GET'])
def model_status():
    """Comprueba el estado del modelo"""
    model_exists = os.path.exists(MODEL_PATH)
    class_names_exist = os.path.exists(CLASS_NAMES_PATH)
    
    status = {
        'model_file_exists': model_exists,
        'class_names_file_exists': class_names_exist,
        'model_loaded': model is not None
    }
    
    if all(status.values()):
        return jsonify({'status': 'ok', 'details': status})
    else:
        return jsonify({'status': 'warning', 'details': status})

def run_server(custom_host=None, custom_port=None):
    """Inicia el servidor Flask"""
    global HOST, PORT
    
    # Actualizamos host y puerto si se proporcionan
    if custom_host:
        HOST = custom_host
    if custom_port:
        PORT = custom_port
    
    # Cargamos el modelo al inicio
    load_model_if_needed()
    
    print(f"Iniciando servidor en http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)

if __name__ == "__main__":
    run_server()