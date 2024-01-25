from flask import Flask 
from flask import request
from flask import jsonify

from app.utils_torch import  leer_dato, obtener_categoria, predecir

import pickle

app=Flask(__name__)

ALLOWED_FILE_EXTENSIONS_TYPES={'pkl'}

def is_allowed(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_FILE_EXTENSIONS_TYPES

@app.route('/predict',methods=['POST','GET'])
def predictautoencoder():
    if request.method == 'POST':
        file= request.files.get('file')
        if file is None or file.filename=='':
            return jsonify({'error':'no file'})
        if not is_allowed(file.filename):
            return jsonify({'error':'not a valid file'})
        try:
            dato= leer_dato(file)
            prediccion= predecir(dato)
            data = obtener_categoria(prediccion)
            data = {'prediction': data, 'class_name': str(prediccion.item())}
            return (data)
        #return "funciona"
        except:
            return jsonify({'error':'error at ml pipiline'})
    return "Predicicon de Fraude en Transacciones Financieras"