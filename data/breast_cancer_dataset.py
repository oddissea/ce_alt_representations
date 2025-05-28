#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
breast_cancer_dataset.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Prueba de evaluación continua 4 - Representaciones Alternativas de Reglas
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 1.0

Descripción:
    Módulo para carga y preprocesamiento del dataset Wisconsin Breast Cancer.
    Proporciona funciones para cargar el dataset original completo y la versión
    con selección de características para validación experimental comparativa.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os


def load_and_preprocess(test_size=0.2, random_state=42):
    """
    Carga el dataset Breast Cancer Wisconsin y realiza preprocesamiento estándar.

    Parámetros:
    - test_size (float): Proporción del conjunto de datos reservado para validación.
    - random_state (int): Semilla para reproducibilidad.

    Retorna:
    - X_train, X_test, y_train, y_test: Conjuntos divididos y estandarizados.
    """
    # Cargar datos
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Estandarizar atributos (media=0, desviación estándar=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir en entrenamiento y prueba (estratificado)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def load_full_dataset():
    """
    Carga el dataset completo sin división, útil para validación cruzada estratificada.

    Retorna:
    - X_scaled, y: Dataset completo y estandarizado.
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def _find_prepared_data_path():
    """
    Función auxiliar para encontrar la ruta del archivo de datos preparados.

    Retorna:
    - str: Ruta al archivo de datos preparados o None si no se encuentra.
    """
    # Rutas a probar en orden de preferencia
    possible_paths = [
        'data/processed/prepared_data.pkl',  # Ruta correcta según tu notebook
        os.path.join('data', 'processed', 'prepared_data.pkl'),
        'data/prepared/prepared_data.pkl',  # Ruta anterior
        os.path.join('data', 'prepared', 'prepared_data.pkl'),
        'prepared_data.pkl',
        os.path.join('prepared', 'prepared_data.pkl'),
        os.path.join('..', 'data', 'processed', 'prepared_data.pkl'),
        os.path.join('..', 'data', 'prepared', 'prepared_data.pkl')
    ]

    # Probar cada ruta
    for path in possible_paths:
        if os.path.exists(path):
            return path

    # No se encontró el archivo
    return None


# Añadir esta función a breast_cancer_dataset.py

def load_prepared_dataset_from_csv():
    """
    Carga el dataset preparado desde archivos CSV.

    Retorna:
    - X_prepared, y: Dataset con características seleccionadas.
    """
    try:
        import pandas as pd

        # Rutas de los archivos CSV
        csv_paths = {
            'X_train': 'data/processed/X_train.csv',
            'X_test': 'data/processed/X_test.csv',
            'y_train': 'data/processed/y_train.csv',
            'y_test': 'data/processed/y_test.csv'
        }

        # Verificar que todos los archivos existen
        for name, path in csv_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"No se encontró el archivo {path}")

        # Cargar desde CSV
        X_train = pd.read_csv(csv_paths['X_train']).values
        X_test = pd.read_csv(csv_paths['X_test']).values
        y_train = pd.read_csv(csv_paths['y_train']).values.ravel()
        y_test = pd.read_csv(csv_paths['y_test']).values.ravel()

        # Combinar para validación cruzada
        X_prepared = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)

        print(
            f"Dataset preparado cargado desde CSV: {X_prepared.shape[0]} muestras, {X_prepared.shape[1]} características.")
        print(f"  - Conjunto de entrenamiento: {X_train.shape[0]} muestras")
        print(f"  - Conjunto de prueba: {X_test.shape[0]} muestras")

        return X_prepared, y

    except Exception as e:
        print(f"Error cargando desde CSV: {e}")
        print("Regresando al dataset original como fallback...")
        return load_full_dataset()


def load_prepared_dataset():
    """
    Carga el dataset preparado con selección de características.
    Primero intenta desde pickle, luego desde CSV, finalmente desde dataset original.

    Retorna:
    - X_prepared, y: Dataset con características seleccionadas.
    """
    try:
        # Primero intentar cargar desde archivo pickle
        prepared_path = _find_prepared_data_path()

        if prepared_path is not None:
            # Cargar desde el archivo pickle
            with open(prepared_path, 'rb') as f:
                data_dict = pickle.load(f)

            # Verificar el contenido del diccionario y combinar los conjuntos
            if all(key in data_dict for key in ['X_train', 'X_test', 'y_train', 'y_test']):
                # Combinar los conjuntos de entrenamiento y prueba para validación cruzada
                X_train = np.asarray(data_dict['X_train'])
                X_test = np.asarray(data_dict['X_test'])
                y_train = np.asarray(data_dict['y_train'])
                y_test = np.asarray(data_dict['y_test'])

                # Concatenar para obtener el conjunto completo
                X_prepared = np.concatenate([X_train, X_test], axis=0)
                y = np.concatenate([y_train, y_test], axis=0)

                print(
                    f"Dataset preparado cargado desde pickle: {X_prepared.shape[0]} muestras, {X_prepared.shape[1]} características.")
                return X_prepared, y

    except Exception as e:
        print(f"Error al cargar desde pickle: {e}")

    # Si falla el pickle, intentar cargar desde CSV
    print("Intentando cargar desde archivos CSV...")
    return load_prepared_dataset_from_csv()

def get_prepared_feature_names():
    """
    Obtiene los nombres de las características del dataset preparado.

    Retorna:
    - list: Nombres de las características seleccionadas.
    """
    try:
        # Encontrar la ruta del archivo
        prepared_path = _find_prepared_data_path()

        if prepared_path is None:
            raise FileNotFoundError("No se encontró el archivo de datos preparados")

        # Cargar desde el archivo
        with open(prepared_path, 'rb') as f:
            data_dict = pickle.load(f)

        # Verificar si existen nombres de características
        if 'feature_names' in data_dict:
            return data_dict['feature_names']
        else:
            # Intentar inferir la forma del array
            if 'X' in data_dict:
                X = data_dict['X']
            else:
                # Tomar el primer valor del diccionario que probablemente sea X
                X = next(iter(data_dict.values()))

            # Crear nombres genéricos basados en la dimensión
            return [f"Feature_{i}" for i in range(X.shape[1])]

    except Exception as e:
        print(f"Error al obtener nombres de características: {e}")
        # Retornar nombres genéricos como fallback
        return [f"Feature_{i}" for i in range(17)]  # Asumiendo 17 características