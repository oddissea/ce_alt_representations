#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hyperellipsoids.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Prueba de evaluación continua 4 - Representaciones Alternativas de Reglas
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 1.0

Descripción:
    Clasificador basado en reglas definidas por hiperelipsoides.
    Utiliza distancia de Mahalanobis para definir regiones de decisión
    elipsoidales que pueden capturar correlaciones entre variables.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class HyperEllipsoidClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador basado en reglas definidas por hiperelipsoides usando distancia de Mahalanobis.

    Cada clase se representa por un hiperelipsoide definido por su centro (media)
    y su matriz de covarianza.
    """

    def __init__(self):
        """
        Inicializa los atributos internos.
        """
        self.classes_ = None
        self.ellipsoids_ = []
        self._estimator_type = "classifier"  # Definir explícitamente como clasificador

    def fit(self, X, y):
        """
        Calcula la media y la matriz de covarianza para cada clase.

        Parámetros:
        - X (array): Datos de entrenamiento.
        - y (array): Etiquetas correspondientes.

        Retorna:
        - self: Instancia entrenada.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.ellipsoids_ = []

        for cls in self.classes_:
            X_cls = X[y == cls]
            if len(X_cls) > 0:
                center = np.mean(X_cls, axis=0)
                # Usar regularización más fuerte para matrices mal condicionadas
                covariance = np.cov(X_cls, rowvar=False) + np.eye(X_cls.shape[1]) * 1e-3
                try:
                    inv_covariance = np.linalg.inv(covariance)
                    threshold = self._calculate_threshold(X_cls, center, inv_covariance)
                    self.ellipsoids_.append({
                        'class': cls,
                        'center': center,
                        'inv_cov': inv_covariance,
                        'threshold': threshold
                    })
                except np.linalg.LinAlgError:
                    # Fallback para matrices singulares: usar matriz identidad
                    print(f"Advertencia: Matriz singular para clase {cls}. Usando matriz identidad.")
                    inv_covariance = np.eye(X_cls.shape[1])
                    threshold = np.max(np.linalg.norm(X_cls - center, axis=1)) * 1.1
                    self.ellipsoids_.append({
                        'class': cls,
                        'center': center,
                        'inv_cov': inv_covariance,
                        'threshold': threshold,
                        'fallback': True
                    })

        return self

    def predict(self, X):
        """
        Realiza predicciones sobre los datos proporcionados.

        Parámetros:
        - X (array): Datos para predecir.

        Retorna:
        - preds (array): Predicciones realizadas.
        """
        X = np.asarray(X, dtype=np.float64)
        preds = []
        for x in X:
            pred = self._predict_single(x)
            preds.append(pred)
        return np.array(preds)

    def _predict_single(self, x):
        """
        Predice la clase para una sola instancia utilizando los hiperelipsoides aprendidos.

        Parámetros:
        - x (array): Instancia individual.

        Retorna:
        - class (int): Clase predicha, basada en la menor distancia de Mahalanobis dentro del umbral.
        """
        distances = []
        for ellipsoid in self.ellipsoids_:
            if 'fallback' in ellipsoid and ellipsoid['fallback']:
                # Caso alternativo: verificar distancia euclidiana
                dist = np.linalg.norm(x - ellipsoid['center'])
                if dist <= ellipsoid['threshold']:
                    distances.append((dist, ellipsoid['class']))
            else:
                try:
                    # Calcular distancia de Mahalanobis manualmente para control de errores
                    diff = x - ellipsoid['center']
                    dist = np.sqrt(np.dot(np.dot(diff, ellipsoid['inv_cov']), diff))
                    if dist <= ellipsoid['threshold']:
                        distances.append((dist, ellipsoid['class']))
                except (ValueError, TypeError, ArithmeticError, OverflowError) as e:
                    # Manejo más específico de excepciones
                    print(f"Error en cálculo de distancia: {e}")
                    # Si hay error, usar distancia euclidiana
                    dist = np.linalg.norm(x - ellipsoid['center'])
                    if dist <= ellipsoid['threshold']:
                        distances.append((dist, ellipsoid['class']))

        if distances:
            # Devuelve la clase con la distancia más corta
            sorted_distances = sorted(distances, key=lambda d: d[0])
            return sorted_distances[0][1]  # Obtener solo el valor de clase
        else:
            # Clase por defecto si ninguna clase se ajusta al umbral
            return self._default_class()

    @staticmethod
    def _calculate_threshold(X_cls, center, inv_cov):
        """
        Calcula un umbral para el hiperelipsoide basado en percentil (ej. 95%).

        Parámetros:
        - X_cls (array): Instancias de la clase específica.
        - center (array): Centro del hiperelipsoide.
        - inv_cov (array): Matriz de covarianza inversa.

        Retorna:
        - threshold (float): Valor umbral de distancia.
        """
        try:
            distances = []
            for x in X_cls:
                diff = x - center
                dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
                distances.append(dist)
            threshold = np.percentile(distances, 95)
            return threshold
        except (ValueError, TypeError, ArithmeticError, OverflowError) as e:
            # Manejo más específico de excepciones
            print(f"Error en cálculo de umbral: {e}")
            # En caso de error, usar distancia euclidiana
            distances = np.linalg.norm(X_cls - center, axis=1)
            return np.percentile(distances, 95)

    def _default_class(self):
        """
        Clase predeterminada cuando una instancia no encaja en ningún hiperelipsoide.

        Retorna:
        - class (int): Clase por defecto (primera clase).
        """
        return self.classes_[0]

    def predict_proba(self, X):
        """
        Estima probabilidades de clase basadas en distancias inversas.

        Parámetros:
        - X (array): Datos para predecir.

        Retorna:
        - array: Matriz [n_samples, n_classes] con probabilidades para cada clase.
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))

        for i, x in enumerate(X):
            # Inicializar pesos para cada clase
            weights = np.zeros(n_classes)
            total_weight = 0

            # Calcular pesos para cada clase basados en distancia inversa
            for j, cls in enumerate(self.classes_):
                for ellipsoid in self.ellipsoids_:
                    if ellipsoid['class'] == cls:
                        try:
                            # Calcular distancia e inversarla para peso
                            if 'fallback' in ellipsoid and ellipsoid['fallback']:
                                # Asegurar que tanto x como center sean arrays numpy
                                x_array = np.asarray(x, dtype=np.float64)
                                center_array = np.asarray(ellipsoid['center'], dtype=np.float64)
                                dist = np.linalg.norm(x_array - center_array)
                            else:
                                # Asegurar que tanto x como center sean arrays numpy
                                x_array = np.asarray(x, dtype=np.float64)
                                center_array = np.asarray(ellipsoid['center'], dtype=np.float64)
                                diff = x_array - center_array
                                dist = np.sqrt(np.dot(np.dot(diff, ellipsoid['inv_cov']), diff))

                            # Transformar distancia a peso (valor más alto = más cercano)
                            weight_value = 1.0 / (1.0 + dist)
                            weights[j] += weight_value
                            total_weight += weight_value
                            break
                        except (ValueError, TypeError, ArithmeticError, OverflowError) as e:
                            # Manejo más específico de excepciones
                            print(f"Error en cálculo de probabilidad: {e}")
                            # Ignorar errores en cálculo de distancia

            # Normalizar pesos a probabilidades
            if total_weight > 0:
                proba[i] = weights / total_weight
            else:
                # Si todos fallan, usar clase por defecto
                default_idx = np.where(self.classes_ == self._default_class())[0][0]
                proba[i, default_idx] = 1.0

        return proba