#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convex_hulls.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Prueba de evaluación continua 4 - Representaciones Alternativas de Reglas
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 1.0

Descripción:
    Implementación de clasificador basado en envolturas convexas.
    Utiliza triangulación Delaunay para definir regiones de decisión
    convexas en espacios de características reducidos mediante PCA.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA


class ConvexHullClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador basado en reglas definidas por envolturas convexas.

    Para espacios de alta dimensionalidad, primero reduce la dimensión usando PCA
    y luego aplica la triangulación Delaunay.
    """

    def __init__(self, n_components=10):
        """
        Inicializa el clasificador.

        Parámetros:
        - n_components: Número de componentes principales a utilizar para reducción
          de dimensionalidad antes de construir envolturas convexas.
        """
        self.n_components = n_components
        self.classes_ = None
        self.hulls_ = []
        self.pca = None
        self._estimator_type = "classifier"  # Definir explícitamente como clasificador

    @staticmethod
    def _create_sphere_fallback(X_cls, cls, reason=""):
        """
        Crea una esfera como alternativa a la envoltura convexa.

        Parámetros:
        - X_cls (array): Datos de la clase específica
        - cls: Etiqueta de la clase
        - reason (str): Razón por la que se usa el fallback

        Retorna:
        - dict: Representación de la esfera
        """
        center = np.mean(X_cls, axis=0)
        radius = np.max(np.linalg.norm(X_cls - center, axis=1)) * 1.1
        if reason:
            print(f"Advertencia: {reason} para clase {cls}. Usando esfera alternativa.")
        return {
            'class': cls,
            'center': center,
            'radius': radius,
            'fallback': True
        }

    def fit(self, X, y):
        """
        Construye las envolturas convexas para cada clase usando Delaunay en espacio reducido.

        Parámetros:
        - X (array): Datos de entrenamiento.
        - y (array): Etiquetas correspondientes.

        Retorna:
        - self: Instancia entrenada.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.hulls_ = []

        # Primero reducir dimensionalidad con PCA
        n_comp = min(self.n_components, X.shape[1])
        self.pca = PCA(n_components=n_comp)
        X_reduced = self.pca.fit_transform(X)

        # Ahora construir los hulls en el espacio reducido
        from scipy.spatial import Delaunay

        for cls in self.classes_:
            X_cls_reduced = X_reduced[y == cls]

            # Verificar si hay suficientes puntos para formar un simplex
            if len(X_cls_reduced) < X_reduced.shape[1] + 1:
                # Usar los datos originales para el fallback por claridad
                X_cls = X[y == cls]
                self.hulls_.append(self._create_sphere_fallback(
                    X_cls, cls, "Pocos puntos"
                ))
                continue

            # Delaunay triangulation para envoltura convexa
            try:
                # Agregar ruido muy pequeño para evitar coplanaridad
                np.random.seed(42)  # Para reproducibilidad
                X_noise = X_cls_reduced + np.random.normal(0, 1e-5, X_cls_reduced.shape)
                hull = Delaunay(X_noise)
                self.hulls_.append({
                    'class': cls,
                    'hull': hull,
                    'reduced': True
                })
            except Exception as e:
                # Alternativa: usar una esfera simple en caso de error
                X_cls = X[y == cls]
                self.hulls_.append(self._create_sphere_fallback(
                    X_cls, cls, f"Error en envoltura convexa: {e}"
                ))

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
        Predice la clase para una instancia individual utilizando las envolturas convexas.

        Parámetros:
        - x (array): Instancia individual.

        Retorna:
        - class (int): Clase predicha (o clase por defecto si no está contenida).
        """
        # Inicializar x_reduced para evitar error de referencia
        x_reduced = None

        # Si tenemos PCA, transformar la entrada
        if self.pca is not None:
            x_reduced = self.pca.transform([x])[0]

        # Primero intentamos con los hulls para mayor precisión
        for hull_dict in self.hulls_:
            if 'hull' in hull_dict:
                try:
                    hull = hull_dict['hull']

                    # Verificar si usamos espacio reducido o completo
                    if 'reduced' in hull_dict and hull_dict['reduced']:
                        if hull.find_simplex(x_reduced) >= 0:
                            return hull_dict['class']
                    else:
                        if hull.find_simplex(x) >= 0:
                            return hull_dict['class']
                except (ValueError, TypeError, AttributeError, IndexError) as e:
                    # Ignorar errores en el cálculo del simplex
                    print(f"Error calculando simplex: {e}")
                    pass

        # Si no hay coincidencia con hulls, intentamos con fallbacks
        distances = []
        for hull_dict in self.hulls_:
            if 'fallback' in hull_dict and hull_dict['fallback']:
                # Convertir a arrays numpy para evitar problemas de tipo
                x_array = np.asarray(x, dtype=np.float64)
                center_array = np.asarray(hull_dict['center'], dtype=np.float64)
                dist = np.linalg.norm(x_array - center_array)
                if dist <= hull_dict['radius']:
                    distances.append((dist, hull_dict['class']))

        if distances:
            # Devuelve la clase con la distancia más corta
            sorted_distances = sorted(distances, key=lambda d: d[0])
            return sorted_distances[0][1]  # Obtener solo el valor de clase

        # Si no hay coincidencia, usar clase por defecto
        return self._default_class()

    def _default_class(self):
        """
        Define la clase predeterminada en caso de que una instancia no coincida con ninguna envoltura convexa.

        Retorna:
        - class (int): Clase por defecto (primera clase).
        """
        return self.classes_[0]

    def predict_proba(self, X):
        """
        Estima probabilidades de clase basadas en pertenencia a los hulls.
        Si un punto pertenece a un hull, su probabilidad es 1, de lo contrario 0.

        Parámetros:
        - X (array): Datos para predecir.

        Retorna:
        - array: Probabilidades por clase
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))

        for i, x in enumerate(X):
            pred_class = self._predict_single(x)
            class_idx = np.where(self.classes_ == pred_class)[0][0]
            proba[i, class_idx] = 1.0

        return proba