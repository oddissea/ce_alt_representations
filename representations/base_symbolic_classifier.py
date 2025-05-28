#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_symbolic_classifier.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Prueba de evaluación continua 4 - Representaciones Alternativas de Reglas
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 1.0

Descripción:
    Clase base para clasificadores basados en programación genética simbólica.
    Proporciona funcionalidad común para representaciones simbólicas con
    fallback a RandomForest si gplearn no está disponible.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_X_y, check_array


class BaseSymbolicClassifier(BaseEstimator, ClassifierMixin):
    """
    Clase base para clasificadores basados en programación genética simbólica.
    Usa RandomForest como fallback si gplearn no está disponible.
    """

    def __init__(self, **kwargs):
        """
        Inicializa el clasificador base con parámetros personalizables.

        Parámetros:
        - kwargs: Parámetros para configurar SymbolicClassifier
        """
        # No llamar a super().__init__() aquí, puede causar problemas en la herencia múltiple
        self.kwargs = kwargs
        self.model = None
        self.program_str = None
        self.classes_ = None
        self.using_fallback = False

    def _create_classifier(self):
        """
        Método para crear la instancia del clasificador simbólico.
        Debe ser implementado por las subclases.
        """
        raise NotImplementedError("Las subclases deben implementar este método")

    def fit(self, X, y):
        """
        Entrena el modelo simbólico.

        Parámetros:
        - X (array): Datos de entrenamiento.
        - y (array): Etiquetas correspondientes.

        Retorna:
        - self: Instancia entrenada.
        """
        try:
            # Convertir a arrays de numpy y asegurar el tipo correcto
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y)

            # Si X es 1D, convertirlo a 2D
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            # Validación de datos con mejor manejo de tipos
            X, y = check_X_y(X, y, force_all_finite=True, dtype=np.float64)

            # Almacenar clases
            self.classes_ = np.unique(y)

            # Crear y entrenar el modelo
            try:
                # Si el modelo ya existe, no lo recreamos
                if self.model is None:
                    self.model = self._create_classifier()

                # Entrenar el modelo
                self.model.fit(X, y)

                # Almacenar expresión si está disponible
                if hasattr(self.model, '_program'):
                    self.program_str = str(self.model._program)
                elif hasattr(self.model, 'estimators_'):
                    # Si es RandomForest, dar una representación básica
                    self.program_str = f"Conjunto de {len(self.model.estimators_)} árboles de decisión"
                else:
                    self.program_str = "Expresión no disponible"

                return self

            except Exception as e:
                print(f"Error en fit: {e}")
                # Fallback a RandomForest si hay error
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                )
                self.using_fallback = True
                self.model.fit(X, y)
                self.program_str = "Modelo RandomForest (fallback)"
                return self

        except Exception as e:
            print(f"Error grave en fit: {e}")
            # Fallback extremo si incluso la conversión falla
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            self.using_fallback = True

            # Intentar hacer una última conversión más segura
            try:
                X_safe = np.array(X, dtype=float)
                y_safe = np.array(y, dtype=int)
                self.model.fit(X_safe, y_safe)
                self.classes_ = np.unique(y_safe)
            except Exception as inner_e:
                print(f"Error crítico, incluso con fallback: {inner_e}")

            self.program_str = "Modelo RandomForest (fallback extremo)"
            return self

    def predict(self, X):
        """
        Realiza predicciones sobre los datos proporcionados.

        Parámetros:
        - X (array): Datos para predecir.

        Retorna:
        - preds (array): Predicciones realizadas.
        """
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir")

        try:
            # Convertir a array numpy con tipo explícito
            X = np.asarray(X, dtype=np.float64)

            # Si X es 1D, convertirlo a 2D
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            # Validar datos con tipo explícito
            X = check_array(X, force_all_finite=True, dtype=np.float64)

            return self.model.predict(X)

        except Exception as e:
            print(f"Error en predict: {e}")
            # Intentar conversión más simple en caso de error
            X_safe = np.array(X, dtype=float)
            if X_safe.ndim == 1:
                X_safe = X_safe.reshape(-1, 1)
            return self.model.predict(X_safe)

    def predict_proba(self, X):
        """
        Predicciones probabilísticas sobre los datos proporcionados.

        Parámetros:
        - X (array): Datos para predecir.

        Retorna:
        - probs (array): Probabilidades por clase.
        """
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir probabilidades")

        try:
            # Convertir a array numpy con tipo explícito
            X = np.asarray(X, dtype=np.float64)

            # Si X es 1D, convertirlo a 2D
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            # Validar datos con tipo explícito
            X = check_array(X, force_all_finite=True, dtype=np.float64)

            # Verificar si el modelo tiene predict_proba
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                # Fallback para modelos sin predict_proba
                y_pred = self.predict(X)

                # Obtener dimensiones de manera más segura
                try:
                    # Intentar obtener n_samples de y_pred que es un array
                    n_samples = len(y_pred)
                except (TypeError, AttributeError):
                    # Si y_pred no tiene longitud, convertirlo a lista
                    y_pred = list(y_pred)
                    n_samples = len(y_pred)

                n_classes = len(self.classes_)
                proba = np.zeros((n_samples, n_classes))

                for i, pred in enumerate(y_pred):
                    try:
                        class_idx = np.where(self.classes_ == pred)[0][0]
                        proba[i, class_idx] = 1.0
                    except (IndexError, TypeError):
                        # En caso de error asignar a la primera clase
                        proba[i, 0] = 1.0

                return proba

        except Exception as e:
            print(f"Error en predict_proba: {e}")
            # Intentar conversión más simple en caso de error
            X_safe = np.array(X, dtype=float)
            if X_safe.ndim == 1:
                X_safe = X_safe.reshape(-1, 1)

            # Verificar si el modelo tiene predict_proba
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X_safe)
            else:
                # Fallback para modelos sin predict_proba
                y_pred = self.predict(X_safe)
                # Usar X_safe que ya es un array numpy
                n_samples = X_safe.shape[0]
                n_classes = len(self.classes_)
                proba = np.zeros((n_samples, n_classes))

                for i, pred in enumerate(y_pred):
                    class_idx = np.where(self.classes_ == pred)[0][0]
                    proba[i, class_idx] = 1.0

                return proba

    def get_expression(self):
        """
        Devuelve la expresión simbólica encontrada durante el entrenamiento.

        Retorna:
        - str: Expresión simbólica como cadena.
        """
        if hasattr(self, 'program_str') and self.program_str:
            return self.program_str
        else:
            return "Modelo no entrenado aún"