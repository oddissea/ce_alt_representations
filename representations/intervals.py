#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
intervals.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Prueba de evaluación continua 4 - Representaciones Alternativas de Reglas
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 1.0

Descripción:
    Clasificador basado en reglas por intervalos (representación min-max).
    Implementa la extensión natural de la representación ternaria tradicional
    para variables continuas usando intervalos de valores permitidos.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class IntervalClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador basado en reglas por intervalos.

    Cada clase tiene asociada una regla definida por intervalos (min-max)
    para cada atributo numérico.
    """

    def __init__(self):
        self.classes_ = None
        self.rules_ = []
        self._estimator_type = "classifier"  # Definir explícitamente como clasificador

    def fit(self, X, y):
        """
        Entrena el clasificador aprendiendo intervalos mínimos y máximos para cada clase.

        Parámetros:
        - X (array): Datos de entrenamiento.
        - y (array): Etiquetas correspondientes.

        Retorna:
        - self: Instancia entrenada.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.rules_ = []

        for cls in self.classes_:
            X_cls = X[y == cls]
            if len(X_cls) > 0:
                lower_bounds = np.min(X_cls, axis=0)
                upper_bounds = np.max(X_cls, axis=0)
                self.rules_.append({'class': cls, 'lower': lower_bounds, 'upper': upper_bounds})

        return self

    def predict(self, X):
        """
        Realiza predicciones sobre los datos proporcionados.

        Parámetros:
        - X (array): Datos para predecir.

        Retorna:
        - preds (array): Predicciones realizadas.
        """
        X = np.asarray(X)
        preds = []
        for x in X:
            pred = self._predict_single(x)
            preds.append(pred)
        return np.array(preds)

    def _predict_single(self, x):
        """
        Predice la clase para una sola instancia utilizando las reglas aprendidas.

        Parámetros:
        - x (array): Instancia individual.

        Retorna:
        - class (int): Clase predicha (0 por defecto si no hay coincidencia exacta).
        """
        for rule in self.rules_:
            if np.all((x >= rule['lower']) & (x <= rule['upper'])):
                return rule['class']
        # Clase por defecto si no coincide exactamente con ningún intervalo
        return self._default_class()

    def _default_class(self):
        """
        Define la clase predeterminada en caso de que una instancia no coincida con ningún intervalo.

        Por defecto devuelve la primera clase.

        Retorna:
        - class (int): Clase por defecto.
        """
        return self.classes_[0]

    def predict_proba(self, X):
        """
        Estima probabilidades de clase de forma binaria:
        1.0 para la clase predicha, 0.0 para las otras.

        Parámetros:
        - X (array): Datos para predecir.

        Retorna:
        - probs (array): Matriz [n_samples, n_classes] con probabilidades.
        """
        X = np.asarray(X)
        y_pred = self.predict(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probas = np.zeros((n_samples, n_classes))

        for i, pred in enumerate(y_pred):
            class_idx = np.where(self.classes_ == pred)[0][0]
            probas[i, class_idx] = 1.0

        return probas