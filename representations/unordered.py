#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unordered.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Prueba de evaluación continua 4 - Representaciones Alternativas de Reglas
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 1.0

Descripción:
    Clasificador desordenado basado en cromosomas desordenados (messy chromosomes).
    Implementa codificación donde genes no tienen posiciones fijas predefinidas,
    permitiendo selección automática de atributos relevantes.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier


class UnorderedClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador desordenado basado en reglas generadas mediante árboles de decisión poco profundos,
    permitiendo seleccionar atributos relevantes automáticamente.
    """

    def __init__(self, max_depth=3, criterion='gini', random_state=42):
        """
        Inicializa el clasificador desordenado.

        Parámetros:
        - max_depth (int): Profundidad máxima del árbol, controla complejidad y selecciona atributos relevantes.
        - criterion (str): Criterio para medir calidad de particiones ('gini' o 'entropy').
        - random_state (int): Semilla para reproducibilidad.
        """
        self.max_depth = max_depth
        self.criterion = criterion
        self.random_state = random_state
        self.tree = None
        self.selected_features_ = None
        self.classes_ = None
        self.feature_importances_ = None
        self._estimator_type = "classifier"  # Definir explícitamente como clasificador

    def fit(self, X, y):
        """
        Entrena el clasificador usando un árbol de decisión superficial para seleccionar atributos relevantes.

        Parámetros:
        - X (array): Datos de entrenamiento.
        - y (array): Etiquetas correspondientes.

        Retorna:
        - self: Instancia entrenada.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        # Entrenar árbol inicial para selección de características
        initial_tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            criterion=self.criterion,
            random_state=self.random_state
        )
        initial_tree.fit(X, y)

        # Almacenar importancias de características
        self.feature_importances_ = initial_tree.feature_importances_

        # Seleccionar características relevantes (importancia > 0)
        self.selected_features_ = np.where(self.feature_importances_ > 0)[0]

        # Si no hay características seleccionadas, usar las N más importantes
        if len(self.selected_features_) == 0:
            top_n = min(5, len(self.feature_importances_))
            self.selected_features_ = np.argsort(self.feature_importances_)[-top_n:]

        # Reentrenar el árbol solo con características seleccionadas
        X_selected = X[:, self.selected_features_]
        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            criterion=self.criterion,
            random_state=self.random_state
        )
        self.tree.fit(X_selected, y)

        return self

    def predict(self, X):
        """
        Realiza predicciones usando solo los atributos seleccionados por el árbol superficial.

        Parámetros:
        - X (array): Datos para predecir.

        Retorna:
        - preds (array): Predicciones realizadas.
        """
        if self.tree is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir")

        X = np.asarray(X)
        X_selected = X[:, self.selected_features_]
        return self.tree.predict(X_selected)

    def predict_proba(self, X):
        """
        Predicciones probabilísticas usando atributos seleccionados.

        Parámetros:
        - X (array): Datos para predecir.

        Retorna:
        - probs (array): Probabilidades por clase.
        """
        if self.tree is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir probabilidades")

        X = np.asarray(X)
        X_selected = X[:, self.selected_features_]
        return self.tree.predict_proba(X_selected)

    def get_selected_features(self):
        """
        Retorna los índices de los atributos seleccionados.

        Retorna:
        - array: Índices de atributos seleccionados.
        """
        if self.selected_features_ is None:
            raise ValueError("El modelo debe ser entrenado antes de obtener atributos seleccionados")

        return self.selected_features_

    def get_feature_importances(self):
        """
        Retorna la importancia de cada atributo en el espacio original.

        Retorna:
        - array: Importancia de cada atributo
        """
        if self.feature_importances_ is None:
            raise ValueError("El modelo debe ser entrenado antes de obtener importancias")

        return self.feature_importances_