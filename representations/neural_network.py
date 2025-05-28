#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
neural_network.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Prueba de evaluación continua 4 - Representaciones Alternativas de Reglas
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 1.0

Descripción:
    Clasificador basado en redes neuronales (perceptrón multicapa).
    Implementa la representación subsimbólica donde cada regla contiene
    una pequeña red neuronal que actúa como condición de activación.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier


class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador basado en Redes Neuronales (Perceptrón multicapa - MLP).
    """

    def __init__(self, hidden_layer_sizes=(50,), activation='relu', solver='adam',
                 learning_rate_init=0.001, max_iter=500, random_state=42):
        """
        Inicializa el MLPClassifier con parámetros estándar o personalizados.

        Parámetros:
        - hidden_layer_sizes (tuple): Número de neuronas en las capas ocultas.
        - activation (str): Función de activación ('relu', 'tanh', 'logistic', etc.).
        - solver (str): Optimizador ('adam', 'sgd', 'lbfgs').
        - learning_rate_init (float): Tasa inicial de aprendizaje.
        - max_iter (int): Número máximo de iteraciones de entrenamiento.
        - random_state (int): Semilla para reproducibilidad.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.classes_ = None
        self._estimator_type = "classifier"  # Definir explícitamente como clasificador

    def fit(self, X, y):
        """
        Entrena la red neuronal con los datos proporcionados.

        Parámetros:
        - X (array): Datos de entrenamiento.
        - y (array): Etiquetas correspondientes.

        Retorna:
        - self: Instancia entrenada.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        # Crear y entrenar el modelo
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,  # Detener anticipadamente si no hay mejoras
            validation_fraction=0.1  # Usar 10% como conjunto de validación
        )
        self.model.fit(X, y)
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

        X = np.asarray(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Retorna probabilidades de predicción sobre los datos proporcionados.

        Parámetros:
        - X (array): Datos para predecir.

        Retorna:
        - probs (array): Probabilidades por clase.
        """
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir probabilidades")

        X = np.asarray(X)
        return self.model.predict_proba(X)

    def get_network_structure(self):
        """
        Retorna información sobre la estructura de la red neuronal.

        Retorna:
        - dict: Información estructural de la red
        """
        if self.model is None:
            raise ValueError("El modelo debe ser entrenado primero")

        info = {
            'n_layers': len(self.model.coefs_) + 1,
            'layer_sizes': [len(self.model.coefs_[0])] + [len(w) for w in self.model.coefs_[1:]],
            'n_iterations': self.model.n_iter_,
            'activation': self.activation,
            'solver': self.solver
        }
        return info