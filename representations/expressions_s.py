#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expressions_s.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Prueba de evaluación continua 4 - Representaciones Alternativas de Reglas
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 2.0 - Nivel 2 Improvements

Descripción:
    Clasificador basado en expresiones simbólicas S (árboles de decisión evolutivos)
    mediante programación genética nativa. Implementa reglas con estructura de expresiones
    S de Lisp para ampliar el modelo conjuntivo tradicional.

    MEJORA NIVEL 2: Implementación nativa de GP eliminando dependencia de gplearn.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier


class ExpressionsSClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador basado en expresiones simbólicas (árboles de decisión evolutivos)
    mediante programación genética nativa.
    """

    def __init__(self, population_size=100, generations=20, tournament_size=7,
                 crossover_prob=0.8, mutation_prob=0.2, max_depth=6,
                 parsimony_coefficient=0.01, random_state=42):
        """
        Inicializa el clasificador simbólico con configuración para expresiones S.

        Parámetros:
        - population_size (int): Tamaño de la población para GP
        - generations (int): Número de generaciones evolutivas
        - tournament_size (int): Tamaño del torneo para selección
        - crossover_prob (float): Probabilidad de cruce
        - mutation_prob (float): Probabilidad de mutación
        - max_depth (int): Profundidad máxima de los árboles
        - parsimony_coefficient (float): Coeficiente de penalización por complejidad
        - random_state (int): Semilla para reproducibilidad
        """
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_depth = max_depth
        self.parsimony_coefficient = parsimony_coefficient
        self.random_state = random_state

        # Estado interno
        self.model = None
        self.classes_ = None
        self.using_fallback = False
        self.program_str = None
        self._estimator_type = "classifier"

    def fit(self, X, y):
        """
        Entrena el modelo simbólico usando programación genética nativa.

        Parámetros:
        - X (array): Datos de entrenamiento.
        - y (array): Etiquetas correspondientes.

        Retorna:
        - self: Instancia entrenada.
        """
        try:
            # Intentar usar implementación nativa
            from representations.native_gp import NativeSymbolicClassifier

            self.model = NativeSymbolicClassifier(
                population_size=self.population_size,
                generations=self.generations,
                tournament_size=self.tournament_size,
                crossover_prob=self.crossover_prob,
                mutation_prob=self.mutation_prob,
                max_depth=self.max_depth,
                parsimony_coefficient=self.parsimony_coefficient,
                random_state=self.random_state
            )

            print("Usando implementación nativa de programación genética...")
            self.model.fit(X, y)
            self.classes_ = self.model.classes_
            self.program_str = self.model.get_expression()
            self.using_fallback = False

            print(f"GP nativa completada. Expresión generada: {self.program_str[:100]}...")

        except Exception as e:
            print(f"Error con GP nativa: {e}")
            print("Usando RandomForest como fallback...")

            # Fallback a RandomForest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=self.random_state
            )
            self.model.fit(X, y)
            self.classes_ = np.unique(y)
            self.program_str = f"Conjunto de {self.model.n_estimators} árboles de decisión (fallback)"
            self.using_fallback = True

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

        return self.model.predict(X)

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

        return self.model.predict_proba(X)

    def get_expression(self):
        """
        Devuelve la expresión simbólica encontrada durante el entrenamiento.

        Retorna:
        - str: Expresión simbólica como cadena.
        """
        if self.program_str:
            return self.program_str
        else:
            return "Modelo no entrenado aún"

    def get_complexity(self):
        """
        Obtiene medidas de complejidad del modelo.

        Retorna:
        - dict: Diccionario con métricas de complejidad
        """
        if self.model is None:
            return {'size': 0, 'depth': 0, 'using_fallback': True}

        if hasattr(self.model, 'get_complexity'):
            complexity = self.model.get_complexity()
            complexity['using_fallback'] = self.using_fallback
            return complexity
        else:
            return {
                'size': getattr(self.model, 'n_estimators', 100),
                'depth': getattr(self.model, 'max_depth', 8),
                'using_fallback': self.using_fallback
            }