#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gene_expressions.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Prueba de evaluación continua 4 - Representaciones Alternativas de Reglas
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 2.0 - Nivel 2 Improvements

Descripción:
    Clasificador basado en expresiones de genes (programación de expresión génica).
    Utiliza codificación lineal fija que se traduce a expresiones arbóreas,
    combinando ventajas de algoritmos genéticos y programación genética.

    MEJORA NIVEL 2: Implementación nativa de GP con restricciones para emular GEP.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier


class GeneExpressionsClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador basado en expresiones de genes usando programación genética nativa
    con restricciones para emular programación de expresión génica (GEP).
    """

    def __init__(self, population_size=150, generations=15, tournament_size=5,
                 crossover_prob=0.7, mutation_prob=0.3, max_depth=4,
                 parsimony_coefficient=0.05, random_state=42):
        """
        Inicializa el clasificador simbólico con configuración para expresiones de genes.

        Parámetros similares a ExpressionsSClassifier pero con valores optimizados
        para emular el comportamiento de expresiones génicas (árboles más simples,
        mayor penalización por complejidad).
        """
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_depth = max_depth  # Más restrictivo que expressions_s
        self.parsimony_coefficient = parsimony_coefficient  # Mayor penalización
        self.random_state = random_state

        # Estado interno
        self.model = None
        self.classes_ = None
        self.using_fallback = False
        self.program_str = None
        self._estimator_type = "classifier"

    def fit(self, X, y):
        """
        Entrena el modelo de expresiones génicas usando programación genética nativa
        con restricciones para emular GEP.

        Parámetros:
        - X (array): Datos de entrenamiento.
        - y (array): Etiquetas correspondientes.

        Retorna:
        - self: Instancia entrenada.
        """
        try:
            # Intentar usar implementación nativa con configuración GEP
            from representations.native_gp import NativeSymbolicClassifier

            self.model = NativeSymbolicClassifier(
                population_size=self.population_size,
                generations=self.generations,
                tournament_size=self.tournament_size,
                crossover_prob=self.crossover_prob,
                mutation_prob=self.mutation_prob,
                max_depth=self.max_depth,  # Más restrictivo
                parsimony_coefficient=self.parsimony_coefficient,  # Mayor penalización
                random_state=self.random_state
            )

            print("Usando GP nativa configurada para expresiones génicas...")
            self.model.fit(X, y)
            self.classes_ = self.model.classes_
            self.program_str = self.model.get_expression()
            self.using_fallback = False

            # Verificar que el árbol generado respete las restricciones GEP
            complexity = self.model.get_complexity()
            if complexity['depth'] > self.max_depth or complexity['size'] > 50:
                print(
                    f"Advertencia: Árbol excede restricciones GEP (profundidad: {complexity['depth']}, tamaño: {complexity['size']})")

            print(f"Expresión génica completada. Expresión: {self.program_str[:100]}...")

        except Exception as e:
            print(f"Error con GP nativa para expresiones génicas: {e}")
            print("Usando GradientBoosting como fallback...")

            # Fallback a GradientBoosting (más apropiado que RandomForest para expresiones)
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=self.max_depth,
                learning_rate=0.1,
                random_state=self.random_state
            )
            self.model.fit(X, y)
            self.classes_ = np.unique(y)
            self.program_str = f"Ensamble gradiente con {self.model.n_estimators} estimadores (fallback)"
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
        Devuelve la expresión génica encontrada durante el entrenamiento.

        Retorna:
        - str: Expresión génica como cadena.
        """
        if self.program_str:
            return self.program_str
        else:
            return "Modelo no entrenado aún"

    def get_complexity(self):
        """
        Obtiene medidas de complejidad del modelo génico.

        Retorna:
        - dict: Diccionario con métricas de complejidad
        """
        if self.model is None:
            return {'size': 0, 'depth': 0, 'using_fallback': True, 'gep_compliant': False}

        if hasattr(self.model, 'get_complexity'):
            complexity = self.model.get_complexity()
            complexity['using_fallback'] = self.using_fallback
            # Verificar cumplimiento de restricciones GEP
            complexity['gep_compliant'] = (
                    complexity['depth'] <= self.max_depth and
                    complexity['size'] <= 50
            )
            return complexity
        else:
            return {
                'size': getattr(self.model, 'n_estimators', 100),
                'depth': getattr(self.model, 'max_depth', self.max_depth),
                'using_fallback': self.using_fallback,
                'gep_compliant': False
            }

    def is_gep_compliant(self):
        """
        Verifica si el modelo cumple con las restricciones de expresiones génicas.

        Retorna:
        - bool: True si cumple restricciones GEP
        """
        complexity = self.get_complexity()
        return complexity.get('gep_compliant', False)