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
Versión: 1.0

Descripción:
    Clasificador basado en expresiones de genes (programación de expresión génica).
    Utiliza codificación lineal fija que se traduce a expresiones arbóreas,
    combinando ventajas de algoritmos genéticos y programación genética.
"""
from representations.base_symbolic_classifier import BaseSymbolicClassifier


class GeneExpressionsClassifier(BaseSymbolicClassifier):
    """
    Clasificador simplificado basado en expresiones simbólicas de genes (codificación lineal),
    usando GPlearn con restricciones para controlar complejidad.
    """

    def __init__(self, population_size=300, generations=15, tournament_size=15,
                 max_samples=0.9, parsimony_coefficient=0.05, random_state=42):
        """
        Inicializa el clasificador simbólico con configuración para expresiones de genes.
        """
        # Establecer atributos primero
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.max_samples = max_samples
        self.parsimony_coefficient = parsimony_coefficient
        self.random_state = random_state

        # Llamar al constructor padre sin parámetros
        super().__init__()

    def _create_classifier(self):
        """
        Crea un clasificador simbólico con configuración para expresiones de genes.

        Retorna:
        - SymbolicClassifier: Instancia configurada.
        """
        try:
            # Intentar usar gplearn
            from gplearn.genetic import SymbolicClassifier

            # Crear nuevo clasificador con los parámetros guardados
            return SymbolicClassifier(
                population_size=self.population_size,
                generations=self.generations,
                tournament_size=self.tournament_size,
                const_range=(-1.0, 1.0),
                init_depth=(2, 4),
                init_method='half and half',
                function_set=('add', 'sub', 'mul', 'div'),
                metric='log loss',
                parsimony_coefficient=self.parsimony_coefficient,
                max_samples=self.max_samples,
                random_state=self.random_state,
                verbose=1
            )
        except (ImportError, AttributeError) as e:
            # Más específico con tipos de error
            print(f"No se pudo crear SymbolicClassifier: {e}")
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state
            )