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
Versión: 1.0

Descripción:
    Clasificador basado en expresiones simbólicas S (árboles de decisión evolutivos)
    mediante programación genética. Implementa reglas con estructura de expresiones
    S de Lisp para ampliar el modelo conjuntivo tradicional.
"""
from representations.base_symbolic_classifier import BaseSymbolicClassifier


class ExpressionsSClassifier(BaseSymbolicClassifier):
    """
    Clasificador basado en expresiones simbólicas (árboles de decisión evolutivos)
    mediante programación genética.
    """

    def __init__(self, population_size=500, generations=20, tournament_size=20,
                 stopping_criteria=0.95, random_state=42):
        """
        Inicializa el clasificador simbólico con configuración para expresiones S.
        """
        # Establecer atributos primero
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
        self.random_state = random_state

        # Llamar al constructor padre sin parámetros
        super().__init__()

    def _create_classifier(self):
        """
        Crea un clasificador simbólico con configuración para expresiones S.

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
                stopping_criteria=self.stopping_criteria,
                const_range=(-1.0, 1.0),
                init_depth=(2, 6),
                init_method='half and half',
                function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv'),
                metric='log loss',
                parsimony_coefficient=0.01,
                random_state=self.random_state,
                verbose=1
            )
        except (ImportError, AttributeError) as e:
            # Más específico con tipos de error
            print(f"No se pudo crear SymbolicClassifier: {e}")
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=self.random_state
            )