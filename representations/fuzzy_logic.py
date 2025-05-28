#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fuzzy_logic.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Prueba de evaluación continua 4 - Representaciones Alternativas de Reglas
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 1.0

Descripción:
    Clasificador basado en lógica difusa para sistemas clasificadores evolutivos.
    Implementa conjuntos difusos y reglas con grados de pertenencia para manejar
    incertidumbre y conocimiento impreciso en espacios de características.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
import warnings


class FuzzyLogicClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador simple basado en lógica difusa.
    """

    def __init__(self, n_bins=3):
        """
        Inicializa el clasificador difuso.

        Parámetros:
        - n_bins (int): Número de conjuntos difusos por característica.
        """
        self.n_bins = n_bins
        self.rules_ = []
        self.classes_ = None
        self.n_features_ = None
        self.bin_edges_ = None
        self._estimator_type = "classifier"  # Definir explícitamente como clasificador

    def get_fuzzy_parameters(self, edges, j):
        """
        Obtiene los parámetros a, b, c para un conjunto difuso.

        Parámetros:
        - edges: Bordes de los conjuntos difusos
        - j: Índice del conjunto difuso

        Retorna:
        - tuple: (a, b, c) que define el conjunto difuso
        """
        if j == 0:
            a, b, c = edges[0], edges[0], edges[1]
        elif j == self.n_bins - 1:
            a, b, c = edges[-2], edges[-1], edges[-1]
        else:
            a, b, c = edges[j - 1], edges[j], edges[j + 1]

        return a, b, c

    def _fuzzify(self, x, bin_edges):
        """
        Calcula el grado de pertenencia de los valores a los conjuntos difusos.

        Parámetros:
        - x (array): Valores a fuzzificar
        - bin_edges (array): Bordes de los conjuntos difusos

        Retorna:
        - array: Matriz de pertenencia a cada conjunto difuso
        """
        fuzzified = np.zeros((len(x), self.n_bins))
        for i, xi in enumerate(x):
            edges = bin_edges[i]
            for j in range(self.n_bins):
                a, b, c = self.get_fuzzy_parameters(edges, j)
                # Usar el método calculate_membership
                fuzzified[i, j] = self.calculate_membership(xi, a, b, c)

        return fuzzified

    def _calculate_rule_votes(self, fuzzified):
        """
        Calcula los votos para cada clase basados en la activación de reglas difusas.

        Parámetros:
        - fuzzified (array): Valores difusos de la instancia

        Retorna:
        - dict: Votos para cada clase
        """
        votes = {}
        for rule in self.rules_:
            match_strengths = [fuzzified[i, label] for i, label in enumerate(rule['labels'])]
            # Usar la operación t-norma mínima para calcular la activación de la regla
            activation = np.min(match_strengths)
            votes[rule['class']] = votes.get(rule['class'], 0) + activation

        return votes

    def fit(self, X, y):
        """
        Entrena el clasificador difuso.

        Parámetros:
        - X (array): Datos de entrenamiento
        - y (array): Etiquetas

        Retorna:
        - self: Instancia entrenada
        """
        X, y = check_X_y(X, y)
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]

        # Evitar el uso de KBinsDiscretizer que puede dar problemas con datos reales
        self.bin_edges_ = []
        for i in range(self.n_features_):
            col = X[:, i]
            # Usar percentiles en lugar de espaciado uniforme
            edges = np.percentile(col, np.linspace(0, 100, self.n_bins))
            self.bin_edges_.append(edges)

        # Generar reglas a partir de los datos de entrenamiento
        self.rules_ = []
        for xi, yi in zip(X, y):
            fuzzified = self._fuzzify(xi, self.bin_edges_)
            labels = np.argmax(fuzzified, axis=1)
            rule = {'labels': labels, 'class': yi}
            self.rules_.append(rule)

        return self

    def predict(self, X):
        """
        Predice la clase más probable para cada muestra basada en coincidencias lingüísticas difusas.

        Retorna:
        - array: Predicciones por instancia.
        """
        X = check_array(X)
        X = np.asarray(X)
        preds = []
        for xi in X:
            fuzzified = self._fuzzify(xi, self.bin_edges_)
            pred = self._predict_instance(fuzzified)
            preds.append(pred)
        return np.array(preds)

    def _predict_instance(self, fuzzified):
        """
        Predice la clase para una instancia basada en activación de reglas difusas.

        Parámetros:
        - fuzzified (array): Valores difusos de la instancia

        Retorna:
        - class: Clase predicha
        """
        votes = self._calculate_rule_votes(fuzzified)

        if not votes:
            warnings.warn("No se activó ninguna regla. Se asigna clase por defecto.")
            return self.classes_[0]

        # Devolver la clase con mayor activación acumulada
        return max(votes.items(), key=lambda item: item[1])[0]

    def predict_proba(self, X):
        """
        Calcula probabilidades basadas en el grado de activación de reglas difusas.

        Parámetros:
        - X (array): Datos para predecir

        Retorna:
        - array: Matriz de probabilidades [n_samples, n_classes]
        """
        X = check_array(X)
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))

        for i, xi in enumerate(X):
            fuzzified = self._fuzzify(xi, self.bin_edges_)

            # Calcular votos para cada clase usando método compartido
            votes = self._calculate_rule_votes(fuzzified)

            # Convertir a array de probabilidades
            if votes:
                total = sum(votes.values())
                if total > 0:
                    for cls_idx, cls in enumerate(self.classes_):
                        proba[i, cls_idx] = votes.get(cls, 0) / total
                else:
                    # Si activación total es 0, asignar probabilidad a clase por defecto
                    default_idx = 0
                    proba[i, default_idx] = 1.0
            else:
                # Si no hay votos, asignar probabilidad a clase por defecto
                default_idx = 0
                proba[i, default_idx] = 1.0

        return proba

    def calculate_membership(self, xi, a, b, c):
        """
        Calcula el grado de pertenencia de un valor a un conjunto difuso.

        Parámetros:
        - xi: Valor a evaluar
        - a, b, c: Parámetros que definen el conjunto difuso

        Retorna:
        - float: Grado de pertenencia (entre 0 y 1)
        """
        if xi <= a or xi >= c:
            mu = 0
        elif xi == b:
            mu = 1
        elif xi < b:
            mu = (xi - a) / (b - a) if b != a else 0
        else:
            mu = (c - xi) / (c - b) if c != b else 0
        return mu