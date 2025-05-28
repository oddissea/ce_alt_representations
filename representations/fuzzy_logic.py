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
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_X_y


class ImprovedFuzzyLogicClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador de lógica difusa mejorado con clustering adaptativo.
    """

    def __init__(self, n_fuzzy_sets=5, membership_type='gaussian',
                 defuzzification='centroid', adaptive_clustering=True):
        self.n_fuzzy_sets = n_fuzzy_sets
        self.membership_type = membership_type  # 'gaussian', 'triangular', 'trapezoidal'
        self.defuzzification = defuzzification  # 'centroid', 'weighted_average', 'max'
        self.adaptive_clustering = adaptive_clustering
        self.fuzzy_sets_ = {}
        self.rules_ = []
        self.classes_ = None
        self.n_features_ = None
        self._estimator_type = "classifier"

    def _create_uniform_fuzzy_set(self, data_values):
        """
        Crea conjuntos difusos con distribución uniforme para un conjunto de datos.

        Parámetros:
        - data_values: Array de valores para crear los conjuntos difusos

        Retorna:
        - dict: Diccionario con 'centers' y 'sigmas'
        """
        min_val, max_val = np.min(data_values), np.max(data_values)

        # Evitar división por cero si todos los valores son iguales
        if min_val == max_val:
            centers = [min_val] * self.n_fuzzy_sets
            sigma = 0.1  # Valor pequeño por defecto
        else:
            centers = np.linspace(min_val, max_val, self.n_fuzzy_sets)
            sigma = (max_val - min_val) / (2 * self.n_fuzzy_sets)

        return {
            'centers': centers,
            'sigmas': [sigma] * self.n_fuzzy_sets
        }


    def _initialize_fuzzy_sets_with_clustering(self, X, y):
        """Inicializar conjuntos difusos usando clustering por clase."""
        self.fuzzy_sets_ = {}

        for feature_idx in range(self.n_features_):
            self.fuzzy_sets_[feature_idx] = {}

            for class_label in self.classes_:
                X_class_feature = X[y == class_label, feature_idx]

                if len(X_class_feature) >= self.n_fuzzy_sets:
                    # Usar clustering para encontrar centros naturales
                    kmeans = KMeans(n_clusters=self.n_fuzzy_sets, random_state=42, n_init=10)
                    clusters = kmeans.fit(X_class_feature.reshape(-1, 1))
                    centers = clusters.cluster_centers_.flatten()

                    # Calcular desviaciones estándar para cada cluster
                    labels = clusters.labels_
                    sigmas = []
                    for i in range(self.n_fuzzy_sets):
                        cluster_data = X_class_feature[labels == i]
                        if len(cluster_data) > 1:
                            sigma = np.std(cluster_data)
                        else:
                            sigma = 0.1  # Valor por defecto pequeño
                        sigmas.append(max(sigma, 0.05))  # Evitar sigmas muy pequeños

                    self.fuzzy_sets_[feature_idx][class_label] = {
                        'centers': sorted(centers),
                        'sigmas': sigmas
                    }
                else:
                    # Fallback: distribución uniforme
                    self.fuzzy_sets_[feature_idx][class_label] = self._create_uniform_fuzzy_set(X_class_feature)

    @staticmethod
    def _gaussian_membership(x, center, sigma):
        """Función de pertenencia gaussiana mejorada."""
        if sigma <= 0:
            sigma = 0.1
        return np.exp(-0.5 * ((x - center) / sigma) ** 2)

    @staticmethod
    def _triangular_membership(x, a, b, c):
        """Función de pertenencia triangular."""
        if x <= a or x >= c:
            return 0.0
        elif x == b:
            return 1.0
        elif x < b:
            return (x - a) / (b - a) if b != a else 0.0
        else:
            return (c - x) / (c - b) if c != b else 0.0

    def _fuzzify_input(self, x):
        """Convierte entrada nítida en valores difusos."""
        fuzzified = {}

        for feature_idx in range(len(x)):
            fuzzified[feature_idx] = {}

            for class_label in self.classes_:
                if feature_idx in self.fuzzy_sets_ and class_label in self.fuzzy_sets_[feature_idx]:
                    fuzzy_set = self.fuzzy_sets_[feature_idx][class_label]
                    memberships = []

                    for i, center in enumerate(fuzzy_set['centers']):
                        if self.membership_type == 'gaussian':
                            membership = self._gaussian_membership(
                                x[feature_idx], center, fuzzy_set['sigmas'][i]
                            )
                        elif self.membership_type == 'triangular':
                            # Para triangular, necesitamos a, b, c
                            if i == 0:
                                a = center - fuzzy_set['sigmas'][i]
                                b = center
                                c = center + fuzzy_set['sigmas'][i]
                            else:
                                prev_center = fuzzy_set['centers'][i - 1]
                                a = (prev_center + center) / 2
                                b = center
                                if i < len(fuzzy_set['centers']) - 1:
                                    next_center = fuzzy_set['centers'][i + 1]
                                    c = (center + next_center) / 2
                                else:
                                    c = center + fuzzy_set['sigmas'][i]

                            membership = self._triangular_membership(x[feature_idx], a, b, c)
                        else:
                            membership = self._gaussian_membership(
                                x[feature_idx], center, fuzzy_set['sigmas'][i]
                            )

                        memberships.append(membership)

                    fuzzified[feature_idx][class_label] = np.array(memberships)
                else:
                    fuzzified[feature_idx][class_label] = np.array([0.0] * self.n_fuzzy_sets)

        return fuzzified

    def _apply_fuzzy_rules(self, fuzzified_input):
        """Aplica reglas difusas usando inferencia Mamdani."""
        class_activations = {class_label: 0.0 for class_label in self.classes_}

        # Para cada clase, calcular la activación total
        for class_label in self.classes_:
            feature_activations = []

            for feature_idx in range(self.n_features_):
                if feature_idx in fuzzified_input and class_label in fuzzified_input[feature_idx]:
                    # Tomar el máximo grado de pertenencia para esta característica y clase
                    max_membership = np.max(fuzzified_input[feature_idx][class_label])
                    feature_activations.append(max_membership)

            # Combinar activaciones usando operador T-norma (mínimo)
            if feature_activations:
                class_activations[class_label] = float(np.mean(feature_activations))  # Promedio en lugar de mínimo para mayor robustez

        return class_activations

    def _defuzzify(self, class_activations):
        """Defuzzificación para obtener clase nítida."""
        if not class_activations or all(v == 0 for v in class_activations.values()):
            return self.classes_[0]  # Clase por defecto

        if self.defuzzification == 'max':
            return max(class_activations.items(), key=lambda x: x[1])[0]
        elif self.defuzzification == 'weighted_average':
            total_weight = sum(class_activations.values())
            if total_weight == 0:
                return self.classes_[0]

            weighted_sum = sum(class_idx * activation
                               for class_idx, activation in enumerate(class_activations.values()))
            avg_class_idx = int(weighted_sum / total_weight)
            return self.classes_[min(avg_class_idx, len(self.classes_) - 1)]
        else:  # centroid (por defecto)
            return max(class_activations.items(), key=lambda x: x[1])[0]

    def fit(self, X, y):
        """Entrena el clasificador difuso con clustering adaptativo."""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]

        if self.adaptive_clustering:
            self._initialize_fuzzy_sets_with_clustering(X, y)
        else:
            # Método original como fallback
            self._initialize_fuzzy_sets_uniform(X)

        return self

    def _initialize_fuzzy_sets_uniform(self, X):
        """Método de inicialización uniforme (fallback)."""
        self.fuzzy_sets_ = {}

        for feature_idx in range(self.n_features_):
            self.fuzzy_sets_[feature_idx] = {}
            col = X[:, feature_idx]

            for class_label in self.classes_:
                # Distribución uniforme simple
                self.fuzzy_sets_[feature_idx][class_label] = self._create_uniform_fuzzy_set(col)

    def predict(self, X):
        """Predice clases para las muestras."""
        X = np.array(X, dtype=np.float64)
        predictions = []

        for x in X:
            # Fuzzificar entrada
            fuzzified = self._fuzzify_input(x)

            # Aplicar reglas difusas
            activations = self._apply_fuzzy_rules(fuzzified)

            # Defuzzificar
            prediction = self._defuzzify(activations)
            predictions.append(prediction)

        return np.array(predictions)

    def predict_proba(self, X):
        """Predice probabilidades de clase."""
        X = np.array(X, dtype=np.float64)
        probabilities = []

        for x in X:
            # Fuzzificar entrada
            fuzzified = self._fuzzify_input(x)

            # Aplicar reglas difusas
            activations = self._apply_fuzzy_rules(fuzzified)

            # Normalizar activaciones a probabilidades
            total_activation = sum(activations.values())
            if total_activation > 0:
                proba = [activations.get(class_label, 0) / total_activation
                         for class_label in self.classes_]
            else:
                # Distribución uniforme si no hay activación
                proba = [1.0 / len(self.classes_)] * len(self.classes_)

            probabilities.append(proba)

        return np.array(probabilities)

    def get_interpretability_score(self):
        """Retorna puntuación de interpretabilidad."""
        # Número total de conjuntos difusos como medida de complejidad
        total_sets = self.n_fuzzy_sets * self.n_features_ * len(self.classes_)
        # Invertir para que menos conjuntos = mayor interpretabilidad
        return 1.0 / (1.0 + total_sets / 100.0)

    def get_complexity(self):
        """Retorna medida de complejidad del modelo."""
        return self.n_fuzzy_sets * self.n_features_ * len(self.classes_)


# Alias para compatibilidad
FuzzyLogicClassifier = ImprovedFuzzyLogicClassifier