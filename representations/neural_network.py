#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
neural_network.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Representación por redes neuronales CORREGIDA según memoria.pdf
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 2.0 - Fidelidad teórica completa

Descripción:
    Sistema clasificador neuronal según especificaciones exactas de memoria.pdf sección 2.4.
    Implementa múltiples pequeñas redes neuronales cooperativas tipo X-NCS de Bull y O'Hara.
    Cada regla contiene una red neuronal que actúa como condición con nodo de activación.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
import random
import warnings

warnings.filterwarnings('ignore')


def sigmoid(x):
    """Función sigmoide estable numéricamente."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def tanh_activation(x):
    """Función tanh para activación."""
    return np.tanh(np.clip(x, -500, 500))


class SmallNeuralRule:
    """
    Regla individual como pequeña red neuronal según paradigma X-NCS.
    Cada regla = una red neuronal pequeña con nodo de activación.
    """

    def __init__(self, input_size, hidden_size, n_actions, rule_id, random_state=42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self.rule_id = rule_id
        self.random_state = random_state

        # Configurar semilla específica para esta regla
        np.random.seed(random_state + rule_id)

        # Pesos de la red (genotipo evolutivo)
        # Inicialización Xavier/Glorot
        limit_ih = np.sqrt(6.0 / (input_size + hidden_size))
        self.weights_input_hidden = np.random.uniform(
            -limit_ih, limit_ih, (input_size, hidden_size)
        )

        limit_ho = np.sqrt(6.0 / (hidden_size + n_actions + 1))
        self.weights_hidden_output = np.random.uniform(
            -limit_ho, limit_ho, (hidden_size, n_actions + 1)
        )
        # +1 para el nodo de activación según especificación teórica

        # Sesgos (bias)
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(n_actions + 1)

        # Historial de rendimiento para evolución
        self.fitness = 0.0
        self.activation_count = 0
        self.correct_predictions = 0

    def evaluate(self, x):
        """
        Propagación hacia adelante según memoria.pdf.
        Retorna: (activación_regla, voto_acción, confianza)
        """
        try:
            # Asegurar formato correcto
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 0:
                x = np.array([x])

            # Capa oculta con activación tanh
            hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
            hidden = tanh_activation(hidden_input)

            # Capa de salida (acciones + nodo activación)
            output_input = np.dot(hidden, self.weights_hidden_output) + self.bias_output
            output = sigmoid(output_input)

            # Nodo adicional determina activación de la regla (último nodo)
            activation_score = output[-1]

            # Acciones disponibles (todos menos el nodo activación)
            action_scores = output[:-1]
            predicted_action = np.argmax(action_scores)
            confidence = np.max(action_scores)

            return activation_score, predicted_action, confidence

        except Exception as e:
            # Fallback seguro
            return 0.0, 0, 0.0

    def update_fitness(self, predicted_correctly, was_activated):
        """Actualiza fitness de la regla según rendimiento."""
        if was_activated:
            self.activation_count += 1
            if predicted_correctly:
                self.correct_predictions += 1

        # Fitness = precisión cuando se activa
        if self.activation_count > 0:
            self.fitness = self.correct_predictions / self.activation_count
        else:
            self.fitness = 0.0

    def mutate(self, mutation_rate=0.1, mutation_strength=0.1):
        """
        Mutación de pesos para evolución según paradigma evolutivo.
        """
        # Mutar pesos input->hidden
        mask = np.random.random(self.weights_input_hidden.shape) < mutation_rate
        noise = np.random.normal(0, mutation_strength, self.weights_input_hidden.shape)
        self.weights_input_hidden += mask * noise

        # Mutar pesos hidden->output
        mask = np.random.random(self.weights_hidden_output.shape) < mutation_rate
        noise = np.random.normal(0, mutation_strength, self.weights_hidden_output.shape)
        self.weights_hidden_output += mask * noise

        # Mutar sesgos ocasionalmente
        if np.random.random() < mutation_rate:
            self.bias_hidden += np.random.normal(0, mutation_strength / 2, self.bias_hidden.shape)
            self.bias_output += np.random.normal(0, mutation_strength / 2, self.bias_output.shape)

    def crossover(self, other_rule):
        """
        Cruce entre dos reglas para generar descendencia.
        """
        child = SmallNeuralRule(
            self.input_size, self.hidden_size, self.n_actions,
            self.rule_id, self.random_state
        )

        # Cruce uniforme de pesos
        mask = np.random.random(self.weights_input_hidden.shape) < 0.5
        child.weights_input_hidden = np.where(
            mask, self.weights_input_hidden, other_rule.weights_input_hidden
        )

        mask = np.random.random(self.weights_hidden_output.shape) < 0.5
        child.weights_hidden_output = np.where(
            mask, self.weights_hidden_output, other_rule.weights_hidden_output
        )

        # Cruce de sesgos
        mask = np.random.random(self.bias_hidden.shape) < 0.5
        child.bias_hidden = np.where(mask, self.bias_hidden, other_rule.bias_hidden)

        mask = np.random.random(self.bias_output.shape) < 0.5
        child.bias_output = np.where(mask, self.bias_output, other_rule.bias_output)

        return child

    def copy(self):
        """Crea copia exacta de la regla."""
        copy_rule = SmallNeuralRule(
            self.input_size, self.hidden_size, self.n_actions,
            self.rule_id, self.random_state
        )

        copy_rule.weights_input_hidden = self.weights_input_hidden.copy()
        copy_rule.weights_hidden_output = self.weights_hidden_output.copy()
        copy_rule.bias_hidden = self.bias_hidden.copy()
        copy_rule.bias_output = self.bias_output.copy()
        copy_rule.fitness = self.fitness

        return copy_rule


class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    """
    Sistema clasificador neuronal según especificaciones exactas de memoria.pdf sección 2.4.
    Implementa múltiples pequeñas redes neuronales cooperativas tipo X-NCS.
    """

    def __init__(self, n_rules=15, hidden_layer_sizes=(8,), max_iter=100,
                 activation_threshold=0.5, mutation_rate=0.15,
                 crossover_rate=0.6, random_state=42):
        """
        Parámetros ajustados para paradigma X-NCS:
        - n_rules: Número de reglas neuronales cooperativas
        - hidden_layer_sizes: Arquitectura compartida (pequeñas redes)
        - max_iter: Generaciones de evolución
        - activation_threshold: Umbral para activación de reglas
        """
        self.n_rules = n_rules
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.activation_threshold = activation_threshold
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.random_state = random_state

        # Estado interno
        self.neural_rules = []
        self.classes_ = None
        self.n_features_ = None
        self.using_fallback = False
        self._estimator_type = "classifier"

        # Configurar semilla
        np.random.seed(random_state)
        random.seed(random_state)

    def fit(self, X, y):
        """
        Entrena el sistema mediante evolución de múltiples redes cooperativas.
        """
        try:
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y)

            self.classes_ = np.unique(y)
            self.n_features_ = X.shape[1]

            print(f"Entrenando sistema neuronal cooperativo:")
            print(f"  {self.n_rules} reglas neuronales")
            print(
                f"  Arquitectura por regla: {self.n_features_} → {self.hidden_layer_sizes[0]} → {len(self.classes_) + 1}")
            print(f"  Evolución: {self.max_iter} generaciones")

            # Verificar problema binario
            if len(self.classes_) != 2:
                raise ValueError("Sistema neuronal requiere clasificación binaria")

            # Mapear clases a índices
            y_mapped = np.array([np.where(self.classes_ == label)[0][0] for label in y])

            # Crear población inicial de reglas neuronales
            self._initialize_neural_rules()

            # Evolución de la población
            self._evolve_neural_rules(X, y_mapped)

            print(f"Sistema entrenado. Reglas activas promedio: {self._count_active_rules(X):.1f}")
            self.using_fallback = False

        except Exception as e:
            print(f"Error en entrenamiento neuronal cooperativo: {e}")
            print("Activando modo fallback con MLPClassifier...")

            # Fallback a implementación estándar
            self._activate_fallback_mode(X, y)

        return self

    def _initialize_neural_rules(self):
        """Inicializa población de reglas neuronales."""
        self.neural_rules = []

        for rule_id in range(self.n_rules):
            rule = SmallNeuralRule(
                input_size=self.n_features_,
                hidden_size=self.hidden_layer_sizes[0],
                n_actions=len(self.classes_),
                rule_id=rule_id,
                random_state=self.random_state + rule_id
            )
            self.neural_rules.append(rule)

    def _evolve_neural_rules(self, X, y):
        """
        Evolución de reglas neuronales según algoritmo evolutivo.
        """
        for generation in range(self.max_iter):
            # Evaluar fitness de todas las reglas
            self._evaluate_population_fitness(X, y)

            # Reportar progreso
            if generation % 20 == 0:
                avg_fitness = np.mean([rule.fitness for rule in self.neural_rules])
                active_rules = self._count_active_rules(X)
                print(f"  Gen {generation}: fitness promedio = {avg_fitness:.3f}, reglas activas = {active_rules:.1f}")

            # Crear nueva generación
            new_population = self._create_new_generation()
            self.neural_rules = new_population

        # Evaluación final
        self._evaluate_population_fitness(X, y)

    def _evaluate_population_fitness(self, X, y):
        """Evalúa fitness de cada regla en la población."""
        # Resetear contadores
        for rule in self.neural_rules:
            rule.activation_count = 0
            rule.correct_predictions = 0

        # Evaluar en cada muestra
        for i, (x, true_label) in enumerate(zip(X, y)):
            for rule in self.neural_rules:
                activation, predicted_action, confidence = rule.evaluate(x)

                if activation > self.activation_threshold:
                    predicted_correctly = (predicted_action == true_label)
                    rule.update_fitness(predicted_correctly, True)

    def _count_active_rules(self, X):
        """Cuenta reglas activas promedio."""
        total_activations = 0

        for x in X:
            activations = 0
            for rule in self.neural_rules:
                activation, _, _ = rule.evaluate(x)
                if activation > self.activation_threshold:
                    activations += 1
            total_activations += activations

        return total_activations / len(X) if len(X) > 0 else 0

    def _create_new_generation(self):
        """Crea nueva generación mediante selección, cruce y mutación."""
        # Ordenar por fitness
        sorted_rules = sorted(self.neural_rules, key=lambda r: r.fitness, reverse=True)

        new_population = []

        # Elitismo: mantener mejores reglas
        elite_size = max(2, self.n_rules // 5)
        for i in range(elite_size):
            new_population.append(sorted_rules[i].copy())

        # Generar resto mediante cruce y mutación
        while len(new_population) < self.n_rules:
            # Selección por torneo
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            if np.random.random() < self.crossover_rate:
                child = parent1.crossover(parent2)
            else:
                child = parent1.copy()

            # Mutación
            if np.random.random() < self.mutation_rate:
                child.mutate(self.mutation_rate, mutation_strength=0.1)

            new_population.append(child)

        return new_population[:self.n_rules]

    def _tournament_selection(self, tournament_size=3):
        """Selección por torneo."""
        tournament = random.sample(self.neural_rules,
                                   min(tournament_size, len(self.neural_rules)))
        return max(tournament, key=lambda r: r.fitness).copy()

    def _activate_fallback_mode(self, X, y):
        """Activa modo fallback con MLPClassifier estándar."""
        self.fallback_model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=200,
            random_state=self.random_state,
            activation='relu',
            solver='adam'
        )
        self.fallback_model.fit(X, y)
        self.using_fallback = True

    def predict(self, X):
        """
        Predicciones mediante cooperación de reglas neuronales activas.
        """
        if self.using_fallback:
            return self.fallback_model.predict(X)

        X = np.asarray(X, dtype=np.float64)
        predictions = []

        for x in X:
            prediction = self._predict_single_cooperative(x)
            predictions.append(prediction)

        return np.array(predictions)

    def _predict_single_cooperative(self, x):
        """
        Predicción individual mediante cooperación de reglas activas.
        Implementa el paradigma cooperativo de memoria.pdf.
        """
        active_votes = []
        active_confidences = []

        # Evaluar todas las reglas
        for rule in self.neural_rules:
            activation, action_vote, confidence = rule.evaluate(x)

            if activation > self.activation_threshold:
                active_votes.append(action_vote)
                active_confidences.append(confidence * activation)  # Ponderar por activación

        # Cooperación: combinar votos de reglas activas
        if active_votes:
            # Voto ponderado por confianza
            if active_confidences:
                vote_counts = {}
                for vote, conf in zip(active_votes, active_confidences):
                    vote_counts[vote] = vote_counts.get(vote, 0) + conf

                predicted_class_idx = max(vote_counts.items(), key=lambda x: x[1])[0]
            else:
                # Voto mayoritario simple
                predicted_class_idx = max(set(active_votes), key=active_votes.count)
        else:
            # Ninguna regla activa: clase más probable por defecto
            predicted_class_idx = 0

        # Mapear índice a clase original
        return self.classes_[min(predicted_class_idx, len(self.classes_) - 1)]

    def predict_proba(self, X):
        """Probabilidades mediante agregación de reglas activas."""
        if self.using_fallback:
            return self.fallback_model.predict_proba(X)

        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)
        n_classes = len(self.classes_)
        probabilities = np.zeros((n_samples, n_classes))

        for i, x in enumerate(X):
            class_scores = np.zeros(n_classes)
            total_activation = 0

            # Agregar scores de reglas activas
            for rule in self.neural_rules:
                activation, action_vote, confidence = rule.evaluate(x)

                if activation > self.activation_threshold:
                    if action_vote < n_classes:
                        class_scores[action_vote] += confidence * activation
                        total_activation += activation

            # Normalizar a probabilidades
            if total_activation > 0:
                probabilities[i] = class_scores / np.sum(class_scores) if np.sum(class_scores) > 0 else np.ones(
                    n_classes) / n_classes
            else:
                # Distribución uniforme si no hay activación
                probabilities[i] = np.ones(n_classes) / n_classes

        return probabilities

    def get_network_structure(self):
        """Información sobre estructura del sistema neuronal."""
        if self.using_fallback:
            return {
                'type': 'MLPClassifier fallback',
                'hidden_layers': self.hidden_layer_sizes,
                'using_fallback': True
            }

        if not self.neural_rules:
            return {'type': 'No entrenado', 'n_rules': 0}

        # Estadísticas de reglas activas
        active_rules = len([r for r in self.neural_rules if r.fitness > 0])
        avg_fitness = np.mean([r.fitness for r in self.neural_rules])

        return {
            'type': 'Sistema neuronal cooperativo X-NCS',
            'n_rules': len(self.neural_rules),
            'active_rules': active_rules,
            'avg_fitness': avg_fitness,
            'architecture_per_rule': f"{self.n_features_} → {self.hidden_layer_sizes[0]} → {len(self.classes_) + 1}",
            'activation_threshold': self.activation_threshold,
            'using_fallback': False
        }

    # Métodos de compatibilidad sklearn
    def get_params(self, deep=True):
        """Parámetros del estimador."""
        return {
            'n_rules': self.n_rules,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'max_iter': self.max_iter,
            'activation_threshold': self.activation_threshold,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        """Establece parámetros del estimador."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        return self

    # Métodos para intelligent_fallback_evaluation
    def simplify_for_fallback(self):
        """Versión simplificada para fallback."""
        simplified = NeuralNetworkClassifier(
            n_rules=8,
            hidden_layer_sizes=(5,),
            max_iter=50,
            random_state=self.random_state
        )
        return simplified

    def increase_regularization(self):
        """Aumenta regularización."""
        self.mutation_rate = min(0.3, self.mutation_rate * 1.2)
        self.activation_threshold = min(0.7, self.activation_threshold + 0.1)

    def reduce_complexity(self):
        """Reduce complejidad del modelo."""
        self.n_rules = max(5, self.n_rules - 3)
        self.max_iter = min(50, self.max_iter)
        if self.hidden_layer_sizes[0] > 4:
            self.hidden_layer_sizes = (self.hidden_layer_sizes[0] - 2,)