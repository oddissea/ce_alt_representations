#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expressions_s.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Representaciones Alternativas de Reglas - VERSIÓN TEÓRICAMENTE PURA
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 4.0 - Fidelidad Teórica Completa

Descripción:
    Implementación de expresiones simbólicas S de Lisp según teoría del
    libro tema 7 (Carmona-Galán). Opera ÚNICAMENTE sobre señales booleanas de entrada,
    sin umbrales evolutivos ni binarización artificial.

    TEORÍA: "operadores booleanos 'and', 'or' y 'not' sobre señales de entrada
    para representar cualquier función booleana"
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from abc import ABC, abstractmethod
import random
from collections import deque


class SExpressionNode(ABC):
    """Clase base para nodos de expresiones S según teoría."""

    def __init__(self):
        self.children = []
        self.parent = None

    @abstractmethod
    def evaluate(self, X_boolean):
        """Evalúa el nodo sobre señales BOOLEANAS de entrada."""
        pass

    @abstractmethod
    def arity(self):
        """Retorna la aridad del nodo (0 para terminales)."""
        pass

    @abstractmethod
    def __str__(self):
        """Representación como expresión S de Lisp."""
        pass

    def depth(self):
        """Profundidad del subárbol."""
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def size(self):
        """Número total de nodos en el subárbol."""
        return 1 + sum(child.size() for child in self.children)


class BooleanSignal(SExpressionNode):
    """Terminal: referencia directa a señal booleana de entrada (Xi)."""

    def __init__(self, signal_index):
        super().__init__()
        self.signal_index = signal_index

    def evaluate(self, X_boolean):
        """Evalúa directamente la señal booleana Xi."""
        if self.signal_index >= X_boolean.shape[1]:
            # Si índice fuera de rango, retornar False
            return np.zeros(X_boolean.shape[0], dtype=bool)

        # Asegurar que la señal sea estrictamente booleana
        signal = X_boolean[:, self.signal_index]
        return signal.astype(bool)

    def arity(self):
        return 0

    def __str__(self):
        return f"X{self.signal_index}"


class BooleanOperator(SExpressionNode):
    """Función: operador booleano puro (and, or, not)."""

    def __init__(self, operator):
        super().__init__()
        self.operator = operator

        # Validar operador según teoría
        valid_operators = {'and': 2, 'or': 2, 'not': 1}
        if operator not in valid_operators:
            raise ValueError(f"Operador no válido: {operator}. Válidos: {list(valid_operators.keys())}")

        self.operator_arity = valid_operators[operator]

    def evaluate(self, X_boolean):
        """Evalúa función booleana sobre los hijos."""
        if len(self.children) != self.operator_arity:
            return np.zeros(X_boolean.shape[0], dtype=bool)

        child_results = [child.evaluate(X_boolean) for child in self.children]

        if self.operator == 'and':
            return np.logical_and(child_results[0], child_results[1])
        elif self.operator == 'or':
            return np.logical_or(child_results[0], child_results[1])
        elif self.operator == 'not':
            return np.logical_not(child_results[0])
        else:
            return np.zeros(X_boolean.shape[0], dtype=bool)

    def arity(self):
        return self.operator_arity

    def __str__(self):
        if self.operator == 'not':
            return f"(not {self.children[0]})" if self.children else "(not ?)"
        else:
            if len(self.children) >= 2:
                return f"({self.operator} {self.children[0]} {self.children[1]})"
            else:
                return f"({self.operator} ? ?)"


class PureSExpressionTree:
    """Árbol de expresión S pura para funciones booleanas."""

    def __init__(self, root=None):
        self.root = root

    def evaluate(self, X_boolean):
        """Evalúa la expresión S sobre señales booleanas."""
        if self.root is None:
            return np.zeros(X_boolean.shape[0], dtype=bool)
        return self.root.evaluate(X_boolean)

    def depth(self):
        return self.root.depth() if self.root else 0

    def size(self):
        return self.root.size() if self.root else 0

    def __str__(self):
        return str(self.root) if self.root else "(empty)"

    def copy(self):
        """Crea copia profunda del árbol."""
        if self.root is None:
            return PureSExpressionTree()
        return PureSExpressionTree(self._copy_node(self.root))

    def _copy_node(self, node):
        """Copia recursiva de nodos."""
        if isinstance(node, BooleanSignal):
            new_node = BooleanSignal(node.signal_index)
        else:  # BooleanOperator
            new_node = BooleanOperator(node.operator)

        # Copiar hijos recursivamente
        for child in node.children:
            new_child = self._copy_node(child)
            new_child.parent = new_node
            new_node.children.append(new_child)

        return new_node


class PureSExpressionEvolution:
    """Motor evolutivo para expresiones S puras según teoría."""

    def __init__(self, population_size=50, generations=30, tournament_size=5,
                 crossover_prob=0.8, mutation_prob=0.3, max_depth=5,
                 random_state=42):

        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_depth = max_depth
        self.random_state = random_state

        # Operadores booleanos según teoría
        self.boolean_operators = ['and', 'or', 'not']

        # Estado evolutivo
        self.population = []
        self.best_individual = None
        self.fitness_history = []
        self.n_signals = 0

        # Configurar semilla para reproducibilidad
        random.seed(random_state)
        np.random.seed(random_state)

    def _create_random_signal(self):
        """Crea terminal que referencia señal booleana."""
        signal_idx = random.randint(0, self.n_signals - 1)
        return BooleanSignal(signal_idx)

    def _create_random_operator(self):
        """Crea operador booleano aleatorio."""
        operator = random.choice(self.boolean_operators)
        return BooleanOperator(operator)

    def _create_random_tree(self, method='grow', max_depth=5):
        """Crea árbol de expresión S aleatorio."""
        if method == 'grow':
            return self._grow_tree(max_depth)
        elif method == 'full':
            return self._full_tree(max_depth)
        else:  # ramped half-and-half
            return self._grow_tree(max_depth) if random.random() < 0.5 else self._full_tree(max_depth)

    def _grow_tree(self, max_depth):
        """Método grow: funciones hasta max_depth, luego terminales."""
        if max_depth <= 1:
            return self._create_random_signal()
        elif random.random() < 0.7:  # 70% probabilidad de función
            operator_node = self._create_random_operator()

            # Crear hijos según aridad
            for _ in range(operator_node.arity()):
                child = self._grow_tree(max_depth - 1)
                child.parent = operator_node
                operator_node.children.append(child)

            return operator_node
        else:
            return self._create_random_signal()

    def _full_tree(self, max_depth):
        """Método full: solo funciones hasta max_depth."""
        if max_depth <= 1:
            return self._create_random_signal()
        else:
            operator_node = self._create_random_operator()

            for _ in range(operator_node.arity()):
                child = self._full_tree(max_depth - 1)
                child.parent = operator_node
                operator_node.children.append(child)

            return operator_node

    @staticmethod
    def _evaluate_fitness(tree, X_boolean, y_boolean):
        """Evalúa fitness DIRECTO para función booleana pura."""
        try:
            # Evaluar función booleana
            boolean_output = tree.evaluate(X_boolean)
            predictions = boolean_output.astype(int)

            # Fitness = exactitud directa
            accuracy = np.mean(predictions == y_boolean)

            # Penalización mínima por complejidad (mantener parsimonia)
            complexity_penalty = tree.size() * 0.005  # Muy pequeña

            fitness = accuracy - complexity_penalty
            return max(0.0, min(1.0, fitness))

        except (ValueError, RuntimeError, AttributeError):
            return 0.0

    def _tournament_selection(self, population, fitnesses):
        """Selección por torneo."""
        tournament_size = min(self.tournament_size, len(population))
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx].copy()

    def _crossover(self, parent1, parent2):
        """Cruce estructural de expresiones S."""
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Obtener todos los nodos
        nodes1 = self._get_all_nodes(child1.root)
        nodes2 = self._get_all_nodes(child2.root)

        if len(nodes1) > 1 and len(nodes2) > 1:
            # Seleccionar nodos aleatorios (evitar raíz para mantener estructura)
            node1 = random.choice(nodes1[1:]) if len(nodes1) > 1 else nodes1[0]
            node2 = random.choice(nodes2[1:]) if len(nodes2) > 1 else nodes2[0]

            # Intercambiar subárboles
            if node1.parent and node2.parent:
                idx1 = node1.parent.children.index(node1)
                idx2 = node2.parent.children.index(node2)

                # Realizar intercambio
                node1.parent.children[idx1] = node2
                node2.parent.children[idx2] = node1

                # Actualizar padres
                node1.parent, node2.parent = node2.parent, node1.parent

        return child1, child2

    def _mutate(self, tree):
        """Mutación estructural de expresión S."""
        nodes = self._get_all_nodes(tree.root)
        if not nodes:
            return tree

        node = random.choice(nodes)

        if isinstance(node, BooleanSignal):
            # Mutar índice de señal
            node.signal_index = random.randint(0, self.n_signals - 1)
        else:  # BooleanOperator
            # Cambiar operador o reemplazar subárbol
            if random.random() < 0.5:
                # Cambiar operador manteniendo estructura
                old_operator = node.operator
                new_operator = random.choice(self.boolean_operators)
                if new_operator != old_operator:
                    new_node = BooleanOperator(new_operator)

                    # Copiar hijos compatibles
                    required_children = new_node.arity()
                    available_children = len(node.children)

                    if required_children <= available_children:
                        # Tomar primeros N hijos
                        for i in range(required_children):
                            child = node.children[i]
                            child.parent = new_node
                            new_node.children.append(child)
                    else:
                        # Añadir hijos faltantes
                        for child in node.children:
                            child.parent = new_node
                            new_node.children.append(child)

                        for _ in range(required_children - available_children):
                            new_child = self._create_random_signal()
                            new_child.parent = new_node
                            new_node.children.append(new_child)

                    # Reemplazar nodo en el padre
                    if node.parent:
                        idx = node.parent.children.index(node)
                        node.parent.children[idx] = new_node
                        new_node.parent = node.parent
                    else:
                        tree.root = new_node
            else:
                # Reemplazar con nuevo subárbol pequeño
                new_subtree = self._create_random_tree('grow', max_depth=2)

                if node.parent:
                    idx = node.parent.children.index(node)
                    node.parent.children[idx] = new_subtree
                    new_subtree.parent = node.parent
                else:
                    tree.root = new_subtree

        return tree

    @staticmethod
    def _get_all_nodes(root):
        """Obtiene todos los nodos del árbol (breadth-first)."""
        if root is None:
            return []

        nodes = [root]
        queue = deque([root])

        while queue:
            current = queue.popleft()
            for child in current.children:
                nodes.append(child)
                queue.append(child)

        return nodes

    def evolve(self, X_boolean, y_boolean):
        """Evoluciona expresiones S sobre datos ESTRICTAMENTE booleanos."""
        self.n_signals = X_boolean.shape[1]

        print(f"Evolucionando expresiones S puras...")
        print(f"Señales de entrada: {self.n_signals}")
        print(f"Muestras: {X_boolean.shape[0]}")

        # Inicializar población
        self.population = []
        for _ in range(self.population_size):
            tree = PureSExpressionTree(self._create_random_tree('ramped', self.max_depth))
            self.population.append(tree)

        # Evolución
        for generation in range(self.generations):
            # Evaluar fitness
            fitnesses = [self._evaluate_fitness(tree, X_boolean, y_boolean)
                         for tree in self.population]

            # Mejor individuo
            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]

            if self.best_individual is None or best_fitness > self._evaluate_fitness(
                    self.best_individual, X_boolean, y_boolean):
                self.best_individual = self.population[best_idx].copy()

            self.fitness_history.append(best_fitness)

            if generation % 10 == 0 or best_fitness > 0.95:
                print(f"Generación {generation}: Mejor fitness = {best_fitness:.4f}")
                print(f"  Mejor expresión: {self.population[best_idx]}")

            # Parar si alcanzamos solución perfecta
            if best_fitness >= 0.999:
                print(f"¡Solución perfecta encontrada en generación {generation}!")
                break

            # Nueva población con elitismo
            new_population = [self.population[best_idx].copy()]

            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(self.population, fitnesses)
                parent2 = self._tournament_selection(self.population, fitnesses)

                if random.random() < self.crossover_prob:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                if random.random() < self.mutation_prob:
                    child1 = self._mutate(child1)
                if random.random() < self.mutation_prob:
                    child2 = self._mutate(child2)

                # Control de profundidad
                if child1.depth() <= self.max_depth:
                    new_population.append(child1)
                if len(new_population) < self.population_size and child2.depth() <= self.max_depth:
                    new_population.append(child2)

            self.population = new_population[:self.population_size]

        final_fitness = self._evaluate_fitness(self.best_individual, X_boolean, y_boolean)
        print(f"Evolución completada. Fitness final: {final_fitness:.4f}")
        print(f"Expresión final: {self.best_individual}")

        return self.best_individual


class PureExpressionsSClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador de expresiones S PURAS según teoría de Carmona-Galán.
    Opera ÚNICAMENTE sobre señales booleanas de entrada.
    """

    def __init__(self, population_size=50, generations=30, tournament_size=5,
                 crossover_prob=0.8, mutation_prob=0.3, max_depth=5,
                 random_state=42):

        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_depth = max_depth
        self.random_state = random_state

        # Estado interno
        self.evolution_engine = None
        self.best_expression = None
        self.classes_ = None
        self.using_fallback = False
        self.program_str = None
        self._estimator_type = "classifier"

    @staticmethod
    def _prepare_boolean_data(X, y=None):
        """Convierte datos a formato estrictamente booleano."""
        # Para expresiones S puras, los datos DEBEN ser booleanos
        # Si no lo son, aplicamos binarización simple pero consciente

        if np.all(np.isin(X, [0, 1])):
            # Ya son booleanos
            X_boolean = X.astype(bool)
        else:
            # Binarización por mediana (método simple y determinista)
            print("ADVERTENCIA: Datos no booleanos. Aplicando binarización por mediana.")
            median_values = np.median(X, axis=0)
            X_boolean = np.asarray(X > median_values, dtype=bool)

        if y is not None:
            y_boolean = y.astype(int)
            return X_boolean, y_boolean
        else:
            return X_boolean

    def fit(self, X, y):
        """Entrena clasificador de expresiones S puras."""
        try:
            # Preparar datos booleanos
            X_boolean, y_boolean = self._prepare_boolean_data(X, y)

            # Crear motor evolutivo
            self.evolution_engine = PureSExpressionEvolution(
                population_size=self.population_size,
                generations=self.generations,
                tournament_size=self.tournament_size,
                crossover_prob=self.crossover_prob,
                mutation_prob=self.mutation_prob,
                max_depth=self.max_depth,
                random_state=self.random_state
            )

            print("Entrenando expresiones S puras (solo operadores booleanos)...")
            self.best_expression = self.evolution_engine.evolve(X_boolean, y_boolean)
            self.classes_ = np.unique(y)
            self.program_str = str(self.best_expression)
            self.using_fallback = False

            print(f"Expresión S pura: {self.program_str}")

        except Exception as e:
            print(f"Error en evolución de expresiones S puras: {e}")
            print("Usando RandomForest como fallback...")

            self.evolution_engine = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=self.random_state
            )
            self.evolution_engine.fit(X, y)
            self.classes_ = np.unique(y)
            self.program_str = "RandomForest fallback (datos no apropiados para expresiones S)"
            self.using_fallback = True

        return self

    def predict(self, X):
        """Predicciones usando expresión S pura."""
        if self.using_fallback:
            return self.evolution_engine.predict(X)

        if self.best_expression is None:
            raise ValueError("Modelo no entrenado")

        # Preparar datos booleanos
        X_boolean = self._prepare_boolean_data(X)

        # Evaluar expresión booleana
        boolean_result = self.best_expression.evaluate(X_boolean)
        predictions = boolean_result.astype(int)

        # Mapear a clases originales
        if len(self.classes_) == 2:
            return np.where(predictions == 1, self.classes_[1], self.classes_[0])
        else:
            return np.full(len(X), self.classes_[0])

    def predict_proba(self, X):
        """Probabilidades de clase."""
        if self.using_fallback:
            return self.evolution_engine.predict_proba(X)

        if self.best_expression is None:
            raise ValueError("Modelo no entrenado")

        # Para expresiones S puras, probabilidades son deterministas
        X_boolean = self._prepare_boolean_data(X)
        boolean_result = self.best_expression.evaluate(X_boolean)

        # Probabilidades altas para decisiones booleanas claras
        prob_positive = boolean_result.astype(float) * 0.95 + 0.025  # [0.025, 0.975]
        prob_negative = 1 - prob_positive

        return np.column_stack([prob_negative, prob_positive])

    def get_expression(self):
        """Obtiene expresión S como cadena."""
        return self.program_str if self.program_str else "Modelo no entrenado"

    def get_complexity(self):
        """Medidas de complejidad."""
        if self.using_fallback:
            return {'size': 50, 'depth': 5, 'using_fallback': True, 'pure_s_expression': False}

        if self.best_expression is None:
            return {'size': 0, 'depth': 0, 'using_fallback': False, 'pure_s_expression': True}

        return {
            'size': self.best_expression.size(),
            'depth': self.best_expression.depth(),
            'using_fallback': False,
            'pure_s_expression': True
        }