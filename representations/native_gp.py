#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
native_gp.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Prueba de evaluación continua 4 - Representaciones Alternativas de Reglas
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 2.0 - Nivel 2 Improvements

Descripción:
    Implementación nativa de programación genética para sistemas clasificadores evolutivos.
    Proporciona control completo sobre el proceso evolutivo sin dependencias externas,
    específicamente optimizada para clasificación con control de bloat automático.
"""

import numpy as np
import random
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from collections import deque
#import math


class GPNode(ABC):
    """Clase base abstracta para nodos del árbol de programación genética."""

    def __init__(self):
        self.children = []
        self.parent = None

    @abstractmethod
    def evaluate(self, X):
        """Evalúa el nodo con los datos de entrada X."""
        pass

    @abstractmethod
    def arity(self):
        """Retorna la aridad (número de hijos) del nodo."""
        pass

    @abstractmethod
    def __str__(self):
        """Representación en cadena del nodo."""
        pass

    def depth(self):
        """Calcula la profundidad del subárbol desde este nodo."""
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def size(self):
        """Calcula el número total de nodos en el subárbol."""
        return 1 + sum(child.size() for child in self.children)


class TerminalNode(GPNode):
    """Nodo terminal: variable de entrada o constante."""

    def __init__(self, symbol, is_variable=True, value=None):
        super().__init__()
        self.symbol = symbol
        self.is_variable = is_variable
        self.value = value

    def evaluate(self, X):
        if self.is_variable:
            # X es una matriz, symbol es el índice de la característica
            feature_idx = int(self.symbol.replace('X', ''))
            if feature_idx < X.shape[1]:
                return X[:, feature_idx]
            else:
                # Si el índice está fuera de rango, retornar zeros
                return np.zeros(X.shape[0])
        else:
            # Constante: retornar array del mismo tamaño que el número de muestras
            return np.full(X.shape[0], self.value)

    def arity(self):
        return 0

    def __str__(self):
        if self.is_variable:
            return self.symbol
        else:
            return f"{self.value:.3f}"


class FunctionNode(GPNode):
    """Nodo función: operador matemático."""

    def __init__(self, function_name, function_impl, arity_val):
        super().__init__()
        self.function_name = function_name
        self.function_impl = function_impl
        self.arity_val = arity_val

    def evaluate(self, X):
        # Evaluar todos los hijos
        child_results = [child.evaluate(X) for child in self.children]

        # Aplicar la función
        try:
            return self.function_impl(*child_results)
        except (ValueError, ZeroDivisionError, OverflowError, RuntimeWarning):
            # En caso de error matemático, retornar zeros
            return np.zeros(X.shape[0])

    def arity(self):
        return self.arity_val

    def __str__(self):
        if len(self.children) == 2:
            return f"({self.children[0]} {self.function_name} {self.children[1]})"
        elif len(self.children) == 1:
            return f"{self.function_name}({self.children[0]})"
        else:
            args = ", ".join(str(child) for child in self.children)
            return f"{self.function_name}({args})"


class GPTree:
    """Árbol de programación genética."""

    def __init__(self, root=None):
        self.root = root

    def evaluate(self, X):
        if self.root is None:
            return np.zeros(X.shape[0])
        return self.root.evaluate(X)

    def depth(self):
        return self.root.depth() if self.root else 0

    def size(self):
        return self.root.size() if self.root else 0

    def __str__(self):
        return str(self.root) if self.root else "Empty"

    def copy(self):
        """Crea una copia profunda del árbol."""
        if self.root is None:
            return GPTree()
        return GPTree(self._copy_node(self.root))

    def _copy_node(self, node):
        """Copia recursivamente un nodo y sus hijos."""
        if isinstance(node, TerminalNode):
            new_node = TerminalNode(node.symbol, node.is_variable, node.value)
        else:
            new_node = FunctionNode(node.function_name, node.function_impl, node.arity_val)

        # Copiar hijos
        for child in node.children:
            new_child = self._copy_node(child)
            new_child.parent = new_node
            new_node.children.append(new_child)

        return new_node


class NativeGeneticProgramming:
    """Implementación nativa de programación genética para clasificación."""

    def __init__(self, population_size=100, generations=20, tournament_size=7,
                 crossover_prob=0.8, mutation_prob=0.2, max_depth=6,
                 parsimony_coefficient=0.01, random_state=42):

        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_depth = max_depth
        self.parsimony_coefficient = parsimony_coefficient
        self.random_state = random_state

        # Configurar funciones y terminales
        self._setup_function_set()
        self.terminal_set = []

        # Estado interno
        self.population = []
        self.best_individual = None
        self.fitness_history = []

        # Configurar semilla aleatoria
        random.seed(random_state)
        np.random.seed(random_state)

    def _setup_function_set(self):
        """Configura el conjunto de funciones disponibles."""

        def safe_divide(a, b):
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.divide(a, b)
                result[np.isnan(result)] = 1.0
                result[np.isinf(result)] = 1.0
                return result

        def safe_sqrt(a):
            return np.sqrt(np.abs(a))

        def safe_log(a):
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.log(np.abs(a) + 1e-10)
                result[np.isnan(result)] = 0.0
                result[np.isinf(result)] = 0.0
                return result

        def safe_sin(a):
            return np.sin(np.clip(a, -10, 10))

        def safe_cos(a):
            return np.cos(np.clip(a, -10, 10))

        self.function_set = {
            'add': (lambda a, b: a + b, 2),
            'sub': (lambda a, b: a - b, 2),
            'mul': (lambda a, b: a * b, 2),
            'div': (safe_divide, 2),
            'sqrt': (safe_sqrt, 1),
            'log': (safe_log, 1),
            'sin': (safe_sin, 1),
            'cos': (safe_cos, 1),
            'abs': (lambda a: np.abs(a), 1),
            'neg': (lambda a: -a, 1)
        }

    def _create_terminal_set(self, n_features):
        """Crea el conjunto de terminales basado en el número de características."""
        self.terminal_set = []

        # Variables de entrada
        for i in range(n_features):
            self.terminal_set.append(TerminalNode(f'X{i}', is_variable=True))

        # Constantes aleatorias
        for _ in range(5):
            value = random.uniform(-1.0, 1.0)
            self.terminal_set.append(TerminalNode(f'C{value:.3f}', is_variable=False, value=value))

    def _create_random_tree(self, method='grow', max_depth=6):
        """Crea un árbol aleatorio usando el método especificado."""
        if method == 'grow':
            return self._grow_tree(max_depth)
        elif method == 'full':
            return self._full_tree(max_depth)
        else:  # half-and-half
            if random.random() < 0.5:
                return self._grow_tree(max_depth)
            else:
                return self._full_tree(max_depth)

    def _grow_tree(self, max_depth):
        """Método grow: puede elegir terminales antes de alcanzar max_depth."""
        if max_depth <= 1 or (max_depth > 1 and random.random() < 0.5):
            # Crear terminal
            return random.choice(self.terminal_set)
        else:
            # Crear función
            func_name = random.choice(list(self.function_set.keys()))
            func_impl, arity = self.function_set[func_name]
            node = FunctionNode(func_name, func_impl, arity)

            # Crear hijos
            for _ in range(arity):
                child = self._grow_tree(max_depth - 1)
                child.parent = node
                node.children.append(child)

            return node

    def _full_tree(self, max_depth):
        """Método full: usa solo funciones hasta alcanzar max_depth."""
        if max_depth <= 1:
            return random.choice(self.terminal_set)
        else:
            func_name = random.choice(list(self.function_set.keys()))
            func_impl, arity = self.function_set[func_name]
            node = FunctionNode(func_name, func_impl, arity)

            for _ in range(arity):
                child = self._full_tree(max_depth - 1)
                child.parent = node
                node.children.append(child)

            return node

    def _evaluate_fitness(self, tree, X, y):
        """Evalúa la aptitud de un árbol."""
        try:
            # Obtener predicciones del árbol
            raw_output = tree.evaluate(X)

            # Convertir a probabilidades usando sigmoide
            probabilities = 1 / (1 + np.exp(-np.clip(raw_output, -500, 500)))

            # Convertir a predicciones binarias
            predictions = (probabilities > 0.5).astype(int)

            # Calcular exactitud
            accuracy = np.mean(predictions == y)

            # Penalización por complejidad (parsimonia)
            complexity_penalty = self.parsimony_coefficient * tree.size()

            # Fitness final
            fitness = accuracy - complexity_penalty

            return max(fitness, 0.0)  # Asegurar que el fitness no sea negativo

        except Exception as e:
            print(f"Error evaluating fitness: {e}")
            return 0.0

    def _tournament_selection(self, population, fitnesses):
        """Selección por torneo."""
        tournament_indices = random.sample(range(len(population)),
                                           min(self.tournament_size, len(population)))
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx].copy()

    def _crossover(self, parent1, parent2):
        """Cruce de dos árboles."""
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Encontrar nodos de cruce
        nodes1 = self._get_all_nodes(child1.root)
        nodes2 = self._get_all_nodes(child2.root)

        if len(nodes1) > 1 and len(nodes2) > 1:
            # Seleccionar nodos aleatoriamente (evitar la raíz)
            node1 = random.choice(nodes1[1:]) if len(nodes1) > 1 else nodes1[0]
            node2 = random.choice(nodes2[1:]) if len(nodes2) > 1 else nodes2[0]

            # Intercambiar subárboles
            if node1.parent and node2.parent:
                # Encontrar índices de los nodos en sus padres
                idx1 = node1.parent.children.index(node1)
                idx2 = node2.parent.children.index(node2)

                # Intercambiar
                node1.parent.children[idx1] = node2
                node2.parent.children[idx2] = node1

                # Actualizar padres
                node1.parent, node2.parent = node2.parent, node1.parent

        return child1, child2

    def _mutate(self, tree):
        """Mutación de un árbol."""
        nodes = self._get_all_nodes(tree.root)
        if not nodes:
            return tree

        # Seleccionar nodo aleatorio
        node = random.choice(nodes)

        # Crear nuevo subárbol
        new_subtree = self._create_random_tree('grow', max_depth=3)

        # Reemplazar
        if node.parent:
            idx = node.parent.children.index(node)
            node.parent.children[idx] = new_subtree
            new_subtree.parent = node.parent
        else:
            # Es la raíz
            tree.root = new_subtree

        return tree

    @staticmethod
    def _get_all_nodes(root):
        """Obtiene todos los nodos del árbol."""
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

    def evolve(self, X, y):
        """Ejecuta el algoritmo evolutivo."""
        # Crear conjunto de terminales
        self._create_terminal_set(X.shape[1])

        # Inicializar población
        print(f"Inicializando población de {self.population_size} individuos...")
        self.population = []
        for _ in range(self.population_size):
            tree = GPTree(self._create_random_tree('half-and-half', self.max_depth))
            self.population.append(tree)

        # Evolución
        for generation in range(self.generations):
            # Evaluar fitness
            fitnesses = [self._evaluate_fitness(tree, X, y) for tree in self.population]

            # Guardar mejor individuo
            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]

            if self.best_individual is None or best_fitness > self._evaluate_fitness(self.best_individual, X, y):
                self.best_individual = self.population[best_idx].copy()

            self.fitness_history.append(best_fitness)

            if generation % 5 == 0:
                print(f"Generación {generation}: Mejor fitness = {best_fitness:.4f}, "
                      f"Tamaño = {self.population[best_idx].size()}")

            # Crear nueva población
            # Elitismo: mantener el mejor
            new_population = [self.population[best_idx].copy()]

            # Generar resto de la población
            while len(new_population) < self.population_size:
                # Selección
                parent1 = self._tournament_selection(self.population, fitnesses)
                parent2 = self._tournament_selection(self.population, fitnesses)

                # Cruce
                if random.random() < self.crossover_prob:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutación
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

        print(f"Evolución completada. Mejor fitness final: {self.fitness_history[-1]:.4f}")
        return self.best_individual


class NativeSymbolicClassifier(BaseEstimator, ClassifierMixin):
    """Clasificador simbólico nativo usando programación genética."""

    def __init__(self, population_size=100, generations=20, tournament_size=7,
                 crossover_prob=0.8, mutation_prob=0.2, max_depth=6,
                 parsimony_coefficient=0.01, random_state=42):

        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_depth = max_depth
        self.parsimony_coefficient = parsimony_coefficient
        self.random_state = random_state

        self.gp_engine = None
        self.best_tree = None
        self.classes_ = None
        self._estimator_type = "classifier"

    def fit(self, X, y):
        """Entrena el clasificador simbólico."""
        # Validar entrada
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        # Convertir a problema binario si es necesario
        if len(self.classes_) != 2:
            raise ValueError("Actualmente solo se soporta clasificación binaria")

        # Asegurar que las clases sean 0 y 1
        y_binary = np.array((y == self.classes_[1])).astype(int)

        # Crear motor de programación genética
        self.gp_engine = NativeGeneticProgramming(
            population_size=self.population_size,
            generations=self.generations,
            tournament_size=self.tournament_size,
            crossover_prob=self.crossover_prob,
            mutation_prob=self.mutation_prob,
            max_depth=self.max_depth,
            parsimony_coefficient=self.parsimony_coefficient,
            random_state=self.random_state
        )

        # Evolucionar
        self.best_tree = self.gp_engine.evolve(X, y_binary)

        return self

    def predict(self, X):
        """Realiza predicciones."""
        if self.best_tree is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir")

        X = check_array(X)

        # Obtener salida del árbol
        raw_output = self.best_tree.evaluate(X)

        # Convertir a probabilidades
        probabilities = 1 / (1 + np.exp(-np.clip(raw_output, -500, 500)))

        # Convertir a predicciones de clase
        binary_predictions = (probabilities > 0.5).astype(int)

        # Mapear de vuelta a clases originales
        return np.where(binary_predictions == 1, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        """Predice probabilidades de clase."""
        if self.best_tree is None:
            raise ValueError("El modelo debe ser entrenado antes de predecir probabilidades")

        X = check_array(X)

        # Obtener salida del árbol
        raw_output = self.best_tree.evaluate(X)

        # Convertir a probabilidades
        prob_class_1 = 1 / (1 + np.exp(-np.clip(raw_output, -500, 500)))
        prob_class_0 = 1 - prob_class_1

        return np.column_stack([prob_class_0, prob_class_1])

    def get_expression(self):
        """Obtiene la expresión simbólica como cadena."""
        if self.best_tree is None:
            return "Modelo no entrenado"
        return str(self.best_tree)

    def get_complexity(self):
        """Obtiene medidas de complejidad del modelo."""
        if self.best_tree is None:
            return {'size': 0, 'depth': 0}
        return {
            'size': self.best_tree.size(),
            'depth': self.best_tree.depth()
        }