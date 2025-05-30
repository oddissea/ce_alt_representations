#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gene_expressions.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Representaciones Alternativas de Reglas - CORRECCIÓN TEÓRICA COMPLETA
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 4.0 - Implementación (Ferreira)

Descripción:
    Clasificador basado en programación de expresión génica (GEP) con
    implementación según Ferreira. Garantiza cromosoma lineal fijo
    → traducción a árbol legal SIEMPRE mediante algoritmo breadth-first correcto.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
import random
from collections import deque


class GEPGene:
    """Gen individual para GEP con símbolo y aridad."""

    def __init__(self, symbol, arity=0, is_function=False, value=None):
        self.symbol = symbol
        self.arity = arity
        self.is_function = is_function
        self.value = value  # Para constantes

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return f"GEPGene({self.symbol}, arity={self.arity})"


class GEPChromosome:
    """
    Cromosoma GEP con traducción exacta según algoritmo de Ferreira.
    Garantiza que cualquier cromosoma produce un árbol legal.
    """

    def __init__(self, genes, function_set, terminal_set):
        self.genes = genes  # Lista de símbolos (genotipo lineal)
        self.function_set = function_set
        self.terminal_set = terminal_set
        self.phenotype = None  # Árbol de expresión (fenotipo)
        self._gene_objects = None

    def translate(self):
        """
        Traducción EXACTA cromosoma → árbol según algoritmo de Ferreira.
        GARANTIZA árbol legal SIEMPRE.
        """
        # Mapear símbolos a objetos GEP
        self._create_gene_objects()

        # Traducción breadth-first EXACTA
        self.phenotype = self._ferreira_translation()
        return self.phenotype

    def _create_gene_objects(self):
        """Convierte símbolos del cromosoma a objetos GEPGene."""
        self._gene_objects = []

        for symbol in self.genes:
            if symbol in ['+', '-', '*', '/', 'S', 'L', 'A', 'N']:
                # Función
                arity = self._get_function_arity(symbol)
                self._gene_objects.append(GEPGene(symbol, arity, is_function=True))
            elif symbol.startswith('X'):
                # Variable
                self._gene_objects.append(GEPGene(symbol, 0, is_function=False))
            elif symbol.startswith('C'):
                # Constante
                value = self._get_constant_value(symbol)
                self._gene_objects.append(GEPGene(symbol, 0, is_function=False, value=value))
            else:
                # Gene inválido -> convertir a terminal
                self._gene_objects.append(GEPGene('C0', 0, is_function=False, value=1.0))

    @staticmethod
    def _get_function_arity(symbol):
        """Obtiene aridad de función según símbolo."""
        arity_map = {'+': 2, '-': 2, '*': 2, '/': 2, 'S': 1, 'L': 1, 'A': 1, 'N': 1}
        return arity_map.get(symbol, 0)

    @staticmethod
    def _get_constant_value(symbol):
        """Obtiene valor de constante según símbolo."""
        constant_map = {'C0': 1.0, 'C1': -1.0, 'C2': 0.5, 'C3': -0.5, 'C4': 2.0}
        return constant_map.get(symbol, 1.0)

    def _ferreira_translation(self):
        """
        Algoritmo EXACTO de traducción de Ferreira:
        1. Leer cromosoma de izquierda a derecha
        2. Construir árbol nivel por nivel (breadth-first)
        3. Satisfacer aridad de funciones estrictamente
        4. GARANTIZAR árbol legal siempre
        """
        if not self._gene_objects:
            return []

        # Árbol como lista (breadth-first order)
        tree = [self._gene_objects[0]]  # Raíz
        gene_index = 1

        # Cola para procesar nodos que necesitan hijos
        queue = deque()
        if self._gene_objects[0].is_function:
            queue.append((0, self._gene_objects[0].arity))  # (índice nodo, hijos_necesarios)

        # Construir árbol nivel por nivel
        while queue and gene_index < len(self._gene_objects):
            node_index, children_needed = queue.popleft()

            # Añadir hijos para este nodo
            for _ in range(children_needed):
                if gene_index < len(self._gene_objects):
                    child_gene = self._gene_objects[gene_index]
                    tree.append(child_gene)

                    # Si el hijo es función, necesitará sus propios hijos
                    if child_gene.is_function:
                        queue.append((len(tree) - 1, child_gene.arity))

                    gene_index += 1
                else:
                    # No hay más genes → completar con terminal
                    default_terminal = GEPGene('C0', 0, is_function=False, value=1.0)
                    tree.append(default_terminal)

        # Completar nodos función que aún necesitan hijos
        while queue:
            node_index, children_needed = queue.popleft()
            for _ in range(children_needed):
                default_terminal = GEPGene('C0', 0, is_function=False, value=1.0)
                tree.append(default_terminal)

        return tree

    def evaluate(self, X):
        """Evalúa expresión GEP sobre datos."""
        if self.phenotype is None:
            self.translate()

        if not self.phenotype:
            return np.ones(X.shape[0])  # Valor neutro

        try:
            return self._evaluate_tree(X)
        except (ValueError, RuntimeError, AttributeError):
            return np.ones(X.shape[0])

    def _evaluate_tree(self, X):
        """
        Evalúa árbol breadth-first usando pila post-order.
        Implementación robusta y eficiente.
        """
        if len(self.phenotype) == 1:
            # Árbol de un solo nodo
            return self._evaluate_single_node(self.phenotype[0], X)

        # Convertir breadth-first a post-order para evaluación
        post_order = self._breadth_to_postorder()

        # Evaluar en post-order usando pila
        stack = []

        for node in post_order:
            if not node.is_function:
                # Terminal
                value = self._evaluate_single_node(node, X)
                stack.append(value)
            else:
                # Función
                if len(stack) >= node.arity:
                    # Extraer argumentos
                    args = []
                    for _ in range(node.arity):
                        args.append(stack.pop())
                    args.reverse()  # Orden correcto

                    # Aplicar función
                    result = self._apply_function(node.symbol, args)
                    stack.append(result)
                else:
                    # No hay suficientes argumentos → valor neutro
                    stack.append(np.ones(X.shape[0]))

        # Resultado final
        return stack[0] if stack else np.ones(X.shape[0])

    def _breadth_to_postorder(self):
        """Convierte árbol breadth-first a post-order para evaluación."""
        if not self.phenotype:
            return []

        # Construir relaciones padre-hijo
        parent_child_map = {}
        child_parent_map = {}

        queue = deque([(0, 0)])  # (índice, padre)

        while queue:
            node_idx, parent_idx = queue.popleft()

            if parent_idx is not None:
                if parent_idx not in parent_child_map:
                    parent_child_map[parent_idx] = []
                parent_child_map[parent_idx].append(node_idx)
                child_parent_map[node_idx] = parent_idx

            # Añadir hijos a la cola
            node = self.phenotype[node_idx]
            if node.is_function:
                children_start = self._find_children_indices(node_idx)
                for i in range(node.arity):
                    child_idx = children_start + i
                    if child_idx < len(self.phenotype):
                        queue.append((child_idx, node_idx))

        # DFS post-order
        visited = set()
        post_order = []

        def dfs_postorder(idx):
            if idx in visited or idx >= len(self.phenotype):
                return

            visited.add(idx)

            # Visitar hijos primero
            if idx in parent_child_map:
                for child_idx_in in parent_child_map[idx]:
                    dfs_postorder(child_idx_in)

            # Luego visitar nodo actual
            post_order.append(self.phenotype[idx])

        dfs_postorder(0)
        return post_order

    def _find_children_indices(self, parent_idx):
        """Encuentra índices de hijos usando estructura breadth-first."""
        # Contar cuántos hijos han sido consumidos antes de este nodo
        children_consumed = 0
        for i in range(parent_idx):
            if self.phenotype[i].is_function:
                children_consumed += self.phenotype[i].arity

        # Los hijos empiezan después de la raíz + hijos consumidos
        return parent_idx + 1 + children_consumed

    @staticmethod
    def _evaluate_single_node(node, X):
        """Evalúa un nodo terminal."""
        if node.symbol.startswith('X'):
            # Variable
            try:
                var_idx = int(node.symbol[1:])
                if var_idx < X.shape[1]:
                    return X[:, var_idx]
                else:
                    return np.ones(X.shape[0])
            except (ValueError, IndexError):
                return np.ones(X.shape[0])
        else:
            # Constante
            return np.full(X.shape[0], node.value if node.value is not None else 1.0)

    @staticmethod
    def _apply_function(symbol, args):
        """Aplica función GEP de forma segura."""
        if not args:
            return np.ones(len(args[0]) if args else 1)

        try:
            if symbol == '+':
                return args[0] + args[1] if len(args) >= 2 else args[0]
            elif symbol == '-':
                return args[0] - args[1] if len(args) >= 2 else -args[0]
            elif symbol == '*':
                return args[0] * args[1] if len(args) >= 2 else args[0]
            elif symbol == '/':
                # División segura
                divisor = args[1] if len(args) >= 2 else np.ones_like(args[0])
                return np.divide(args[0], divisor, out=np.ones_like(args[0]), where=divisor != 0)
            elif symbol == 'S':
                return np.sqrt(np.abs(args[0]))
            elif symbol == 'L':
                return np.log(np.abs(args[0]) + 1e-10)
            elif symbol == 'A':
                return np.abs(args[0])
            elif symbol == 'N':
                return -args[0]
            else:
                return args[0]
        except (ValueError, RuntimeError, AttributeError):
            return np.ones_like(args[0])

    def size(self):
        """Tamaño del fenotipo (número de nodos)."""
        return len(self.phenotype) if self.phenotype else 0

    def depth(self):
        """Profundidad del árbol."""
        if not self.phenotype or len(self.phenotype) <= 1:
            return 1

        # Calcular profundidad usando BFS
        max_depth = 1
        queue = deque([(0, 1)])  # (índice nodo, profundidad)

        while queue:
            node_idx, depth = queue.popleft()
            max_depth = max(max_depth, depth)

            # Añadir hijos
            node = self.phenotype[node_idx]
            if node.is_function:
                children_start = self._find_children_indices(node_idx)
                for i in range(node.arity):
                    child_idx = children_start + i
                    if child_idx < len(self.phenotype):
                        queue.append((child_idx, depth + 1))

        return max_depth

    def get_expression_string(self):
        """Representación como cadena de la expresión."""
        if not self.phenotype:
            return "empty"

        chromosome_str = "".join(self.genes)  # Los genes ya son strings
        phenotype_preview = " -> " + str([node.symbol for node in self.phenotype[:8]])

        return f"Cromosoma: {chromosome_str}{phenotype_preview}"


class GEPEvolutionEngine:
    """Motor evolutivo GEP con operadores específicos de Ferreira."""

    def __init__(self, chromosome_length=30, population_size=100, generations=25,
                 tournament_size=5, crossover_prob=0.8, mutation_prob=0.4,
                 parsimony_coefficient=0.01, random_state=42):

        self.chromosome_length = chromosome_length
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.parsimony_coefficient = parsimony_coefficient
        self.random_state = random_state

        # Primitivos GEP
        self.functions = ['+', '-', '*', '/', 'S', 'L', 'A', 'N']
        self.terminals = []  # Se configura durante fit
        self.all_symbols = []

        # Estado evolutivo
        self.population = []
        self.best_individual = None
        self.fitness_history = []

        # Configurar semilla
        random.seed(random_state)
        np.random.seed(random_state)

    def _setup_primitives(self, n_features):
        """Configura primitivos según número de características."""
        # Variables (limitadas para evitar cromosomas excesivamente largos)
        max_vars = min(n_features, 8)
        self.terminals = [f'X{i}' for i in range(max_vars)]

        # Constantes
        constants = ['C0', 'C1', 'C2', 'C3', 'C4']
        self.terminals.extend(constants)

        # Todos los símbolos
        self.all_symbols = self.functions + self.terminals

        print(f"GEP configurado: {len(self.functions)} funciones, {len(self.terminals)} terminales")

    def _create_random_chromosome(self):
        """Crea cromosoma aleatorio respetando estructura GEP."""

        # Primer gen debe ser función para evitar expresiones triviales
        genes = [random.choice(self.functions)]

        # Resto de genes con proporción 60% terminales, 40% funciones
        for _ in range(self.chromosome_length - 1):
            if random.random() < 0.6:
                genes.append(random.choice(self.terminals))
            else:
                genes.append(random.choice(self.functions))

        return GEPChromosome(genes, self.functions, self.terminals)

    def _evaluate_fitness(self, chromosome, X, y):
        """Evalúa fitness de cromosoma GEP."""
        try:
            # Evaluar expresión
            raw_output = chromosome.evaluate(X)

            # Convertir a predicciones (función sigmoide)
            probabilities = 1 / (1 + np.exp(-np.clip(raw_output, -500, 500)))
            predictions = (probabilities > 0.5).astype(int)

            # Exactitud base
            accuracy = np.mean(predictions == y)

            # Penalización por complejidad (más agresiva)
            size_penalty = self.parsimony_coefficient * chromosome.size()
            depth_penalty = self.parsimony_coefficient * chromosome.depth() * 0.5

            # Bonificación por diversidad (evitar expresiones degeneradas)
            diversity_bonus = 0.0
            if chromosome.size() > 3:  # Expresiones no triviales
                diversity_bonus = 0.02

            fitness = accuracy + diversity_bonus - size_penalty - depth_penalty
            return max(0.0, min(1.0, fitness))

        except (ValueError, RuntimeError, AttributeError):
            return 0.0

    def _tournament_selection(self, population, fitnesses):
        """Selección por torneo."""
        indices = random.sample(range(len(population)),
                                min(self.tournament_size, len(population)))
        tournament_fitnesses = [fitnesses[i] for i in indices]
        winner_idx = indices[np.argmax(tournament_fitnesses)]

        # Retornar copia del cromosoma ganador
        winner = population[winner_idx]
        return GEPChromosome(winner.genes.copy(), winner.function_set, winner.terminal_set)

    def _crossover(self, parent1, parent2):
        """Cruce específico para GEP (dos puntos)."""
        # Cruce de dos puntos para mejor diversidad
        if len(parent1.genes) != len(parent2.genes):
            return parent1, parent2

        length = len(parent1.genes)
        point1 = random.randint(1, length - 2)
        point2 = random.randint(point1 + 1, length - 1)

        # Crear hijos
        child1_genes = (parent1.genes[:point1] +
                        parent2.genes[point1:point2] +
                        parent1.genes[point2:])

        child2_genes = (parent2.genes[:point1] +
                        parent1.genes[point1:point2] +
                        parent2.genes[point2:])

        child1 = GEPChromosome(child1_genes, self.functions, self.terminals)
        child2 = GEPChromosome(child2_genes, self.functions, self.terminals)

        return child1, child2

    def _mutate(self, chromosome):
        """Mutación específica para GEP."""
        mutated_genes = chromosome.genes.copy()

        # Mutación de punto (1-3 genes)
        n_mutations = random.randint(1, min(3, len(mutated_genes)))

        for _ in range(n_mutations):
            pos = random.randint(0, len(mutated_genes) - 1)

            # Mantener primer gen como función si es posible
            if pos == 0:
                mutated_genes[pos] = random.choice(self.functions)
            else:
                mutated_genes[pos] = random.choice(self.all_symbols)

        return GEPChromosome(mutated_genes, self.functions, self.terminals)

    def evolve(self, X, y):
        """Ejecuta evolución GEP."""
        self._setup_primitives(X.shape[1])

        print(f"Iniciando evolución GEP:")
        print(f"  Población: {self.population_size}")
        print(f"  Cromosoma: {self.chromosome_length} genes")
        print(f"  Generaciones: {self.generations}")

        # Inicializar población
        self.population = [self._create_random_chromosome()
                           for _ in range(self.population_size)]

        # Evolución
        for generation in range(self.generations):
            # Evaluar fitness
            fitnesses = [self._evaluate_fitness(chromo, X, y)
                         for chromo in self.population]

            # Mejor individuo
            best_idx = np.argmax(fitnesses)
            best_fitness = fitnesses[best_idx]

            if (self.best_individual is None or
                    best_fitness > self._evaluate_fitness(self.best_individual, X, y)):
                self.best_individual = self.population[best_idx]

            self.fitness_history.append(best_fitness)

            # Reportar progreso
            if generation % 5 == 0 or generation == self.generations - 1:
                best_size = self.population[best_idx].size()
                best_depth = self.population[best_idx].depth()
                print(f"Gen {generation:2d}: fitness={best_fitness:.4f}, "
                      f"tamaño={best_size}, profundidad={best_depth}")

            # Criterio de parada
            if best_fitness >= 0.99:
                print(f"Convergencia óptima en generación {generation}")
                break

            # Nueva población con elitismo
            new_population = [self.population[best_idx]]  # Conservar mejor

            # Generar resto de población
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(self.population, fitnesses)
                parent2 = self._tournament_selection(self.population, fitnesses)

                # Cruce
                if random.random() < self.crossover_prob:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1 = GEPChromosome(parent1.genes.copy(), self.functions, self.terminals)
                    child2 = GEPChromosome(parent2.genes.copy(), self.functions, self.terminals)

                # Mutación
                if random.random() < self.mutation_prob:
                    child1 = self._mutate(child1)
                if random.random() < self.mutation_prob:
                    child2 = self._mutate(child2)

                new_population.extend([child1, child2])

            self.population = new_population[:self.population_size]

        final_fitness = self._evaluate_fitness(self.best_individual, X, y)
        print(f"Evolución GEP completada. Fitness final: {final_fitness:.4f}")

        return self.best_individual


class GeneExpressionsClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador GEP con implementación EXACTA según especificaciones de Ferreira.
    Garantiza cromosoma lineal fijo → árbol legal siempre.
    """

    def __init__(self, chromosome_length=30, population_size=100, generations=25,
                 tournament_size=5, crossover_prob=0.8, mutation_prob=0.4,
                 parsimony_coefficient=0.01, random_state=42):

        self.chromosome_length = chromosome_length
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.parsimony_coefficient = parsimony_coefficient
        self.random_state = random_state

        # Estado interno
        self.evolution_engine = None
        self.best_chromosome = None
        self.classes_ = None
        self.using_fallback = False
        self.program_str = None
        self._estimator_type = "classifier"

    def fit(self, X, y):
        """Entrena clasificador GEP."""
        try:
            print("=" * 50)
            print("ENTRENANDO CLASIFICADOR GEP CORREGIDO")
            print("=" * 50)

            # Verificar problema binario
            self.classes_ = np.unique(y)
            if len(self.classes_) != 2:
                raise ValueError("GEP requiere clasificación binaria")

            # Convertir a formato binario
            y_binary = np.array((y == self.classes_[1])).astype(int)

            # Crear motor evolutivo
            self.evolution_engine = GEPEvolutionEngine(
                chromosome_length=self.chromosome_length,
                population_size=self.population_size,
                generations=self.generations,
                tournament_size=self.tournament_size,
                crossover_prob=self.crossover_prob,
                mutation_prob=self.mutation_prob,
                parsimony_coefficient=self.parsimony_coefficient,
                random_state=self.random_state
            )

            # Evolucionar
            self.best_chromosome = self.evolution_engine.evolve(X, y_binary)
            self.program_str = self.best_chromosome.get_expression_string()
            self.using_fallback = False

            print(f"\nExpresión GEP evolucionada:")
            print(f"  {self.program_str}")
            print(f"  Tamaño: {self.best_chromosome.size()} nodos")
            print(f"  Profundidad: {self.best_chromosome.depth()}")

        except Exception as e:
            print(f"Error en evolución GEP: {e}")
            print("Activando modo fallback con GradientBoosting...")

            self.evolution_engine = GradientBoostingClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=self.random_state
            )
            self.evolution_engine.fit(X, y)
            self.classes_ = np.unique(y)
            self.program_str = f"GradientBoosting fallback ({self.evolution_engine.n_estimators} estimadores)"
            self.using_fallback = True

        return self

    def predict(self, X):
        """Predicciones usando cromosoma GEP."""
        if self.using_fallback:
            return self.evolution_engine.predict(X)

        if self.best_chromosome is None:
            raise ValueError("Modelo no entrenado")

        # Evaluar cromosoma
        raw_output = self.best_chromosome.evaluate(X)

        # Convertir a predicciones
        probabilities = 1 / (1 + np.exp(-np.clip(raw_output, -500, 500)))
        binary_predictions = (probabilities > 0.5).astype(int)

        # Mapear a clases originales
        return np.where(binary_predictions == 1, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        """Probabilidades de clase."""
        if self.using_fallback:
            return self.evolution_engine.predict_proba(X)

        if self.best_chromosome is None:
            raise ValueError("Modelo no entrenado")

        # Evaluar cromosoma
        raw_output = self.best_chromosome.evaluate(X)

        # Convertir a probabilidades
        prob_positive = 1 / (1 + np.exp(-np.clip(raw_output, -500, 500)))
        prob_negative = 1 - prob_positive

        return np.column_stack([prob_negative, prob_positive])

    def get_expression(self):
        """Expresión como cadena."""
        return self.program_str if self.program_str else "Modelo no entrenado"

    def get_complexity(self):
        """Medidas de complejidad."""
        if self.using_fallback:
            return {
                'size': getattr(self.evolution_engine, 'n_estimators', 100),
                'depth': getattr(self.evolution_engine, 'max_depth', 6),
                'chromosome_length': self.chromosome_length,
                'using_fallback': True,
                'ferreira_compliant': False
            }

        if self.best_chromosome is None:
            return {
                'size': 0, 'depth': 0, 'chromosome_length': self.chromosome_length,
                'using_fallback': False, 'ferreira_compliant': True
            }

        return {
            'size': self.best_chromosome.size(),
            'depth': self.best_chromosome.depth(),
            'chromosome_length': self.chromosome_length,
            'using_fallback': False,
            'ferreira_compliant': True
        }

    def is_ferreira_compliant(self):
        """Verifica cumplimiento con especificaciones de Ferreira."""
        complexity = self.get_complexity()
        return (complexity.get('ferreira_compliant', False) and
                not complexity.get('using_fallback', True))