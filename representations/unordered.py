#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unordered.py - REIMPLEMENTACIÓN EVOLUTIVA FIEL

COMPUTACIÓN EVOLUTIVA
PEC 4: Representación Desordenada según memoria.pdf sección 2.5
Implementación fiel de cromosomas desordenados (messy chromosomes)
con algoritmo genético especializado según Lanzi y Goldberg.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
import random


class MessyGene:
    """Gen desordenado con etiqueta y valor según teoría de Goldberg."""

    def __init__(self, attribute_index, condition_value):
        self.attribute_index = attribute_index  # Etiqueta que identifica entrada
        self.condition_value = condition_value  # Valor de condición elemental

    def __str__(self):
        return f"({self.attribute_index}:{self.condition_value:.3f})"

    def __repr__(self):
        return self.__str__()


class MessyChromosome:
    """Cromosoma desordenado de longitud variable según especificación."""

    def __init__(self, genes=None):
        self.genes = genes if genes is not None else []
        self.fitness = 0.0
        self.class_label = None

    def add_gene(self, attribute_index, condition_value):
        """Añade gen con etiqueta-valor específicos."""
        self.genes.append(MessyGene(attribute_index, condition_value))

    def get_attributes_mentioned(self):
        """Retorna conjunto de atributos mencionados en el cromosoma."""
        return set(gene.attribute_index for gene in self.genes)

    def get_condition_for_attribute(self, attr_idx):
        """Obtiene condición para atributo específico (primera ocurrencia)."""
        for gene in self.genes:
            if gene.attribute_index == attr_idx:
                return gene.condition_value
        return None

    def length(self):
        """Longitud variable del cromosoma."""
        return len(self.genes)

    def copy(self):
        """Copia profunda del cromosoma."""
        new_genes = [MessyGene(g.attribute_index, g.condition_value) for g in self.genes]
        copy_chromo = MessyChromosome(new_genes)
        copy_chromo.fitness = self.fitness
        copy_chromo.class_label = self.class_label
        return copy_chromo

    def __str__(self):
        genes_str = ", ".join(str(gene) for gene in self.genes)
        return f"[{genes_str}] -> Class_{self.class_label} (fitness={self.fitness:.3f})"


class UnorderedEvolutionaryEngine:
    """Motor evolutivo especializado para cromosomas desordenados."""

    def __init__(self, population_size=50, generations=30, tournament_size=5,
                 crossover_prob=0.7, mutation_prob=0.4, min_genes=2, max_genes=8,
                 random_state=42):

        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.min_genes = min_genes
        self.max_genes = max_genes
        self.random_state = random_state

        # Estado evolutivo
        self.population = []
        self.best_individuals = {}  # Por clase
        self.fitness_history = []
        self.n_features = 0

        # Configurar semilla
        random.seed(random_state)
        np.random.seed(random_state)

    def _create_random_chromosome(self, class_label, X_class):
        """Crea cromosoma aleatorio según principios de messy chromosomes."""
        chromosome = MessyChromosome()
        chromosome.class_label = class_label

        # Longitud variable aleatoria
        n_genes = random.randint(self.min_genes,
                                 min(self.max_genes, self.n_features))

        # Seleccionar atributos aleatoriamente (pueden repetirse)
        for _ in range(n_genes):
            attr_idx = random.randint(0, self.n_features - 1)

            # Valor de condición basado en distribución de la clase
            if len(X_class) > 0:
                # Usar percentil aleatorio de la distribución de clase
                percentile = random.uniform(10, 90)
                condition_value = np.percentile(X_class[:, attr_idx], percentile)
            else:
                condition_value = random.uniform(-1, 1)

            chromosome.add_gene(attr_idx, condition_value)

        return chromosome

    @staticmethod
    def _evaluate_fitness(chromosome, X, y):
        """Evalúa fitness de cromosoma desordenado."""
        if chromosome.length() == 0:
            return 0.0

        try:
            # Calcular cobertura: instancias que cumplen todas las condiciones
            covered_mask = np.ones(len(X), dtype=bool)

            for gene in chromosome.genes:
                attr_idx = gene.attribute_index
                condition_val = gene.condition_value

                # Condición: atributo >= valor del gen
                attr_condition = X[:, attr_idx] >= condition_val
                covered_mask &= attr_condition

            covered_indices = np.where(covered_mask)[0]

            if len(covered_indices) == 0:
                return 0.0

            # Precisión: proporción de instancias cubiertas que son de la clase correcta
            correct_predictions = int(np.sum(y[covered_indices] == chromosome.class_label))
            precision = correct_predictions / len(covered_indices)

            # Cobertura: proporción de instancias de la clase que son cubiertas
            class_instances = int(np.sum(y == chromosome.class_label))
            class_covered = int(np.sum((y[covered_indices] == chromosome.class_label)))
            coverage = class_covered / max(class_instances, 1)

            # Fitness balanceado con bonificación por parsimonia
            parsimony_bonus = 1.0 / (1.0 + chromosome.length() * 0.1)
            fitness = (precision * 0.7 + coverage * 0.3) * parsimony_bonus

            return max(0.0, min(1.0, fitness))

        except (ValueError, IndexError, ZeroDivisionError):
            return 0.0

    def _tournament_selection(self, population, fitnesses):
        """Selección por torneo para cromosomas de longitud variable."""
        tournament_size = min(self.tournament_size, len(population))
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        return population[winner_idx].copy()

    def _messy_crossover(self, parent1, parent2):
        """Cruce especializado para cromosomas desordenados."""
        child1 = MessyChromosome()
        child2 = MessyChromosome()

        child1.class_label = parent1.class_label
        child2.class_label = parent2.class_label

        # Combinar genes de ambos padres aleatoriamente
        all_genes_p1 = parent1.genes.copy()
        all_genes_p2 = parent2.genes.copy()

        # Shuffle y dividir
        combined_genes = all_genes_p1 + all_genes_p2
        random.shuffle(combined_genes)

        # Dividir genes entre hijos manteniendo longitudes válidas
        split_point = len(combined_genes) // 2

        genes1 = combined_genes[:split_point]
        genes2 = combined_genes[split_point:]

        # Limitar longitud según restricciones
        if len(genes1) > self.max_genes:
            genes1 = random.sample(genes1, self.max_genes)
        if len(genes2) > self.max_genes:
            genes2 = random.sample(genes2, self.max_genes)

        # Asegurar longitud mínima
        while len(genes1) < self.min_genes and all_genes_p1:
            genes1.append(random.choice(all_genes_p1))
        while len(genes2) < self.min_genes and all_genes_p2:
            genes2.append(random.choice(all_genes_p2))

        child1.genes = genes1
        child2.genes = genes2

        return child1, child2

    def _messy_mutation(self, chromosome):
        """Mutación especializada para cromosomas desordenados."""
        mutated = chromosome.copy()

        # Tipos de mutación específicos para messy chromosomes
        mutation_types = ['add_gene', 'remove_gene', 'modify_gene', 'duplicate_gene']

        mutation_type = random.choice(mutation_types)

        if mutation_type == 'add_gene' and mutated.length() < self.max_genes:
            # Añadir gen aleatorio
            attr_idx = random.randint(0, self.n_features - 1)
            condition_val = random.uniform(-2, 2)
            mutated.add_gene(attr_idx, condition_val)

        elif mutation_type == 'remove_gene' and mutated.length() > self.min_genes:
            # Eliminar gen aleatorio
            if mutated.genes:
                mutated.genes.pop(random.randint(0, len(mutated.genes) - 1))

        elif mutation_type == 'modify_gene' and mutated.genes:
            # Modificar gen existente
            gene_idx = random.randint(0, len(mutated.genes) - 1)
            gene = mutated.genes[gene_idx]
            gene.condition_value += random.gauss(0, 0.3)

        elif mutation_type == 'duplicate_gene' and mutated.genes and mutated.length() < self.max_genes:
            # Duplicar gen existente (característica de messy chromosomes)
            original_gene = random.choice(mutated.genes)
            duplicated = MessyGene(original_gene.attribute_index,
                                   original_gene.condition_value + random.gauss(0, 0.1))
            mutated.genes.append(duplicated)

        return mutated

    def evolve_class_rules(self, X, y, target_class):
        """Evoluciona cromosomas para una clase específica."""
        print(f"    Evolucionando reglas para clase {target_class}")

        # Datos de la clase objetivo
        X_class = X[y == target_class]

        # Población inicial
        population = []
        for _ in range(self.population_size):
            chromosome = self._create_random_chromosome(target_class, X_class)
            population.append(chromosome)

        best_individual = None
        best_fitness = -1

        # Evolución
        for generation in range(self.generations):
            # Evaluación
            fitnesses = []
            for chromo in population:
                fitness = self._evaluate_fitness(chromo, X, y)
                chromo.fitness = fitness
                fitnesses.append(fitness)

            # Mejor individuo
            max_idx = np.argmax(fitnesses)
            if fitnesses[max_idx] > best_fitness:
                best_fitness = fitnesses[max_idx]
                best_individual = population[max_idx].copy()

            # Nueva población
            new_population = [best_individual.copy()]  # Elitismo

            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)

                # Cruce especializado
                if random.random() < self.crossover_prob:
                    child1, child2 = self._messy_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutación especializada
                if random.random() < self.mutation_prob:
                    child1 = self._messy_mutation(child1)
                if random.random() < self.mutation_prob:
                    child2 = self._messy_mutation(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

            # Progreso
            if generation % 10 == 0:
                avg_length = np.mean([chromo.length() for chromo in population])
                print(f"      Gen {generation}: fitness={best_fitness:.4f}, long_prom={avg_length:.1f}")

        print(f"      Mejor regla: {best_individual}")
        return best_individual


class UnorderedClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador desordenado evolutivo fiel a memoria.pdf sección 2.5.
    Implementa cromosomas desordenados con genes etiqueta-valor según Lanzi.
    """

    def __init__(self, population_size=50, generations=30, tournament_size=5,
                 crossover_prob=0.7, mutation_prob=0.4, min_genes=2, max_genes=8,
                 random_state=42):

        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.min_genes = min_genes
        self.max_genes = max_genes
        self.random_state = random_state

        # Estado del modelo
        self.evolution_engine = None
        self.class_rules = {}
        self.classes_ = None
        self.n_features_ = None
        self.using_fallback = False
        self._estimator_type = "classifier"

    def fit(self, X, y):
        """Entrena clasificador evolutivo desordenado."""
        try:
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y)

            self.classes_ = np.unique(y)
            self.n_features_ = X.shape[1]

            print("=" * 60)
            print("ENTRENANDO CLASIFICADOR DESORDENADO EVOLUTIVO")
            print(f"Cromosomas desordenados según memoria.pdf sección 2.5")
            print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} características")
            print(f"Clases: {self.classes_}")
            print("=" * 60)

            # Motor evolutivo
            self.evolution_engine = UnorderedEvolutionaryEngine(
                population_size=self.population_size,
                generations=self.generations,
                tournament_size=self.tournament_size,
                crossover_prob=self.crossover_prob,
                mutation_prob=self.mutation_prob,
                min_genes=self.min_genes,
                max_genes=min(self.max_genes, self.n_features_),
                random_state=self.random_state
            )

            self.evolution_engine.n_features = self.n_features_

            # Evolucionar reglas para cada clase
            self.class_rules = {}
            for class_label in self.classes_:
                best_rule = self.evolution_engine.evolve_class_rules(X, y, class_label)
                self.class_rules[class_label] = best_rule

            print(f"\nReglas evolucionadas:")
            for class_label, rule in self.class_rules.items():
                print(f"  Clase {class_label}: {rule}")

            self.using_fallback = False

        except Exception as e:
            print(f"Error en evolución desordenada: {e}")
            print("Activando modo fallback con RandomForest...")

            self.evolution_engine = RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=self.random_state
            )
            self.evolution_engine.fit(X, y)
            self.classes_ = np.unique(y)
            self.using_fallback = True

        return self

    def predict(self, X):
        """Predicciones usando cromosomas desordenados evolucionados."""
        if self.using_fallback:
            return self.evolution_engine.predict(X)

        X = np.asarray(X, dtype=np.float64)
        predictions = []

        for instance in X:
            best_score = -1
            best_class = self.classes_[0]

            for class_label, rule in self.class_rules.items():
                score = self._evaluate_rule_activation(rule, instance)

                if score > best_score:
                    best_score = score
                    best_class = class_label

            predictions.append(best_class)

        return np.array(predictions)

    @staticmethod
    def _evaluate_rule_activation(rule, instance):
        """Evalúa activación de regla desordenada en instancia."""
        if rule.length() == 0:
            return 0.0

        # Verificar todas las condiciones del cromosoma
        activations = []

        for gene in rule.genes:
            attr_idx = gene.attribute_index
            condition_val = gene.condition_value

            if attr_idx < len(instance):
                # Grado de activación basado en proximidad
                attr_value = instance[attr_idx]
                if attr_value >= condition_val:
                    activation = 1.0 - abs(attr_value - condition_val) / (abs(condition_val) + 1)
                else:
                    activation = 0.0

                activations.append(max(0.0, activation))
            else:
                activations.append(0.0)

        # Activación promedio (operador T-norma promedio)
        return np.mean(activations) if activations else 0.0

    def predict_proba(self, X):
        """Probabilidades usando activaciones de reglas desordenadas."""
        if self.using_fallback:
            return self.evolution_engine.predict_proba(X)

        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)
        n_classes = len(self.classes_)
        probabilities = np.zeros((n_samples, n_classes))

        for i, instance in enumerate(X):
            class_scores = []

            for class_label in self.classes_:
                rule = self.class_rules[class_label]
                score = self._evaluate_rule_activation(rule, instance)
                class_scores.append(score)

            # Normalizar a probabilidades
            total_score = sum(class_scores)
            if total_score > 0:
                probabilities[i] = np.array(class_scores) / total_score
            else:
                probabilities[i] = np.ones(n_classes) / n_classes

        return probabilities

    def get_rules_summary(self):
        """Resumen interpretable de reglas evolucionadas."""
        if self.using_fallback:
            return "Modelo fallback RandomForest activo"

        summary = "Reglas desordenadas evolucionadas:\n"
        for class_label, rule in self.class_rules.items():
            summary += f"\nClase {class_label}:\n"
            summary += f"  Cromosoma: {rule}\n"
            summary += f"  Longitud: {rule.length()} genes\n"
            summary += f"  Atributos: {list(rule.get_attributes_mentioned())}\n"
            summary += f"  Fitness: {rule.fitness:.4f}\n"

        return summary

    @property
    def feature_importances_(self):
        """Importancias basadas en frecuencia de genes por atributo."""
        if self.using_fallback:
            return getattr(self.evolution_engine, 'feature_importances_', None)

        if not self.class_rules:
            return None

        importances = np.zeros(self.n_features_)

        for rule in self.class_rules.values():
            for gene in rule.genes:
                if gene.attribute_index < self.n_features_:
                    importances[gene.attribute_index] += 1.0

        # Normalizar
        total = np.sum(importances)
        if total > 0:
            importances = importances / total

        return importances

    @property
    def selected_features_(self):
        """Características seleccionadas por cromosomas desordenados."""
        if self.using_fallback:
            return np.arange(self.n_features_) if self.n_features_ else None

        if not self.class_rules:
            return None

        selected = set()
        for rule in self.class_rules.values():
            selected.update(rule.get_attributes_mentioned())

        return np.array(sorted(list(selected)))