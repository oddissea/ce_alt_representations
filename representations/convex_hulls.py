#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convex_hulls.py
COMPUTACIÓN EVOLUTIVA
PEC 4: Representaciones Alternativas de Reglas - VERSIÓN TEÓRICAMENTE PURA
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 4.0 - Fidelidad Teórica Completa

Descripción:
    1. Dimensión variable del genotipo → Cromosomas de longitud fija
    2. Inflación de reglas → Control explícito de bloat con parsimonia
    3. Complejidad evolutiva → Operadores genéticos simples y robustos
    4. Sobredimensionamiento → Fitness que penaliza complejidad excesiva

Autor: Fernando H. Nasser-Eddine López
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, Delaunay
import warnings

warnings.filterwarnings('ignore')


class ConvexHullClassifier(BaseEstimator, ClassifierMixin):
    """
    Clasificador evolutivo de envolturas convexas con algoritmo genético básico.

    Resuelve los problemas documentados en la literatura mediante:
    - Genotipo de longitud fija para operadores genéticos estables
    - Control de parsimonia para evitar inflación de reglas
    - Operadores evolutivos simples y robustos
    - Fitness que balancea precisión y simplicidad
    """

    def __init__(self, population_size=30, generations=20, n_components=4,
                 n_vertices=6, mutation_rate=0.2, crossover_rate=0.7,
                 parsimony_weight=0.1, random_state=42):
        """
        Inicializa el clasificador evolutivo.

        Parámetros optimizados para resolver problemas identificados:
        - n_vertices: Longitud fija del genotipo (resuelve dimensión variable)
        - parsimony_weight: Control de bloat (resuelve inflación de reglas)
        - Parámetros simples: Reducir complejidad evolutiva
        """
        self.population_size = population_size
        self.generations = generations
        self.n_components = n_components
        self.n_vertices = n_vertices  # ← CLAVE: Longitud fija del genotipo
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.parsimony_weight = parsimony_weight  # ← CLAVE: Control de bloat
        self.random_state = random_state

        # Atributos internos
        self.classes_ = None
        self.hulls_ = []
        self.pca = None
        self.X_reduced_ = None
        self.bounds_ = None  # Límites del espacio para generar vértices válidos
        self._estimator_type = "classifier"

        # Configurar semilla
        np.random.seed(self.random_state)

    def fit(self, X, y):
        """
        Entrena el clasificador mediante algoritmo genético básico.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        # Reducción dimensional conservadora
        n_comp = min(self.n_components, X.shape[1], X.shape[0] // 10)
        self.pca = PCA(n_components=n_comp)
        self.X_reduced_ = self.pca.fit_transform(X)

        # Calcular límites del espacio para generar vértices válidos
        self.bounds_ = {
            'min': np.min(self.X_reduced_, axis=0),
            'max': np.max(self.X_reduced_, axis=0),
            'center': np.mean(self.X_reduced_, axis=0),
            'std': np.std(self.X_reduced_, axis=0)
        }

        print(f"Dimensionalidad reducida: {X.shape[1]} → {n_comp} componentes")
        print(f"Evolución con {self.population_size} individuos x {self.generations} generaciones")

        # Evolucionar envoltura para cada clase
        self.hulls_ = []
        for cls in self.classes_:
            X_cls = self.X_reduced_[y == cls]

            if len(X_cls) < 4:  # Muy pocos datos
                hull = self._create_fallback_sphere(X_cls, cls)
            else:
                hull = self._evolve_convex_hull(X_cls, cls)

            self.hulls_.append(hull)

        return self

    def _evolve_convex_hull(self, X_cls, cls):
        """
        Algoritmo genético básico para evolucionar envoltura convexa óptima.

        Soluciona problemas identificados:
        1. Cromosomas de longitud fija (n_vertices puntos)
        2. Fitness con parsimonia para control de bloat
        3. Operadores genéticos simples y robustos
        """
        print(f"  Evolucionando clase {cls}: {len(X_cls)} puntos")

        # 1. Inicialización de población con cromosomas de longitud fija
        population = self._initialize_population(X_cls)

        best_individual = None
        best_fitness = -1
        fitness_history = []

        # 2. Bucle evolutivo principal
        for generation in range(self.generations):
            # Evaluación de fitness con parsimonia
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_fitness_with_parsimony(individual, X_cls)
                fitness_scores.append(fitness)

            # Rastrear mejor individuo
            max_fitness_idx = np.argmax(fitness_scores)
            if fitness_scores[max_fitness_idx] > best_fitness:
                best_fitness = fitness_scores[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()

            fitness_history.append(best_fitness)

            # 3. Selección, cruce y mutación (operadores simples)
            new_population = [best_individual.copy()]

            # Elitismo: mantener mejor individuo

            # Generar resto de la población
            while len(new_population) < self.population_size:
                # Selección por torneo
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                # Cruce simple
                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._simple_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutación simple
                if np.random.random() < self.mutation_rate:
                    child1 = self._simple_mutation(child1, X_cls)
                if np.random.random() < self.mutation_rate:
                    child2 = self._simple_mutation(child2, X_cls)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

            # Progreso cada 5 generaciones
            if generation % 5 == 0:
                print(f"    Gen {generation}: fitness = {best_fitness:.4f}")

        print(f"    Resultado final: fitness = {best_fitness:.4f}")

        # 4. Crear envoltura final
        return self._create_final_hull(best_individual, cls, fitness_history)

    def _initialize_population(self, X_cls):
        """
        Inicializa población con cromosomas de longitud fija.

        Cada cromosoma = n_vertices puntos en el espacio PCA.
        Resuelve el problema de "dimensión variable del genotipo".
        """
        population = []

        for _ in range(self.population_size):
            # Estrategia mixta para diversidad inicial
            if np.random.random() < 0.5:
                # 50%: Seleccionar puntos de datos existentes
                if len(X_cls) >= self.n_vertices:
                    indices = np.random.choice(len(X_cls), self.n_vertices, replace=False)
                    individual = X_cls[indices].copy()
                else:
                    # Si hay pocos datos, repetir con perturbación
                    individual = []
                    for _ in range(self.n_vertices):
                        base_point = X_cls[np.random.randint(len(X_cls))]
                        noise = np.random.normal(0, 0.1, base_point.shape)
                        individual.append(base_point + noise)
                    individual = np.array(individual)
            else:
                # 50%: Generar puntos aleatorios en región válida
                individual = []
                for _ in range(self.n_vertices):
                    point = np.random.uniform(
                        self.bounds_['min'] - 0.5 * self.bounds_['std'],
                        self.bounds_['max'] + 0.5 * self.bounds_['std']
                    )
                    individual.append(point)
                individual = np.array(individual)

            population.append(individual)

        return population

    def _evaluate_fitness_with_parsimony(self, vertices, X_cls):
        """
        Función de fitness que incluye control de parsimonia.

        Fitness = Precisión - parsimony_weight * Complejidad
        Resuelve el problema de "inflación de reglas".
        """
        try:
            # Calcular precisión: cobertura de puntos de la clase
            coverage = self._calculate_coverage(vertices, X_cls)

            # Calcular complejidad: dispersión de vértices
            complexity = self._calculate_complexity(vertices)

            # Fitness con parsimonia
            fitness = coverage - self.parsimony_weight * complexity

            return max(0.0, fitness)  # Asegurar fitness no negativo

        except (ValueError, RuntimeError, AttributeError):
            return 0.0

    def _calculate_coverage(self, vertices, X_cls):
        """
        Calcula qué proporción de puntos de clase están cubiertos por la envoltura.
        """
        try:
            # Crear envoltura convexa
            hull = ConvexHull(vertices)
            delaunay = Delaunay(vertices[hull.vertices])

            # Contar puntos dentro
            inside_count = 0
            for point in X_cls:
                if delaunay.find_simplex(point) >= 0:
                    inside_count += 1

            return inside_count / len(X_cls)

        except (ValueError, RuntimeError, AttributeError):
            # Fallback: distancia promedio a vértices más cercanos
            min_distances = []
            for point in X_cls:
                distances = [np.linalg.norm(point - v) for v in vertices]
                min_distances.append(min(distances))

            # Convertir distancias a score de cobertura
            avg_distance = np.mean(min_distances)
            max_distance = np.max([np.linalg.norm(self.bounds_['max'] - self.bounds_['min'])])
            coverage = max(0, 1 - (avg_distance / max_distance))

            return coverage

    def _calculate_complexity(self, vertices):
        """
        Calcula complejidad como dispersión de vértices.
        """
        # Complejidad = dispersión promedio entre vértices
        if len(vertices) < 2:
            return 0.0

        center = np.mean(vertices, axis=0)
        distances = [np.linalg.norm(v - center) for v in vertices]

        # Normalizar por tamaño del espacio
        space_size = np.linalg.norm(self.bounds_['max'] - self.bounds_['min'])
        complexity = np.std(distances) / (space_size + 1e-6)

        return complexity

    @staticmethod
    def _tournament_selection(population, fitness_scores, tournament_size=3):
        """
        Selección por torneo simple.
        """
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]

        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()

    def _simple_crossover(self, parent1, parent2):
        """
        Cruce simple: intercambio de vértices.

        Operador robusto que mantiene longitud fija del cromosoma.
        """
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Intercambiar vértices aleatoriamente
        for i in range(self.n_vertices):
            if np.random.random() < 0.5:
                child1[i], child2[i] = child2[i].copy(), child1[i].copy()

        return child1, child2

    def _simple_mutation(self, individual, X_cls):
        """
        Mutación simple: perturbación de vértices.

        Operador robusto que mantiene vértices en región válida.
        """
        mutated = individual.copy()

        for i in range(self.n_vertices):
            if np.random.random() < 0.3:  # 30% probabilidad por vértice

                mutation_type = np.random.choice(['perturb', 'replace'])

                if mutation_type == 'perturb':
                    # Perturbación gaussiana
                    noise = np.random.normal(0, 0.2 * self.bounds_['std'])
                    mutated[i] += noise

                    # Mantener en límites válidos
                    mutated[i] = np.clip(mutated[i],
                                         self.bounds_['min'] - self.bounds_['std'],
                                         self.bounds_['max'] + self.bounds_['std'])

                elif mutation_type == 'replace':
                    # Reemplazar con punto de datos existente
                    if len(X_cls) > 0:
                        mutated[i] = X_cls[np.random.randint(len(X_cls))].copy()

        return mutated

    def _create_final_hull(self, best_vertices, cls, fitness_history):
        """
        Crea estructura final de envoltura convexa.
        """
        try:
            hull = ConvexHull(best_vertices)
            delaunay = Delaunay(best_vertices[hull.vertices])

            return {
                'class': cls,
                'vertices': best_vertices,
                'hull': hull,
                'delaunay': delaunay,
                'method': 'evolved_convex',
                'fitness_history': fitness_history
            }

        except (ValueError, RuntimeError, AttributeError):
            # Fallback: envoltura esférica
            return self._create_fallback_sphere(best_vertices, cls)

    def _create_fallback_sphere(self, points, cls):
        """
        Crea envoltura esférica como fallback robusto.
        """
        if len(points) == 0:
            center = self.bounds_['center']
            radius = np.mean(self.bounds_['std'])
        else:
            center = np.mean(points, axis=0)
            if len(points) == 1:
                radius = np.mean(self.bounds_['std']) * 0.5
            else:
                distances = [np.linalg.norm(p - center) for p in points]
                radius = np.max(distances) * 1.3

        return {
            'class': cls,
            'center': center,
            'radius': radius,
            'method': 'sphere_fallback'
        }

    def predict(self, X):
        """
        Realiza predicciones usando las envolturas evolucionadas.
        """
        X = np.asarray(X, dtype=np.float64)
        X_reduced = self.pca.transform(X)

        predictions = []
        for x in X_reduced:
            pred_class = self._predict_single(x)
            predictions.append(pred_class)

        return np.array(predictions)

    def _predict_single(self, x):
        """
        Predice clase para una instancia individual.
        """
        best_score = -1
        best_class = self.classes_[0]

        for hull_dict in self.hulls_:
            score = 0.0

            if hull_dict['method'] == 'evolved_convex':
                try:
                    delaunay = hull_dict['delaunay']
                    if delaunay.find_simplex(x) >= 0:
                        # Punto dentro de la envoltura
                        vertices = hull_dict['vertices']
                        centroid = np.mean(vertices, axis=0)
                        distance = np.linalg.norm(x - centroid)
                        score = 1.0 / (1.0 + distance)
                except (ValueError, RuntimeError, AttributeError):
                    score = 0.0

            elif hull_dict['method'] == 'sphere_fallback':
                distance = np.linalg.norm(x - hull_dict['center'])
                if distance <= hull_dict['radius']:
                    score = 1.0 - (distance / hull_dict['radius'])

            if score > best_score:
                best_score = score
                best_class = hull_dict['class']

        return best_class

    def predict_proba(self, X):
        """
        Estima probabilidades de clase.
        """
        X = np.asarray(X, dtype=np.float64)
        X_reduced = self.pca.transform(X)

        n_samples = len(X_reduced)
        n_classes = len(self.classes_)
        probabilities = np.zeros((n_samples, n_classes))

        for i, x in enumerate(X_reduced):
            scores = [self._calculate_hull_score(hull_dict, x) for hull_dict in self.hulls_]

            # Normalizar probabilidades
            total_score = sum(scores)
            if total_score > 0:
                for j, cls in enumerate(self.classes_):
                    class_idx = np.where(self.classes_ == cls)[0][0]
                    probabilities[i, class_idx] = scores[j] / total_score
            else:
                # Distribución uniforme si no hay scores
                probabilities[i, :] = 1.0 / n_classes

        return probabilities

    @staticmethod
    def _calculate_hull_score(hull_dict, x):
        """Calcula score para un hull específico."""
        if hull_dict['method'] == 'evolved_convex':
            try:
                delaunay = hull_dict['delaunay']
                if delaunay.find_simplex(x) >= 0:
                    vertices = hull_dict['vertices']
                    centroid = np.mean(vertices, axis=0)
                    distance = np.linalg.norm(x - centroid)
                    return 1.0 / (1.0 + distance)
                else:
                    return 0.0
            except (ValueError, RuntimeError, AttributeError):
                return 0.0

        elif hull_dict['method'] == 'sphere_fallback':
            distance = np.linalg.norm(x - hull_dict['center'])
            return max(0.0, 1.0 - (distance / hull_dict['radius']))

        else:
            return 0.0

    # Métodos de compatibilidad sklearn

    def get_params(self, deep=True):
        """Obtiene parámetros del estimador."""
        return {
            'population_size': self.population_size,
            'generations': self.generations,
            'n_components': self.n_components,
            'n_vertices': self.n_vertices,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'parsimony_weight': self.parsimony_weight,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        """Establece parámetros del estimador."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
        return self

    # Métodos de robustez para intelligent_fallback_evaluation

    def simplify_for_fallback(self):
        """Crea versión simplificada para fallback."""
        simplified = ConvexHullClassifier(
            population_size=15,
            generations=10,
            n_components=3,
            n_vertices=4,
            random_state=self.random_state
        )
        return simplified

    def increase_regularization(self):
        """Aumenta regularización."""
        self.parsimony_weight = min(0.3, self.parsimony_weight * 1.5)
        self.n_vertices = max(4, self.n_vertices - 1)

    def reduce_complexity(self):
        """Reduce complejidad del modelo."""
        self.population_size = min(20, self.population_size)
        self.generations = min(15, self.generations)
        self.n_components = min(3, self.n_components)