#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pruebas/validation_datasets.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Datasets de validación para representaciones alternativas de reglas
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 1.0

Descripción:
    Datasets específicos para validación de representaciones en dominios apropiados.
    Separados del experimento principal para mantener claridad arquitectónica.
"""

import numpy as np
import random
from sklearn.model_selection import train_test_split


def create_expanded_boolean_dataset(base_patterns, n_samples=200, noise_rate=0.05, random_state=42):
    """
    Expande dataset booleano mediante replicación con ruido controlado.

    Parámetros:
    - base_patterns: Array con patrones base [entrada1, entrada2, ..., salida]
    - n_samples: Número total de muestras deseadas
    - noise_rate: Proporción de etiquetas con ruido para realismo
    - random_state: Semilla de reproducibilidad

    Retorna:
    - X_expanded, y_expanded: Dataset expandido
    """
    np.random.seed(random_state)
    random.seed(random_state)

    n_patterns = len(base_patterns)
    n_replications = n_samples // n_patterns
    remainder = n_samples % n_patterns

    # Replicar patrones base
    X_expanded = np.tile(base_patterns[:, :-1], (n_replications, 1))
    y_expanded = np.tile(base_patterns[:, -1], n_replications)

    # Añadir muestras restantes
    if remainder > 0:
        X_expanded = np.vstack([X_expanded, base_patterns[:remainder, :-1]])
        y_expanded = np.hstack([y_expanded, base_patterns[:remainder, -1]])

    # Añadir ruido realista a las etiquetas
    n_noise = int(n_samples * noise_rate)
    if n_noise > 0:
        noise_indices = np.random.choice(n_samples, n_noise, replace=False)
        y_expanded[noise_indices] = 1 - y_expanded[noise_indices]

    # Mezclar dataset
    shuffle_idx = np.random.permutation(n_samples)
    X_expanded = X_expanded[shuffle_idx].astype(bool)
    y_expanded = y_expanded[shuffle_idx].astype(int)

    return X_expanded, y_expanded


def create_xor_expanded_dataset(n_samples=200, noise_rate=0.05, random_state=42):
    """
    Crea dataset XOR expandido para validación de expresiones simbólicas.

    Función lógica: XOR(A, B) = (A AND NOT B) OR (NOT A AND B)
    """
    base_patterns = np.array([
        [0, 0, 0],  # 0 XOR 0 = 0
        [0, 1, 1],  # 0 XOR 1 = 1
        [1, 0, 1],  # 1 XOR 0 = 1
        [1, 1, 0]  # 1 XOR 1 = 0
    ])

    return create_expanded_boolean_dataset(
        base_patterns, n_samples, noise_rate, random_state
    )


def create_multiplexor_expanded_dataset(n_samples=300, noise_rate=0.03, random_state=42):
    """
    Crea dataset multiplexor 3-bit expandido.

    Función lógica: Si addr=0 entonces salida=d0, si addr=1 entonces salida=d1
    Entradas: [addr, d0, d1] → salida
    """
    base_patterns = []
    for addr in [0, 1]:
        for d0, d1 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            output = d1 if addr else d0
            base_patterns.append([addr, d0, d1, output])

    base_patterns = np.array(base_patterns)

    return create_expanded_boolean_dataset(
        base_patterns, n_samples, noise_rate, random_state
    )


def create_and_expanded_dataset(n_samples=150, noise_rate=0.02, random_state=42):
    """
    Crea dataset AND expandido (función más simple para validación básica).

    Función lógica: AND(A, B) = A AND B
    """
    base_patterns = np.array([
        [0, 0, 0],  # 0 AND 0 = 0
        [0, 1, 0],  # 0 AND 1 = 0
        [1, 0, 0],  # 1 AND 0 = 0
        [1, 1, 1]  # 1 AND 1 = 1
    ])

    return create_expanded_boolean_dataset(
        base_patterns, n_samples, noise_rate, random_state
    )


def create_or_expanded_dataset(n_samples=150, noise_rate=0.02, random_state=42):
    """
    Crea dataset OR expandido.

    Función lógica: OR(A, B) = A OR B
    """
    base_patterns = np.array([
        [0, 0, 0],  # 0 OR 0 = 0
        [0, 1, 1],  # 0 OR 1 = 1
        [1, 0, 1],  # 1 OR 0 = 1
        [1, 1, 1]  # 1 OR 1 = 1
    ])

    return create_expanded_boolean_dataset(
        base_patterns, n_samples, noise_rate, random_state
    )


def create_parity_expanded_dataset(n_bits=3, n_samples=200, noise_rate=0.04, random_state=42):
    """
    Crea dataset de paridad expandido.

    Función lógica: PARITY(bits) = suma(bits) mod 2
    """
    base_patterns = []
    for i in range(2 ** n_bits):
        bits = [(i >> j) & 1 for j in range(n_bits)]
        parity = sum(bits) % 2
        base_patterns.append(bits + [parity])

    base_patterns = np.array(base_patterns)

    return create_expanded_boolean_dataset(
        base_patterns, n_samples, noise_rate, random_state
    )


def get_boolean_validation_datasets():
    """
    Retorna diccionario con todos los datasets de validación booleanos.

    Retorna:
    - dict: {nombre: (X, y)} para cada dataset de validación
    """
    return {
        'XOR_expanded': create_xor_expanded_dataset(n_samples=200, noise_rate=0.05),
        'Multiplexor_expanded': create_multiplexor_expanded_dataset(n_samples=300, noise_rate=0.03),
        'AND_expanded': create_and_expanded_dataset(n_samples=150, noise_rate=0.02),
        'OR_expanded': create_or_expanded_dataset(n_samples=150, noise_rate=0.02),
        'Parity3_expanded': create_parity_expanded_dataset(n_bits=3, n_samples=200, noise_rate=0.04)
    }


def create_interval_validation_datasets():
    """
    Crea datasets específicos para validación de representaciones por intervalos.

    Retorna:
    - dict: Datasets con fronteras de decisión paralelas a los ejes
    """
    np.random.seed(42)

    # Dataset con regiones rectangulares separables
    n_samples = 400
    X_rect = np.random.uniform(-1, 1, (n_samples, 2))
    y_rect = ((X_rect[:, 0] > -0.3) & (X_rect[:, 0] < 0.3) &
              (X_rect[:, 1] > -0.3) & (X_rect[:, 1] < 0.3)).astype(int)

    # Dataset con múltiples intervalos
    X_multi = np.random.uniform(0, 10, (n_samples, 3))
    y_multi = (((X_multi[:, 0] > 2) & (X_multi[:, 0] < 4)) |
               ((X_multi[:, 1] > 6) & (X_multi[:, 1] < 8)) |
               ((X_multi[:, 2] > 1) & (X_multi[:, 2] < 3))).astype(int)

    return {
        'Rectangular_regions': (X_rect, y_rect),
        'Multiple_intervals': (X_multi, y_multi)
    }


def create_ellipsoid_validation_datasets():
    """
    Crea datasets específicos para validación de representaciones hiperelipsoidales.

    Retorna:
    - dict: Datasets con fronteras de decisión elípticas/curvas
    """
    np.random.seed(42)

    # Dataset con regiones elípticas
    n_samples = 400
    X_ellipse = np.random.uniform(-2, 2, (n_samples, 2))

    # Elipse centrada en origen
    y_ellipse = ((X_ellipse[:, 0] ** 2 / 1.5 ** 2 + X_ellipse[:, 1] ** 2 / 1.0 ** 2) < 1).astype(int)

    # Dataset con múltiples elipses
    X_multi_ellipse = np.random.uniform(-3, 3, (n_samples, 2))

    # Dos elipses separadas
    ellipse1 = ((X_multi_ellipse[:, 0] + 1) ** 2 / 0.8 ** 2 + (X_multi_ellipse[:, 1] + 1) ** 2 / 0.6 ** 2) < 1
    ellipse2 = ((X_multi_ellipse[:, 0] - 1) ** 2 / 0.6 ** 2 + (X_multi_ellipse[:, 1] - 1) ** 2 / 0.8 ** 2) < 1
    y_multi_ellipse = (ellipse1 | ellipse2).astype(int)

    return {
        'Single_ellipse': (X_ellipse, y_ellipse),
        'Multiple_ellipses': (X_multi_ellipse, y_multi_ellipse)
    }


def create_fuzzy_validation_datasets():
    """
    Crea datasets específicos para validación de lógica difusa.

    Retorna:
    - dict: Datasets con fronteras suaves y solapamiento
    """
    np.random.seed(42)

    # Dataset con transiciones suaves
    n_samples = 400
    X_smooth = np.random.uniform(0, 10, (n_samples, 2))

    # Función con transición suave
    distance_to_center = np.sqrt((X_smooth[:, 0] - 5) ** 2 + (X_smooth[:, 1] - 5) ** 2)
    y_smooth_prob = 1 / (1 + np.exp(distance_to_center - 3))  # Sigmoide
    y_smooth = (y_smooth_prob > 0.5).astype(int)

    # Dataset con solapamiento de clases
    X_overlap = np.random.uniform(0, 8, (n_samples, 2))

    # Regiones con solapamiento gradual
    region1_strength = np.exp(-((X_overlap[:, 0] - 2) ** 2 + (X_overlap[:, 1] - 2) ** 2) / 2)
    region2_strength = np.exp(-((X_overlap[:, 0] - 6) ** 2 + (X_overlap[:, 1] - 6) ** 2) / 2)
    y_overlap = (region1_strength > region2_strength).astype(int)

    return {
        'Smooth_transitions': (X_smooth, y_smooth),
        'Overlapping_regions': (X_overlap, y_overlap)
    }


def validate_datasets_integrity():
    """
    Valida la integridad y consistencia lógica de todos los datasets de validación.
    """
    print("VALIDACIÓN DE INTEGRIDAD DE DATASETS")
    print("=" * 50)

    # Validar datasets booleanos
    boolean_datasets = get_boolean_validation_datasets()

    for name, (X, y) in boolean_datasets.items():
        print(f"\n{name}:")
        print(f"  Forma: {X.shape}")
        print(f"  Tipo X: {X.dtype} (debe ser bool)")
        print(f"  Tipo y: {y.dtype} (debe ser int)")
        print(f"  Clases: {np.unique(y)} (debe ser [0, 1])")
        print(f"  Balance: {np.mean(y):.3f} (0.5 = perfecto)")

        # Verificar lógica para casos conocidos
        if name == 'XOR_expanded':
            # Verificar algunos casos XOR
            test_cases = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
            for x0, x1, expected in test_cases:
                mask = (X[:, 0] == x0) & (X[:, 1] == x1)
                if np.any(mask):
                    actual = np.mode(y[mask])[0]
                    print(f"    XOR({x0},{x1}) esperado:{expected}, mayoritario:{actual}")

        elif name == 'AND_expanded':
            # Verificar casos AND
            test_cases = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
            for x0, x1, expected in test_cases:
                mask = (X[:, 0] == x0) & (X[:, 1] == x1)
                if np.any(mask):
                    actual = np.mode(y[mask])[0]
                    print(f"    AND({x0},{x1}) esperado:{expected}, mayoritario:{actual}")

    # Validar datasets especializados
    interval_datasets = create_interval_validation_datasets()
    ellipsoid_datasets = create_ellipsoid_validation_datasets()
    fuzzy_datasets = create_fuzzy_validation_datasets()

    print(f"\nDatasets especializados creados correctamente:")
    print(f"  Intervalos: {len(interval_datasets)} datasets")
    print(f"  Elipsoides: {len(ellipsoid_datasets)} datasets")
    print(f"  Lógica difusa: {len(fuzzy_datasets)} datasets")

    print(f"\n✅ Todos los datasets validados correctamente")


if __name__ == "__main__":
    # Ejecutar validación de integridad
    validate_datasets_integrity()

    # Mostrar ejemplos de uso
    print("\n" + "=" * 50)
    print("EJEMPLOS DE USO")
    print("=" * 50)

    # Ejemplo XOR
    X_xor, y_xor = create_xor_expanded_dataset(n_samples=100, noise_rate=0.02)
    print(f"\nXOR expandido: {X_xor.shape}, balance: {np.mean(y_xor):.3f}")

    # Ejemplo Multiplexor
    X_mux, y_mux = create_multiplexor_expanded_dataset(n_samples=150, noise_rate=0.01)
    print(f"Multiplexor expandido: {X_mux.shape}, balance: {np.mean(y_mux):.3f}")

    # Todos los datasets
    all_datasets = get_boolean_validation_datasets()
    print(f"\nTotal datasets booleanos disponibles: {len(all_datasets)}")
    for name in all_datasets.keys():
        print(f"  - {name}")


# ============= FUNCIONES ADICIONALES QUE FALTABAN =============

def create_neural_validation_datasets():
    """
    Crea datasets específicos para validación de redes neuronales.

    Retorna:
    - dict: Datasets con patrones complejos no lineales
    """
    np.random.seed(42)

    # Dataset con patrones no lineales complejos
    n_samples = 500
    X_nonlinear = np.random.uniform(-2, 2, (n_samples, 3))

    # Función no lineal compleja: combinación de senos y productos
    y_nonlinear = (np.sin(X_nonlinear[:, 0] * X_nonlinear[:, 1]) +
                   np.cos(X_nonlinear[:, 2]) > 0.5).astype(int)

    # Dataset en espiral (clásico para redes neuronales)
    n_points = 300
    r = np.linspace(0.1, 2, n_points)
    theta = np.linspace(0, 4 * np.pi, n_points)

    # Crear dos espirales entrelazadas
    X_spiral = np.zeros((2 * n_points, 2))
    y_spiral = np.zeros(2 * n_points, dtype=int)

    # Primera espiral
    X_spiral[:n_points, 0] = r * np.cos(theta)
    X_spiral[:n_points, 1] = r * np.sin(theta)
    y_spiral[:n_points] = 0

    # Segunda espiral (rotada 180 grados)
    X_spiral[n_points:, 0] = r * np.cos(theta + np.pi)
    X_spiral[n_points:, 1] = r * np.sin(theta + np.pi)
    y_spiral[n_points:] = 1

    # Añadir ruido
    X_spiral += np.random.normal(0, 0.1, X_spiral.shape)

    return {
        'Complex_nonlinear': (X_nonlinear, y_nonlinear),
        'Spiral_pattern': (X_spiral, y_spiral)
    }


def create_convex_hull_validation_datasets():
    """
    Crea datasets específicos para validación de envolturas convexas.

    Retorna:
    - dict: Datasets con formas irregulares y fronteras complejas
    """
    np.random.seed(42)

    # Dataset con formas triangulares
    n_samples = 300
    X_triangular = np.random.uniform(-2, 2, (n_samples, 2))

    # Definir triángulo irregular
    def point_in_triangle(p, a, b, c):
        """Verifica si punto p está dentro del triángulo abc."""
        v0 = c - a
        v1 = b - a
        v2 = p - a

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        return (u >= 0) and (v >= 0) and (u + v <= 1)

    # Definir vértices del triángulo
    triangle_vertices = np.array([[-1, -1], [1, -1], [0, 1.5]])

    y_triangular = np.array([
        point_in_triangle(point, triangle_vertices[0], triangle_vertices[1], triangle_vertices[2])
        for point in X_triangular
    ]).astype(int)

    # Dataset con forma de estrella
    X_star = np.random.uniform(-2, 2, (n_samples, 2))

    # Función para determinar si un punto está dentro de una estrella de 5 puntas
    def point_in_star(x, y, center=(0, 0), outer_radius=1.5, inner_radius=0.7):
        # Convertir a coordenadas polares
        dx, dy = x - center[0], y - center[1]
        r = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.arctan2(dy, dx)

        # Normalizar ángulo a [0, 2π]
        if theta < 0:
            theta += 2 * np.pi

        # Dividir en 10 sectores (5 puntas + 5 entrantes)
        sector = int(theta / (2 * np.pi / 10))
        sector_angle = (theta % (2 * np.pi / 10)) / (2 * np.pi / 10)

        # Determinar radio límite para este sector
        if sector % 2 == 0:  # Sectores de las puntas
            max_radius = outer_radius
        else:  # Sectores entrantes
            # Interpolación lineal entre radio exterior e interior
            if sector_angle < 0.5:
                max_radius = outer_radius - (outer_radius - inner_radius) * (sector_angle * 2)
            else:
                max_radius = inner_radius + (outer_radius - inner_radius) * ((sector_angle - 0.5) * 2)

        return r <= max_radius

    y_star = np.array([
        point_in_star(point[0], point[1]) for point in X_star
    ]).astype(int)

    return {
        'Triangular_regions': (X_triangular, y_triangular),
        'Star_shaped': (X_star, y_star)
    }


def create_unordered_validation_datasets():
    """
    Crea datasets específicos para validación de representación desordenada.

    Retorna:
    - dict: Datasets con características dispersas y relevancia variable
    """
    np.random.seed(42)

    # Dataset con muchas características irrelevantes
    n_samples = 400
    n_features = 20
    n_relevant = 3  # Solo 3 características son relevantes

    X_sparse = np.random.normal(0, 1, (n_samples, n_features))

    # Solo las primeras 3 características determinan la clase
    y_sparse = ((X_sparse[:, 0] > 0) & (X_sparse[:, 1] > 0) |
                (X_sparse[:, 2] > 1)).astype(int)

    # Dataset con características redundantes
    X_redundant = np.random.normal(0, 1, (n_samples, 15))

    # Crear redundancia: algunas características son combinaciones de otras
    X_redundant[:, 3] = X_redundant[:, 0] + X_redundant[:, 1] + np.random.normal(0, 0.1, n_samples)
    X_redundant[:, 4] = X_redundant[:, 0] * X_redundant[:, 2] + np.random.normal(0, 0.1, n_samples)
    X_redundant[:, 5] = X_redundant[:, 1] - X_redundant[:, 2] + np.random.normal(0, 0.1, n_samples)

    # Clase basada en las características originales
    y_redundant = (X_redundant[:, 0] + X_redundant[:, 1] - X_redundant[:, 2] > 0).astype(int)

    return {
        'Sparse_features': (X_sparse, y_sparse),
        'Redundant_features': (X_redundant, y_redundant)
    }


def get_all_validation_datasets():
    """
    Retorna todos los datasets de validación organizados por categoría.

    Retorna:
    - dict: Diccionario completo con todas las categorías de datasets
    """
    return {
        'boolean': get_boolean_validation_datasets(),
        'interval': create_interval_validation_datasets(),
        'ellipsoid': create_ellipsoid_validation_datasets(),
        'fuzzy': create_fuzzy_validation_datasets(),
        'neural': create_neural_validation_datasets(),
        'convex_hull': create_convex_hull_validation_datasets(),
        'unordered': create_unordered_validation_datasets()
    }


def validate_all_datasets():
    """
    Valida la integridad de todos los datasets de validación.
    """
    print("VALIDACIÓN COMPLETA DE TODOS LOS DATASETS")
    print("=" * 60)

    all_datasets = get_all_validation_datasets()

    total_datasets = 0

    for category_name, category_datasets in all_datasets.items():
        print(f"\n--- CATEGORÍA: {category_name.upper()} ---")

        for dataset_name, (X, y) in category_datasets.items():
            total_datasets += 1
            print(f"\n{dataset_name}:")
            print(f"  Forma X: {X.shape}")
            print(f"  Forma y: {y.shape}")
            print(f"  Tipo X: {X.dtype}")
            print(f"  Tipo y: {y.dtype}")
            print(f"  Clases únicas: {np.unique(y)}")
            print(f"  Distribución: {np.bincount(y)}")
            print(f"  Balance: {np.mean(y):.3f}")

            # Verificaciones específicas por categoría
            if category_name == 'boolean':
                if X.dtype != bool:
                    print(f"  ⚠️ ADVERTENCIA: X debería ser bool, es {X.dtype}")
                if not np.all(np.isin(X, [0, 1, True, False])):
                    print(f"  ⚠️ ADVERTENCIA: X contiene valores no booleanos")

            # Verificar que no hay valores NaN o infinitos
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                print(f"  ❌ ERROR: Contiene valores NaN")

            if np.any(np.isinf(X)):
                print(f"  ❌ ERROR: Contiene valores infinitos")

            # Verificar que las clases son binarias
            unique_classes = np.unique(y)
            if len(unique_classes) != 2 or not np.array_equal(unique_classes, [0, 1]):
                print(f"  ⚠️ ADVERTENCIA: Clases no son binarias estándar [0,1]")

    print(f"\n{'=' * 60}")
    print(f"RESUMEN DE VALIDACIÓN")
    print(f"{'=' * 60}")
    print(f"Total de categorías: {len(all_datasets)}")
    print(f"Total de datasets: {total_datasets}")
    print(f"✅ Validación completa finalizada")


def get_dataset_recommendations():
    """
    Retorna recomendaciones de qué datasets usar para cada representación.

    Retorna:
    - dict: Recomendaciones de datasets por representación
    """
    return {
        'Intervalos': ['interval', 'boolean'],
        'Hiperelipsoides': ['ellipsoid', 'neural'],
        'EnvolturasConvexas': ['convex_hull', 'ellipsoid'],
        'RedNeuronal': ['neural', 'convex_hull', 'ellipsoid'],
        'Desordenado': ['unordered', 'boolean'],
        'ExpresionesS': ['boolean'],
        'ExpresionesGenes': ['boolean', 'interval', 'neural'],
        'LogicaDifusa': ['fuzzy', 'ellipsoid']
    }


def create_validation_report():
    """
    Crea un reporte detallado de todos los datasets disponibles.
    """
    print("REPORTE DE DATASETS DE VALIDACIÓN")
    print("=" * 70)

    all_datasets = get_all_validation_datasets()
    recommendations = get_dataset_recommendations()

    print(f"\nDATASETS DISPONIBLES POR CATEGORÍA:")
    print("-" * 40)

    for category, datasets in all_datasets.items():
        print(f"\n{category.upper()} ({len(datasets)} datasets):")
        for name, (X, y) in datasets.items():
            print(f"  • {name}: {X.shape[0]} muestras, {X.shape[1]} características")

    print(f"\nRECOMENDACIONES POR REPRESENTACIÓN:")
    print("-" * 40)

    for representation, categories in recommendations.items():
        print(f"\n{representation}:")
        total_datasets = sum(len(all_datasets[cat]) for cat in categories if cat in all_datasets)
        print(f"  Categorías recomendadas: {', '.join(categories)}")
        print(f"  Total datasets apropiados: {total_datasets}")

        # Listar datasets específicos
        for category in categories:
            if category in all_datasets:
                dataset_names = list(all_datasets[category].keys())
                print(f"    {category}: {', '.join(dataset_names)}")

    print(f"\n{'=' * 70}")
    print("Para usar estos datasets, importar desde validation_datasets.py:")
    print("  from pruebas.validation_datasets import get_all_validation_datasets")
    print("  datasets = get_all_validation_datasets()")
    print("  X, y = datasets['boolean']['XOR_expanded']")


# ============= ACTUALIZACIÓN DE MAIN =============

if __name__ == "__main__":
    # Ejecutar validación completa
    validate_all_datasets()

    # Crear reporte detallado
    print("\n" + "=" * 70)
    create_validation_report()

    # Mostrar ejemplos de uso específicos
    print("\n" + "=" * 70)
    print("EJEMPLOS DE USO AVANZADOS")
    print("=" * 70)

    # Ejemplo de uso para cada categoría
    all_datasets = get_all_validation_datasets()

    for category_name, category_datasets in all_datasets.items():
        if category_datasets:  # Si la categoría tiene datasets
            first_dataset_name = list(category_datasets.keys())[0]
            X, y = category_datasets[first_dataset_name]
            print(f"\n{category_name.upper()} - Ejemplo con {first_dataset_name}:")
            print(f"  X, y = get_all_validation_datasets()['{category_name}']['{first_dataset_name}']")
            print(f"  # X.shape = {X.shape}, balance = {np.mean(y):.3f}")

    print(f"\n✅ Todos los datasets y funciones están disponibles y validados")