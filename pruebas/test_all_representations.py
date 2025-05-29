#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pruebas/test_all_representations.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Validación integral de todas las representaciones alternativas
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 1.0

Descripción:
    Suite completa de validación para todas las representaciones alternativas
    de reglas en sistemas clasificadores evolutivos, con datasets apropiados
    para cada tipo de representación.
"""

import os
import numpy as np
import time
from sklearn.metrics import accuracy_score
import json

from pruebas.validation_datasets import (
    get_boolean_validation_datasets,
    create_interval_validation_datasets,
    create_ellipsoid_validation_datasets,
    create_fuzzy_validation_datasets
)

# Importar todas las representaciones
from representations.intervals import IntervalClassifier
from representations.hyperellipsoids import HyperEllipsoidClassifier
from representations.convex_hulls import ConvexHullClassifier
from representations.neural_network import NeuralNetworkClassifier
from representations.unordered import UnorderedClassifier
from representations.expressions_s import PureExpressionsSClassifier
from representations.gene_expressions import GeneExpressionsClassifier
from representations.fuzzy_logic import FuzzyLogicClassifier


def get_representation_configs():
    """
    Configuraciones optimizadas para cada representación.

    Retorna:
    - dict: Configuración por representación
    """
    return {
        'Intervalos': {
            'classifier': IntervalClassifier,
            'params': {},
            'appropriate_datasets': ['interval', 'simple_classification']
        },
        'Hiperelipsoides': {
            'classifier': HyperEllipsoidClassifier,
            'params': {},
            'appropriate_datasets': ['ellipsoid', 'curved_boundaries']
        },
        'EnvolturasConvexas': {
            'classifier': ConvexHullClassifier,
            'params': {'n_components': 5, 'parsimony_weight': 0.02},
            'appropriate_datasets': ['irregular_shapes', 'complex_boundaries']
        },
        'RedNeuronal': {
            'classifier': NeuralNetworkClassifier,
            'params': {'hidden_layer_sizes': (15, 10), 'max_iter': 150},
            'appropriate_datasets': ['nonlinear', 'high_dimensional']
        },
        'Desordenado': {
            'classifier': UnorderedClassifier,
            'params': {'max_depth': 4},
            'appropriate_datasets': ['feature_selection', 'sparse_features']
        },
        'ExpresionesS': {
            'classifier': PureExpressionsSClassifier,
            'params': {'population_size': 40, 'generations': 60, 'max_depth': 5},
            'appropriate_datasets': ['boolean', 'logical_functions']
        },
        'ExpresionesGenes': {
            'classifier': GeneExpressionsClassifier,
            'params': {'population_size': 100, 'generations': 25, 'chromosome_length': 30},
            'appropriate_datasets': ['symbolic_regression', 'mathematical_functions']
        },
        'LogicaDifusa': {
            'classifier': FuzzyLogicClassifier,
            'params': {'n_fuzzy_sets': 3},
            'appropriate_datasets': ['fuzzy', 'overlapping_classes', 'uncertainty']
        }
    }


def prepare_validation_datasets():
    """
    Prepara todos los datasets de validación organizados por tipo.

    Retorna:
    - dict: Datasets organizados por categoría
    """
    datasets = {}

    # Datasets booleanos (apropiados para Expresiones S)
    datasets['boolean'] = get_boolean_validation_datasets()

    # Datasets de intervalos (apropiados para representación por intervalos)
    datasets['interval'] = create_interval_validation_datasets()

    # Datasets elipsoidales (apropiados para hiperelipsoides)
    datasets['ellipsoid'] = create_ellipsoid_validation_datasets()

    # Datasets difusos (apropiados para lógica difusa)
    datasets['fuzzy'] = create_fuzzy_validation_datasets()

    # Datasets sintéticos adicionales
    from sklearn.datasets import make_classification, make_blobs

    # Dataset linealmente separable (apropiado para intervalos)
    X_linear, y_linear = make_classification(
        n_samples=300, n_features=4, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, random_state=42
    )
    datasets['simple_classification'] = {'Linear_separable': (X_linear, y_linear)}

    # Dataset con múltiples clústeres (apropiado para hiperelipsoides)
    X_clusters, y_clusters = make_blobs(
        n_samples=400, centers=4, n_features=3, random_state=42
    )
    y_clusters = (y_clusters % 2)  # Convertir a binario
    datasets['curved_boundaries'] = {'Multiple_clusters': (X_clusters, y_clusters)}

    # Dataset con alta dimensionalidad (apropiado para desordenado/neuronal)
    X_high_dim, y_high_dim = make_classification(
        n_samples=250, n_features=15, n_informative=8, n_redundant=3,
        random_state=42
    )
    datasets['high_dimensional'] = {'High_dimensional': (X_high_dim, y_high_dim)}

    return datasets


def test_single_representation(representation_name, config, datasets_dict, selected_categories=None):
    """
    Prueba una representación específica en datasets apropiados.

    Parámetros:
    - representation_name: Nombre de la representación
    - config: Configuración de la representación
    - datasets_dict: Diccionario con todos los datasets
    - selected_categories: Categorías específicas a probar (opcional)

    Retorna:
    - dict: Resultados de la representación
    """
    print(f"\n{'=' * 60}")
    print(f"VALIDANDO {representation_name.upper()}")
    print(f"{'=' * 60}")

    # Determinar datasets apropiados
    appropriate_categories = config['appropriate_datasets']
    if selected_categories:
        test_categories = [cat for cat in appropriate_categories if cat in selected_categories]
    else:
        test_categories = appropriate_categories

    print(f"Categorías de prueba: {test_categories}")

    results = {}

    for category in test_categories:
        if category not in datasets_dict:
            print(f"  ⚠️ Categoría '{category}' no disponible")
            continue

        print(f"\n--- Categoría: {category.upper()} ---")
        category_datasets = datasets_dict[category]

        for dataset_name, (X, y) in category_datasets.items():
            print(f"\nDataset: {dataset_name}")
            print(f"  Forma: {X.shape}")
            print(f"  Clases: {np.unique(y)} (distribución: {np.bincount(y)})")

            try:
                # Crear clasificador
                classifier_class = config['classifier']
                classifier = classifier_class(**config['params'])

                # Entrenar
                start_time = time.time()
                classifier.fit(X, y)
                training_time = time.time() - start_time

                # Evaluar
                predictions = classifier.predict(X)
                accuracy = accuracy_score(y, predictions)

                # Información adicional
                using_fallback = getattr(classifier, 'using_fallback', False)

                print(f"  Resultados:")
                print(f"    Exactitud: {accuracy:.4f} ({accuracy * 100:.1f}%)")
                print(f"    Tiempo: {training_time:.3f}s")
                print(f"    Fallback: {'Sí' if using_fallback else 'No'}")

                # Guardar resultados
                results[f"{category}_{dataset_name}"] = {
                    'accuracy': accuracy,
                    'training_time': training_time,
                    'using_fallback': using_fallback,
                    'dataset_info': {
                        'category': category,
                        'n_samples': int(X.shape[0]),
                        'n_features': int(X.shape[1]),
                        'n_classes': len(np.unique(y))
                    }
                }

            except Exception as e:
                print(f"  ❌ Error: {e}")
                results[f"{category}_{dataset_name}"] = {
                    'error': str(e),
                    'dataset_info': {
                        'category': category,
                        'n_samples': int(X.shape[0]),
                        'n_features': int(X.shape[1])
                    }
                }

    return results


def analyze_representation_performance(representation_name, results):
    """
    Analiza el rendimiento de una representación específica.

    Parámetros:
    - representation_name: Nombre de la representación
    - results: Resultados de la representación
    """
    print(f"\n--- ANÁLISIS DE {representation_name.upper()} ---")

    successful_results = {k: v for k, v in results.items() if 'accuracy' in v}
    failed_results = {k: v for k, v in results.items() if 'error' in v}

    if failed_results:
        print(f"Fallos detectados ({len(failed_results)}):")
        for dataset, error_info in failed_results.items():
            print(f"  ❌ {dataset}: {error_info['error'][:50]}...")

    if not successful_results:
        print("❌ No hay resultados válidos para analizar")
        return

    # Estadísticas generales
    accuracies = [r['accuracy'] for r in successful_results.values()]
    times = [r['training_time'] for r in successful_results.values()]
    fallback_count = sum(1 for r in successful_results.values() if r.get('using_fallback', False))

    print(f"\nEstadísticas generales:")
    print(f"  Tests exitosos: {len(successful_results)}")
    print(f"  Exactitud promedio: {np.mean(accuracies):.3f}")
    print(f"  Exactitud máxima: {max(accuracies):.3f}")
    print(f"  Exactitud mínima: {min(accuracies):.3f}")
    print(f"  Tiempo promedio: {np.mean(times):.3f}s")
    print(f"  Tests con fallback: {fallback_count}/{len(successful_results)}")

    # Rendimiento por categoría
    categories = set(r['dataset_info']['category'] for r in successful_results.values())

    print(f"\nRendimiento por categoría:")
    for category in categories:
        category_results = [r for r in successful_results.values()
                            if r['dataset_info']['category'] == category]

        if category_results:
            cat_accuracies = [r['accuracy'] for r in category_results]
            print(f"  {category}: {np.mean(cat_accuracies):.3f} promedio ({len(category_results)} tests)")

    # Mejores y peores casos
    best_dataset = max(successful_results.items(), key=lambda x: x[1]['accuracy'])
    worst_dataset = min(successful_results.items(), key=lambda x: x[1]['accuracy'])

    print(f"\nMejor caso:")
    print(f"  {best_dataset[0]}: {best_dataset[1]['accuracy']:.3f}")

    print(f"Peor caso:")
    print(f"  {worst_dataset[0]}: {worst_dataset[1]['accuracy']:.3f}")


def run_comprehensive_validation(selected_representations=None, selected_categories=None):
    """
    Ejecuta validación completa de todas las representaciones.

    Parámetros:
    - selected_representations: Lista de representaciones a probar (opcional)
    - selected_categories: Lista de categorías de datasets a usar (opcional)

    Retorna:
    - dict: Resultados completos de validación
    """
    print("=" * 80)
    print("VALIDACIÓN INTEGRAL DE REPRESENTACIONES ALTERNATIVAS")
    print("=" * 80)
    print("Probando cada representación en sus dominios apropiados")
    print()

    # Preparar configuraciones y datasets
    configs = get_representation_configs()
    datasets = prepare_validation_datasets()

    # Filtrar representaciones si se especifica
    if selected_representations:
        configs = {k: v for k, v in configs.items() if k in selected_representations}

    print(f"Representaciones a probar: {list(configs.keys())}")
    print(f"Categorías de datasets disponibles: {list(datasets.keys())}")
    print()

    # Ejecutar validaciones
    all_results = {}

    for repr_name, config in configs.items():
        try:
            results = test_single_representation(repr_name, config, datasets, selected_categories)
            all_results[repr_name] = results

            # Análisis inmediato
            analyze_representation_performance(repr_name, results)

        except Exception as e:
            print(f"❌ Error validando {repr_name}: {e}")
            all_results[repr_name] = {'global_error': str(e)}

    return all_results


def generate_comparative_report(all_results):
    """
    Genera reporte comparativo entre todas las representaciones.

    Parámetros:
    - all_results: Resultados de todas las representaciones
    """
    print("\n" + "=" * 80)
    print("REPORTE COMPARATIVO FINAL")
    print("=" * 80)

    # Calcular estadísticas por representación
    representation_stats = {}

    for repr_name, results in all_results.items():
        if 'global_error' in results:
            representation_stats[repr_name] = {'status': 'error', 'error': results['global_error']}
            continue

        successful = {k: v for k, v in results.items() if 'accuracy' in v}

        if not successful:
            representation_stats[repr_name] = {'status': 'no_results'}
            continue

        accuracies = [r['accuracy'] for r in successful.values()]
        times = [r['training_time'] for r in successful.values()]
        fallback_count = sum(1 for r in successful.values() if r.get('using_fallback', False))

        representation_stats[repr_name] = {
            'status': 'success',
            'n_tests': len(successful),
            'avg_accuracy': np.mean(accuracies),
            'max_accuracy': max(accuracies),
            'avg_time': np.mean(times),
            'fallback_rate': fallback_count / len(successful) if successful else 0
        }

    # Ranking por exactitud promedio
    successful_reprs = {k: v for k, v in representation_stats.items() if v['status'] == 'success'}

    if successful_reprs:
        sorted_reprs = sorted(successful_reprs.items(),
                              key=lambda x: x[1]['avg_accuracy'],
                              reverse=True)

        print("RANKING POR EXACTITUD PROMEDIO:")
        for i, (repr_name, stats) in enumerate(sorted_reprs, 1):
            fallback_note = f" ({stats['fallback_rate'] * 100:.0f}% fallback)" if stats['fallback_rate'] > 0 else ""
            print(f"  {i}. {repr_name:15}: {stats['avg_accuracy']:.3f} "
                  f"({stats['n_tests']} tests){fallback_note}")

        # Estadísticas generales
        print(f"\nESTADÍSTICAS GENERALES:")
        print(f"  Representaciones probadas: {len(all_results)}")
        print(f"  Representaciones exitosas: {len(successful_reprs)}")
        print(f"  Tests totales: {sum(s['n_tests'] for s in successful_reprs.values())}")
        print(f"  Exactitud promedio global: {np.mean([s['avg_accuracy'] for s in successful_reprs.values()]):.3f}")

        # Mejores en cada categoría
        print(f"\nMEJORES POR CATEGORÍA:")
        categories = set()
        for results in all_results.values():
            for result in results.values():
                if isinstance(result, dict) and 'dataset_info' in result:
                    categories.add(result['dataset_info']['category'])

        for category in categories:
            best_repr = None
            best_accuracy = 0

            for repr_name, results in all_results.items():
                category_accuracies = [r['accuracy'] for r in results.values()
                                       if isinstance(r, dict) and 'accuracy' in r and
                                       r['dataset_info']['category'] == category]

                if category_accuracies:
                    avg_accuracy = np.mean(category_accuracies)
                    if avg_accuracy > best_accuracy:
                        best_accuracy = avg_accuracy
                        best_repr = repr_name

            if best_repr:
                print(f"  {category:15}: {best_repr} ({best_accuracy:.3f})")

    else:
        print("❌ No hay resultados exitosos para comparar")

    # Recomendaciones
    print(f"\nRECOMENDACIONES:")

    # Detectar representaciones problemáticas
    problematic = []
    for repr_name, stats in representation_stats.items():
        if stats['status'] == 'error':
            problematic.append(f"{repr_name} (error global)")
        elif stats['status'] == 'no_results':
            problematic.append(f"{repr_name} (sin resultados)")
        elif stats.get('fallback_rate', 0) > 0.8:
            problematic.append(f"{repr_name} (alto fallback: {stats['fallback_rate'] * 100:.0f}%)")

    if problematic:
        print(f"  🔧 Revisar implementación: {', '.join(problematic)}")

    # Representaciones sólidas
    solid = [name for name, stats in representation_stats.items()
             if stats['status'] == 'success' and stats.get('fallback_rate', 0) < 0.2]

    if solid:
        print(f"  ✅ Implementaciones sólidas: {', '.join(solid)}")


def save_comprehensive_results(all_results, output_file="pruebas/results_all_representations.json"):
    """
    Guarda resultados completos de validación.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Preparar metadatos
    summary = {
        'validation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'representations_tested': list(all_results.keys()),
        'total_representations': len(all_results),
        'successful_representations': len(
            [r for r in all_results.values() if not any('error' in str(v) for v in r.values())]),
        'results': all_results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResultados completos guardados en: {output_file}")


def main():
    """Función principal para ejecución independiente."""
    import argparse

    parser = argparse.ArgumentParser(description='Validación integral de representaciones alternativas')
    parser.add_argument('--representations', nargs='+',
                        choices=['Intervalos', 'Hiperelipsoides', 'EnvolturasConvexas',
                                 'RedNeuronal', 'Desordenado', 'ExpresionesS',
                                 'ExpresionesGenes', 'LogicaDifusa'],
                        help='Representaciones específicas a probar')
    parser.add_argument('--categories', nargs='+',
                        choices=['boolean', 'interval', 'ellipsoid', 'fuzzy',
                                 'simple_classification', 'curved_boundaries', 'high_dimensional'],
                        help='Categorías de datasets específicas a usar')

    args = parser.parse_args()

    try:
        # Ejecutar validación
        results = run_comprehensive_validation(args.representations, args.categories)

        # Generar reporte
        generate_comparative_report(results)

        # Guardar resultados
        save_comprehensive_results(results)

        print("\n" + "=" * 80)
        print("VALIDACIÓN INTEGRAL COMPLETADA")
        print("=" * 80)
        print("✅ Todos los tests ejecutados")
        print("📊 Reporte comparativo generado")
        print("💾 Resultados guardados")

        return results

    except Exception as e:
        print(f"❌ Error durante validación integral: {e}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    main()