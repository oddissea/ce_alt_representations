#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pruebas/test_gene_expressions.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Validación específica de Expresiones de Genes (GEP)
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 1.0

Descripción:
    Validación específica de la implementación de Programación de Expresión Génica
    en problemas de regresión simbólica y clasificación, separada del experimento principal.
"""

import sys
import os
import numpy as np
import time
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.datasets import make_regression, make_classification
import json

# Añadir path del proyecto principal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pruebas.validation_datasets import get_boolean_validation_datasets, create_interval_validation_datasets
from representations.gene_expressions import GeneExpressionsClassifier


def create_symbolic_regression_datasets():
    """
    Crea datasets específicos para validación de regresión simbólica con GEP.

    Retorna:
    - dict: Datasets con funciones matemáticas conocidas
    """
    np.random.seed(42)

    datasets = {}

    # Función polinómica simple: f(x) = x^2 + 2x + 1
    n_samples = 200
    X_poly = np.random.uniform(-5, 5, (n_samples, 1))
    y_poly = X_poly[:, 0] ** 2 + 2 * X_poly[:, 0] + 1 + np.random.normal(0, 0.1, n_samples)
    datasets['Polynomial_simple'] = (X_poly, y_poly)

    # Función trigonométrica: f(x) = sin(x) + cos(x/2)
    X_trig = np.random.uniform(-2 * np.pi, 2 * np.pi, (n_samples, 1))
    y_trig = np.sin(X_trig[:, 0]) + np.cos(X_trig[:, 0] / 2) + np.random.normal(0, 0.05, n_samples)
    datasets['Trigonometric'] = (X_trig, y_trig)

    # Función multivariable: f(x,y) = x*y + x^2 - y
    X_multi = np.random.uniform(-3, 3, (n_samples, 2))
    y_multi = (X_multi[:, 0] * X_multi[:, 1] +
               X_multi[:, 0] ** 2 - X_multi[:, 1] +
               np.random.normal(0, 0.1, n_samples))
    datasets['Multivariable'] = (X_multi, y_multi)

    # Función exponencial: f(x) = e^(-x/2) * sin(x)
    X_exp = np.random.uniform(0, 4 * np.pi, (n_samples, 1))
    y_exp = np.exp(-X_exp[:, 0] / 2) * np.sin(X_exp[:, 0]) + np.random.normal(0, 0.02, n_samples)
    datasets['Exponential'] = (X_exp, y_exp)

    return datasets


def create_gep_classification_datasets():
    """
    Crea datasets específicos para validación de clasificación con GEP.

    Retorna:
    - dict: Datasets sintéticos con diferentes características
    """
    datasets = {}

    # Dataset linealmente separable
    X_linear, y_linear = make_classification(
        n_samples=300, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, random_state=42
    )
    datasets['Linear_separable'] = (X_linear, y_linear)

    # Dataset con fronteras no lineales
    X_nonlinear, y_nonlinear = make_classification(
        n_samples=400, n_features=3, n_redundant=0, n_informative=3,
        n_clusters_per_class=2, random_state=42
    )
    datasets['Nonlinear_boundaries'] = (X_nonlinear, y_nonlinear)

    # Dataset con ruido
    X_noisy, y_noisy = make_classification(
        n_samples=250, n_features=4, n_redundant=1, n_informative=2,
        flip_y=0.1, random_state=42
    )
    datasets['Noisy_classification'] = (X_noisy, y_noisy)

    return datasets


def test_gep_on_regression():
    """
    Prueba GEP en problemas de regresión simbólica.

    Retorna:
    - dict: Resultados de regresión
    """
    print("=" * 60)
    print("VALIDACIÓN GEP - REGRESIÓN SIMBÓLICA")
    print("=" * 60)

    regression_datasets = create_symbolic_regression_datasets()
    results = {}

    for dataset_name, (X, y) in regression_datasets.items():
        print(f"\n--- {dataset_name.upper()} ---")
        print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} variables")
        print(f"Rango y: [{y.min():.3f}, {y.max():.3f}]")

        # Configurar GEP para regresión
        try:
            gep_regressor = GeneExpressionsClassifier(
                population_size=120,
                generations=35,
                chromosome_length=40,
                random_state=42
            )

            # Simular regresión con clasificación binaria
            y_binary = (y > np.median(y)).astype(int)

            start_time = time.time()
            gep_regressor.fit(X, y_binary)
            training_time = time.time() - start_time

            # Evaluar
            predictions = gep_regressor.predict(X)
            accuracy = accuracy_score(y_binary, predictions)

            print(f"Resultados (clasificación binaria simulada):")
            print(f"  Exactitud: {accuracy:.4f}")
            print(f"  Tiempo: {training_time:.2f}s")

            # Obtener expresión si está disponible
            if hasattr(gep_regressor, 'get_expression'):
                expression = gep_regressor.get_expression()
                print(f"  Expresión: {expression}")

            results[dataset_name] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'using_fallback': getattr(gep_regressor, 'using_fallback', True),
                'dataset_info': {
                    'n_samples': int(X.shape[0]),
                    'n_features': int(X.shape[1])
                }
            }

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[dataset_name] = {'error': str(e)}

    return results


def test_gep_on_classification():
    """
    Prueba GEP en problemas de clasificación pura.

    Retorna:
    - dict: Resultados de clasificación
    """
    print("\n" + "=" * 60)
    print("VALIDACIÓN GEP - CLASIFICACIÓN")
    print("=" * 60)

    classification_datasets = create_gep_classification_datasets()
    results = {}

    for dataset_name, (X, y) in classification_datasets.items():
        print(f"\n--- {dataset_name.upper()} ---")
        print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} características")
        print(f"Clases: {np.unique(y)} (distribución: {np.bincount(y)})")

        try:
            # Configurar GEP
            gep_classifier = GeneExpressionsClassifier(
                population_size=100,
                generations=30,
                chromosome_length=35,
                random_state=42
            )

            start_time = time.time()
            gep_classifier.fit(X, y)
            training_time = time.time() - start_time

            # Evaluar
            predictions = gep_classifier.predict(X)
            accuracy = accuracy_score(y, predictions)

            print(f"Resultados:")
            print(f"  Exactitud: {accuracy:.4f}")
            print(f"  Tiempo: {training_time:.2f}s")
            print(f"  Fallback: {'Sí' if getattr(gep_classifier, 'using_fallback', True) else 'No'}")

            results[dataset_name] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'using_fallback': getattr(gep_classifier, 'using_fallback', True),
                'dataset_info': {
                    'n_samples': int(X.shape[0]),
                    'n_features': int(X.shape[1]),
                    'n_classes': len(np.unique(y))
                }
            }

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[dataset_name] = {'error': str(e)}

    return results


def test_gep_on_boolean_problems():
    """
    Prueba GEP en problemas booleanos para comparar con Expresiones S.

    Retorna:
    - dict: Resultados en problemas booleanos
    """
    print("\n" + "=" * 60)
    print("VALIDACIÓN GEP - PROBLEMAS BOOLEANOS")
    print("=" * 60)
    print("Comparación con dominio de Expresiones S")

    boolean_datasets = get_boolean_validation_datasets()

    # Seleccionar subconjunto representativo
    selected_datasets = {
        'XOR_expanded': boolean_datasets['XOR_expanded'],
        'AND_expanded': boolean_datasets['AND_expanded'],
        'Multiplexor_expanded': boolean_datasets['Multiplexor_expanded']
    }

    results = {}

    for dataset_name, (X, y) in selected_datasets.items():
        print(f"\n--- {dataset_name.upper()} ---")
        print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} características")
        print(f"Problema booleano: {np.all(np.isin(X, [0, 1]))}")

        try:
            # Configurar GEP para problemas booleanos
            gep_boolean = GeneExpressionsClassifier(
                population_size=80,
                generations=25,
                chromosome_length=25,
                random_state=42
            )

            # Convertir a float para GEP
            X_float = X.astype(float)

            start_time = time.time()
            gep_boolean.fit(X_float, y)
            training_time = time.time() - start_time

            # Evaluar
            predictions = gep_boolean.predict(X_float)
            accuracy = accuracy_score(y, predictions)

            print(f"Resultados:")
            print(f"  Exactitud: {accuracy:.4f} ({accuracy * 100:.1f}%)")
            print(f"  Tiempo: {training_time:.2f}s")
            print(f"  Fallback: {'Sí' if getattr(gep_boolean, 'using_fallback', True) else 'No'}")

            results[dataset_name] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'using_fallback': getattr(gep_boolean, 'using_fallback', True),
                'dataset_info': {
                    'n_samples': int(X.shape[0]),
                    'n_features': int(X.shape[1])
                }
            }

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[dataset_name] = {'error': str(e)}

    return results


def analyze_gep_implementation():
    """
    Analiza la implementación actual de GEP para detectar problemas.
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS DE IMPLEMENTACIÓN GEP")
    print("=" * 60)

    try:
        # Crear instancia para análisis
        gep = GeneExpressionsClassifier(population_size=10, generations=5)

        print("Configuración detectada:")
        print(f"  Población: {gep.population_size}")
        print(f"  Generaciones: {gep.generations}")
        print(f"  Cromosoma: {gep.chromosome_length} genes")

        # Verificar métodos esenciales
        essential_methods = ['fit', 'predict', '_create_chromosome', '_decode_chromosome']

        print(f"\nMétodos esenciales:")
        for method in essential_methods:
            exists = hasattr(gep, method)
            print(f"  {method}: {'✅' if exists else '❌'}")

        # Verificar conjuntos de funciones y terminales
        if hasattr(gep, 'function_set'):
            print(f"\nConjunto de funciones: {gep.function_set}")

        if hasattr(gep, 'terminal_set'):
            print(f"Conjunto de terminales: será dinámico según dataset")

        # Probar creación de cromosoma
        try:
            # Simular terminal_set
            gep.terminal_set = ['X0', 'X1', '1.0', '0.0']
            chromosome = gep._create_chromosome()
            print(f"\nCromosoma de prueba creado: {len(chromosome)} genes")
            print(f"Primeros 5 genes: {chromosome[:5]}")

            # Probar decodificación
            try:
                tree = gep._decode_chromosome(chromosome)
                print(f"Decodificación: {'✅ Exitosa' if tree else '❌ Falló'}")
            except Exception as decode_error:
                print(f"Decodificación: ❌ Error - {decode_error}")

        except Exception as chromosome_error:
            print(f"Creación cromosoma: ❌ Error - {chromosome_error}")

    except Exception as e:
        print(f"❌ Error en análisis de implementación: {e}")


def compare_gep_vs_expressions_s(gep_results, expressions_s_results=None):
    """
    Compara rendimiento de GEP vs Expresiones S en problemas booleanos.

    Parámetros:
    - gep_results: Resultados de GEP
    - expressions_s_results: Resultados de Expresiones S (opcional)
    """
    print("\n" + "=" * 60)
    print("COMPARACIÓN GEP vs EXPRESIONES S")
    print("=" * 60)

    if not gep_results:
        print("❌ No hay resultados de GEP para comparar")
        return

    successful_gep = {k: v for k, v in gep_results.items() if 'accuracy' in v}

    print("RENDIMIENTO GEP EN PROBLEMAS BOOLEANOS:")
    for dataset, result in successful_gep.items():
        fallback_note = " (FALLBACK)" if result.get('using_fallback', True) else ""
        print(f"  {dataset:20}: {result['accuracy']:.3f} ({result['accuracy'] * 100:.1f}%){fallback_note}")

    if expressions_s_results:
        print("\nRENDIMIENTO EXPRESIONES S (referencia):")
        for dataset, result in expressions_s_results.items():
            if 'accuracy' in result:
                fallback_note = " (FALLBACK)" if result.get('using_fallback', False) else ""
                print(f"  {dataset:20}: {result['accuracy']:.3f} ({result['accuracy'] * 100:.1f}%){fallback_note}")

        # Análisis comparativo
        print(f"\nANÁLISIS COMPARATIVO:")
        for dataset in successful_gep.keys():
            if dataset in expressions_s_results and 'accuracy' in expressions_s_results[dataset]:
                gep_acc = successful_gep[dataset]['accuracy']
                expr_s_acc = expressions_s_results[dataset]['accuracy']
                diff = (gep_acc - expr_s_acc) * 100

                if diff > 5:
                    comparison = f"GEP superior (+{diff:.1f}pp)"
                elif diff < -5:
                    comparison = f"Expr S superior ({diff:.1f}pp)"
                else:
                    comparison = f"Rendimiento similar ({diff:+.1f}pp)"

                print(f"  {dataset}: {comparison}")

    else:
        print("\nNota: Sin resultados de Expresiones S para comparar")


def save_gep_validation_results(results, output_file="pruebas/results_gene_expressions.json"):
    """
    Guarda resultados de validación de GEP.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Combinar todos los resultados
    all_results = {}
    for category, category_results in results.items():
        for dataset, result in category_results.items():
            all_results[f"{category}_{dataset}"] = result

    validation_summary = {
        'validation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_tests': len(all_results),
        'successful_tests': len([r for r in all_results.values() if 'accuracy' in r]),
        'failed_tests': len([r for r in all_results.values() if 'error' in r]),
        'results_by_category': results,
        'all_results': all_results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(validation_summary, f, indent=2, ensure_ascii=False)

    print(f"\nResultados GEP guardados en: {output_file}")


def run_gep_validation():
    """
    Función principal de validación de GEP.
    """
    print("INICIANDO VALIDACIÓN COMPLETA DE GEP")
    print("Programación de Expresión Génica según Ferreira")
    print()

    # Análisis de implementación
    analyze_gep_implementation()

    # Ejecutar validaciones
    results = {}

    # Regresión simbólica
    results['regression'] = test_gep_on_regression()

    # Clasificación
    results['classification'] = test_gep_on_classification()

    # Problemas booleanos
    results['boolean'] = test_gep_on_boolean_problems()

    # Comparación con Expresiones S (sin datos por ahora)
    compare_gep_vs_expressions_s(results['boolean'])

    # Guardar resultados
    save_gep_validation_results(results)

    return results


def main():
    """Función principal para ejecución independiente."""
    try:
        results = run_gep_validation()

        # Resumen final
        total_tests = sum(len(category) for category in results.values())
        successful_tests = sum(
            len([r for r in category.values() if 'accuracy' in r])
            for category in results.values()
        )

        print("\n" + "=" * 60)
        print("RESUMEN FINAL GEP")
        print("=" * 60)
        print(f"Tests ejecutados: {successful_tests}/{total_tests}")

        if successful_tests > 0:
            all_accuracies = []
            fallback_count = 0

            for category in results.values():
                for result in category.values():
                    if 'accuracy' in result:
                        all_accuracies.append(result['accuracy'])
                        if result.get('using_fallback', True):
                            fallback_count += 1

            print(f"Exactitud promedio: {np.mean(all_accuracies):.3f}")
            print(f"Mejor resultado: {max(all_accuracies):.3f}")
            print(f"Tests con fallback: {fallback_count}/{successful_tests}")

            if fallback_count == successful_tests:
                print("⚠️ TODOS LOS TESTS USARON FALLBACK")
                print("❌ Implementación GEP no funciona correctamente")
            elif fallback_count > successful_tests * 0.5:
                print("⚠️ Mayoría de tests usaron fallback")
                print("🔧 Revisar implementación de decodificación")
            else:
                print("✅ Implementación GEP funciona correctamente")

        return results

    except Exception as e:
        print(f"❌ Error durante validación GEP: {e}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    main()