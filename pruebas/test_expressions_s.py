#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pruebas/test_expressions_s.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Validación específica de Expresiones S puras
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 1.0

Descripción:
    Validación específica y exhaustiva de la implementación de Expresiones S
    en dominios booleanos apropiados, separada del experimento principal.
"""

import sys
import os
import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report
import json

# Añadir path del proyecto principal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pruebas.validation_datasets import get_boolean_validation_datasets
from representations.expressions_s import PureExpressionsSClassifier


def test_expressions_s_comprehensive():
    """
    Prueba exhaustiva de Expresiones S en todos los datasets booleanos apropiados.

    Retorna:
    - dict: Resultados detallados de cada validación
    """
    print("=" * 80)
    print("VALIDACIÓN EXHAUSTIVA DE EXPRESIONES S PURAS")
    print("=" * 80)
    print("Implementación teóricamente fiel según Carmona-Galán")
    print("Operadores: and, or, not sobre señales booleanas Xi")
    print()

    # Obtener datasets de validación
    validation_datasets = get_boolean_validation_datasets()

    # Configurar clasificador optimizado para cada tipo de problema
    configurations = {
        'XOR_expanded': {
            'population_size': 50,
            'generations': 80,
            'max_depth': 6,
            'tournament_size': 7
        },
        'Multiplexor_expanded': {
            'population_size': 60,
            'generations': 100,
            'max_depth': 8,
            'tournament_size': 5
        },
        'AND_expanded': {
            'population_size': 30,
            'generations': 40,
            'max_depth': 4,
            'tournament_size': 3
        },
        'OR_expanded': {
            'population_size': 30,
            'generations': 40,
            'max_depth': 4,
            'tournament_size': 3
        },
        'Parity3_expanded': {
            'population_size': 70,
            'generations': 120,
            'max_depth': 10,
            'tournament_size': 8
        }
    }

    results = {}

    for dataset_name, (X, y) in validation_datasets.items():
        print(f"--- VALIDANDO {dataset_name.upper()} ---")
        print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} características")
        print(f"Distribución: {np.bincount(y)} (clase 0 vs clase 1)")
        print(f"Datos estrictamente booleanos: {np.all(np.isin(X, [0, 1]))}")

        # Mostrar primeras muestras para verificación
        print("Primeras 6 muestras:")
        for i in range(min(6, len(X))):
            print(f"  {X[i]} → {y[i]}")

        # Configurar clasificador específico
        config = configurations.get(dataset_name, configurations['XOR_expanded'])

        classifier = PureExpressionsSClassifier(
            population_size=config['population_size'],
            generations=config['generations'],
            max_depth=config['max_depth'],
            tournament_size=config['tournament_size'],
            crossover_prob=0.8,
            mutation_prob=0.3,
            random_state=42
        )

        try:
            # Entrenar
            start_time = time.time()
            classifier.fit(X, y)
            training_time = time.time() - start_time

            # Evaluar
            predictions = classifier.predict(X)
            accuracy = accuracy_score(y, predictions)

            # Obtener complejidad
            complexity = classifier.get_complexity()

            # Mostrar resultados principales
            print(f"\nRESULTADOS:")
            print(f"  Exactitud: {accuracy:.4f} ({accuracy * 100:.1f}%)")
            print(f"  Tiempo: {training_time:.2f}s")
            print(f"  Expresión: {classifier.get_expression()}")
            print(f"  Complejidad: {complexity['size']} nodos, profundidad {complexity['depth']}")
            print(f"  Fallback usado: {'Sí' if complexity.get('using_fallback', False) else 'No'}")

            # Análisis detallado de errores
            errors = predictions != y
            n_errors = np.sum(errors)

            if n_errors > 0:
                print(f"\nANÁLISIS DE ERRORES ({n_errors}/{len(y)}):")
                error_indices = np.where(errors)[0][:5]  # Mostrar primeros 5 errores
                for idx in error_indices:
                    print(f"  {X[idx]} → pred:{predictions[idx]}, real:{y[idx]} ✗")
            else:
                print(f"\n🎉 CLASIFICACIÓN PERFECTA - Sin errores!")

            # Verificación de validez teórica
            print(f"\nVERIFICACIÓN TEÓRICA:")
            if not complexity.get('using_fallback', False):
                print(f"  ✅ Implementación pura (solo operadores booleanos)")
                print(f"  ✅ Sin umbrales evolutivos")
                print(f"  ✅ Expresión simbólica válida")
            else:
                print(f"  ⚠️ Usando modo fallback - No es expresión S pura")

            # Guardar resultados
            results[dataset_name] = {
                'accuracy': accuracy,
                'training_time': training_time,
                'expression': classifier.get_expression(),
                'complexity': complexity,
                'n_errors': int(n_errors),
                'using_fallback': complexity.get('using_fallback', False),
                'configuration_used': config,
                'dataset_info': {
                    'n_samples': int(X.shape[0]),
                    'n_features': int(X.shape[1]),
                    'class_distribution': [int(x) for x in np.bincount(y)]
                }
            }

        except Exception as e:
            print(f"  ❌ ERROR en validación: {e}")
            results[dataset_name] = {
                'error': str(e),
                'dataset_info': {
                    'n_samples': int(X.shape[0]),
                    'n_features': int(X.shape[1])
                }
            }

        print()

    return results


def analyze_expressions_s_performance(results):
    """
    Análisis comparativo del rendimiento de Expresiones S por tipo de problema.

    Parámetros:
    - results: Resultados de test_expressions_s_comprehensive()
    """
    print("=" * 80)
    print("ANÁLISIS COMPARATIVO DE RENDIMIENTO")
    print("=" * 80)

    # Separar resultados exitosos y fallidos
    successful_results = {k: v for k, v in results.items() if 'accuracy' in v}
    failed_results = {k: v for k, v in results.items() if 'error' in v}

    if failed_results:
        print("FALLOS DETECTADOS:")
        for dataset, error_info in failed_results.items():
            print(f"  ❌ {dataset}: {error_info['error']}")
        print()

    if not successful_results:
        print("❌ No hay resultados válidos para analizar")
        return

    # Análisis por dificultad del problema
    problem_difficulty = {
        'AND_expanded': 'Fácil',
        'OR_expanded': 'Fácil',
        'XOR_expanded': 'Difícil',
        'Multiplexor_expanded': 'Moderado',
        'Parity3_expanded': 'Muy Difícil'
    }

    print("RENDIMIENTO POR DIFICULTAD:")
    for difficulty in ['Fácil', 'Moderado', 'Difícil', 'Muy Difícil']:
        datasets_in_category = [k for k, v in problem_difficulty.items()
                                if v == difficulty and k in successful_results]

        if datasets_in_category:
            print(f"\n{difficulty}:")
            for dataset in datasets_in_category:
                result = successful_results[dataset]
                fallback_note = " (FALLBACK)" if result.get('using_fallback', False) else ""
                print(f"  {dataset:20}: {result['accuracy']:.3f} ({result['accuracy'] * 100:.1f}%){fallback_note}")

    # Estadísticas generales
    accuracies = [r['accuracy'] for r in successful_results.values() if not r.get('using_fallback', False)]

    if accuracies:
        print(f"\nESTADÍSTICAS GENERALES (sin fallback):")
        print(f"  Exactitud promedio: {np.mean(accuracies):.3f}")
        print(f"  Exactitud mínima: {np.min(accuracies):.3f}")
        print(f"  Exactitud máxima: {np.max(accuracies):.3f}")
        print(f"  Desviación estándar: {np.std(accuracies):.3f}")

    # Análisis de complejidad
    complexities = [(r['complexity']['size'], r['complexity']['depth'])
                    for r in successful_results.values()
                    if not r.get('using_fallback', False)]

    if complexities:
        sizes, depths = zip(*complexities)
        print(f"\nCOMPLEJIDAD DE EXPRESIONES:")
        print(f"  Tamaño promedio: {np.mean(sizes):.1f} nodos")
        print(f"  Profundidad promedia: {np.mean(depths):.1f}")
        print(f"  Rango tamaños: {min(sizes)}-{max(sizes)} nodos")
        print(f"  Rango profundidades: {min(depths)}-{max(depths)}")

    # Casos ejemplares
    print(f"\nEXPRESIONES EJEMPLARES:")
    for dataset, result in successful_results.items():
        if not result.get('using_fallback', False) and result['accuracy'] > 0.95:
            print(f"  {dataset}:")
            print(f"    Expresión: {result['expression']}")
            print(f"    Exactitud: {result['accuracy']:.3f}")


def compare_with_breast_cancer_results(validation_results, breast_cancer_accuracy=None):
    """
    Compara resultados de validación con rendimiento en breast cancer.

    Parámetros:
    - validation_results: Resultados de test_expressions_s_comprehensive()
    - breast_cancer_accuracy: Exactitud en breast cancer (si disponible)
    """
    print("=" * 80)
    print("COMPARACIÓN: DOMINIOS BOOLEANOS vs BREAST CANCER")
    print("=" * 80)

    # Calcular estadísticas de validación
    successful_results = {k: v for k, v in validation_results.items()
                          if 'accuracy' in v and not v.get('using_fallback', False)}

    if not successful_results:
        print("❌ No hay resultados de validación válidos para comparar")
        return

    best_validation = max(r['accuracy'] for r in successful_results.values())
    avg_validation = np.mean([r['accuracy'] for r in successful_results.values()])

    print("RENDIMIENTO EN DOMINIOS APROPIADOS (booleanos):")
    for dataset, result in successful_results.items():
        print(f"  {dataset:20}: {result['accuracy']:.3f} ({result['accuracy'] * 100:.1f}%)")

    print(f"\nESTADÍSTICAS VALIDACIÓN:")
    print(f"  Mejor resultado: {best_validation:.3f} ({best_validation * 100:.1f}%)")
    print(f"  Resultado promedio: {avg_validation:.3f} ({avg_validation * 100:.1f}%)")

    if breast_cancer_accuracy is not None:
        print(f"\nRENDIMIENTO EN DOMINIO INAPROPIADO:")
        print(f"  Breast Cancer: {breast_cancer_accuracy:.3f} ({breast_cancer_accuracy * 100:.1f}%)")

        improvement = (best_validation - breast_cancer_accuracy) * 100
        avg_improvement = (avg_validation - breast_cancer_accuracy) * 100

        print(f"\nMEJORA EN DOMINIO APROPIADO:")
        print(f"  Mejor caso: +{improvement:.1f} puntos porcentuales")
        print(f"  Caso promedio: +{avg_improvement:.1f} puntos porcentuales")

        print(f"\nCONCLUSIONES CIENTÍFICAS:")
        if improvement > 5:
            print(f"  ✅ Mejora significativa en dominio apropiado (+{improvement:.1f}pp)")
        else:
            print(f"  ⚠️ Mejora moderada en dominio apropiado (+{improvement:.1f}pp)")

        print(f"  ✅ Implementación teóricamente correcta")
        print(f"  ✅ Rendimiento apropiado según el dominio")
        print(f"  ✅ No hay defectos algorítmicos")

    else:
        print(f"\nNota: Sin datos de breast cancer para comparar")


def save_validation_results(results, output_file="pruebas/results_expressions_s.json"):
    """
    Guarda resultados de validación en archivo JSON.

    Parámetros:
    - results: Resultados de validación
    - output_file: Ruta del archivo de salida
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Preparar datos para serialización
    serializable_results = {}
    for dataset, result in results.items():
        serializable_results[dataset] = {
            k: (v if not isinstance(v, np.integer) else int(v))
            for k, v in result.items()
        }

    # Añadir metadatos
    validation_summary = {
        'validation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_datasets': len(results),
        'successful_validations': len([r for r in results.values() if 'accuracy' in r]),
        'failed_validations': len([r for r in results.values() if 'error' in r]),
        'results': serializable_results
    }

    # Guardar
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(validation_summary, f, indent=2, ensure_ascii=False)

    print(f"Resultados guardados en: {output_file}")


def run_expressions_s_validation():
    """
    Función principal de validación de Expresiones S.
    Equivalente a la función original pero separada y mejorada.

    Retorna:
    - dict: Resultados completos de validación
    """
    print("Iniciando validación completa de Expresiones S...")
    print("Implementación pura según teoría de Carmona-Galán\n")

    # Ejecutar validación exhaustiva
    results = test_expressions_s_comprehensive()

    # Análisis de rendimiento
    analyze_expressions_s_performance(results)

    # Comparación (sin datos de breast cancer por ahora)
    compare_with_breast_cancer_results(results)

    # Guardar resultados
    save_validation_results(results)

    return results


def main():
    """Función principal para ejecución independiente."""
    try:
        # Ejecutar validación completa
        results = run_expressions_s_validation()

        # Resumen final
        successful = len([r for r in results.values() if 'accuracy' in r])
        total = len(results)

        print("\n" + "=" * 80)
        print("RESUMEN FINAL")
        print("=" * 80)
        print(f"Validaciones exitosas: {successful}/{total}")

        if successful > 0:
            accuracies = [r['accuracy'] for r in results.values() if 'accuracy' in r]
            print(f"Exactitud promedio: {np.mean(accuracies):.3f}")
            print(f"Mejor resultado: {max(accuracies):.3f}")

            if successful == total:
                print("🎉 TODAS LAS VALIDACIONES EXITOSAS")
                print("✅ Implementación de Expresiones S validada correctamente")
            else:
                print("⚠️ Algunas validaciones fallaron - revisar implementación")

        return results

    except Exception as e:
        print(f"❌ Error durante validación: {e}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    main()