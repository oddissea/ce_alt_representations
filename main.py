#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Prueba de evaluación continua 4 - Representaciones Alternativas de Reglas
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 1.0

Descripción:
    Script principal para la comparación experimental de representaciones
    alternativas de reglas en sistemas clasificadores evolutivos.
    Evalúa ocho paradigmas representacionales sobre el dataset Wisconsin
    Breast Cancer con análisis dimensional comparativo.
"""

from utils.evaluation import evaluate_classifier, save_results
from utils.visualizations import create_visualizations

from representations.intervals import IntervalClassifier
from representations.hyperellipsoids import HyperEllipsoidClassifier
from representations.convex_hulls import ConvexHullClassifier
from representations.neural_network import NeuralNetworkClassifier
from representations.unordered import UnorderedClassifier
from representations.expressions_s import ExpressionsSClassifier
from representations.gene_expressions import GeneExpressionsClassifier
from representations.fuzzy_logic import FuzzyLogicClassifier

import pandas as pd
import traceback
import time
import os
import argparse


def parse_arguments():
    """Configura los argumentos de línea de comandos para el script."""
    parser = argparse.ArgumentParser(
        description='Comparar diferentes representaciones de reglas en sistemas clasificadores evolutivos.')

    # Argumentos para omitir representaciones
    parser.add_argument('-i', '--skip-intervals', action='store_true',
                        help='Omitir evaluación de la representación por intervalos')
    parser.add_argument('-he', '--skip-hyperellipsoids', action='store_true',
                        help='Omitir evaluación de la representación por hiperelipsoides')
    parser.add_argument('-ch', '--skip-convexhull', action='store_true',
                        help='Omitir evaluación de la representación por envolturas convexas')
    parser.add_argument('-nn', '--skip-neural', action='store_true',
                        help='Omitir evaluación de la representación por redes neuronales')
    parser.add_argument('-u', '--skip-unordered', action='store_true',
                        help='Omitir evaluación de la representación desordenada')
    parser.add_argument('-es', '--skip-expressions', action='store_true',
                        help='Omitir evaluación de la representación por expresiones S')
    parser.add_argument('-ge', '--skip-gene', action='store_true',
                        help='Omitir evaluación de la representación por expresiones de genes')
    parser.add_argument('-fl', '--skip-fuzzy', action='store_true',
                        help='Omitir evaluación de la representación por lógica difusa')

    # Grupo para selección de dataset
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument('-dso', '--dataset-original', action='store_true',
                               help='Usar solo el dataset original completo')
    dataset_group.add_argument('-dsp', '--dataset-prepared', action='store_true',
                               help='Usar solo el dataset preparado con selección de características')
    dataset_group.add_argument('-both', '--both-datasets', action='store_true',
                               help='Ejecutar con ambos datasets y guardar resultados separados')

    # Argumento para omitir todas las visualizaciones
    parser.add_argument('-nv', '--no-visualizations', action='store_true',
                        help='No generar visualizaciones')

    # Argumentos adicionales
    parser.add_argument('-o', '--output-dir', type=str, default='results',
                        help='Directorio base donde guardar los resultados')
    parser.add_argument('-v', '--visualizations-dir', type=str, default='visualizations',
                        help='Directorio base donde guardar las visualizaciones')

    # Argumento para especificar parámetros específicos de la lógica difusa
    parser.add_argument('--fuzzy-bins', type=int, default=3,
                        help='Número de conjuntos difusos por característica (por defecto: 3)')

    return parser.parse_args()


def get_classifiers(fuzzy_bins):
    """Retorna el diccionario de clasificadores con la configuración especificada."""
    return {
        "Intervalos": IntervalClassifier(),
        "Hiperelipsoides": HyperEllipsoidClassifier(),
        "EnvolturasConvexas": ConvexHullClassifier(n_components=5),
        "RedNeuronal": NeuralNetworkClassifier(hidden_layer_sizes=(20, 10), max_iter=200),
        "Desordenado": UnorderedClassifier(max_depth=4),
        "ExpresionesS": ExpressionsSClassifier(population_size=200, generations=10),
        "ExpresionesGenes": GeneExpressionsClassifier(population_size=150, generations=8),
        "LogicaDifusa": FuzzyLogicClassifier(n_bins=fuzzy_bins)
    }


def filter_classifiers(all_classifiers, args):
    """Filtra los clasificadores según los argumentos de línea de comandos."""
    classifiers = {}
    if not args.skip_intervals:
        classifiers["Intervalos"] = all_classifiers["Intervalos"]
    if not args.skip_hyperellipsoids:
        classifiers["Hiperelipsoides"] = all_classifiers["Hiperelipsoides"]
    if not args.skip_convexhull:
        classifiers["EnvolturasConvexas"] = all_classifiers["EnvolturasConvexas"]
    if not args.skip_neural:
        classifiers["RedNeuronal"] = all_classifiers["RedNeuronal"]
    if not args.skip_unordered:
        classifiers["Desordenado"] = all_classifiers["Desordenado"]
    if not args.skip_expressions:
        classifiers["ExpresionesS"] = all_classifiers["ExpresionesS"]
    if not args.skip_gene:
        classifiers["ExpresionesGenes"] = all_classifiers["ExpresionesGenes"]
    if not args.skip_fuzzy:
        classifiers["LogicaDifusa"] = all_classifiers["LogicaDifusa"]

    return classifiers


def run_experiment(dataset_name, X, y, classifiers, results_dir, vis_dir, no_visualizations):
    """Ejecuta el experimento completo para un dataset específico."""
    print(f"\n{'=' * 60}")
    print(f"EJECUTANDO EXPERIMENTO CON DATASET {dataset_name.upper()}")
    print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} características")
    print(f"{'=' * 60}")

    print(f"\nEvaluando {len(classifiers)} clasificadores: {', '.join(classifiers.keys())}\n")

    results = []

    for name, clf in classifiers.items():
        print(f"→ {name}")
        start_time = time.time()
        try:
            # Máximo 10 minutos por clasificador
            scores = evaluate_classifier(clf, X, y, cv=5)
            elapsed = time.time() - start_time

            # Verificar si hubo error o fallback
            if 'failed' in scores and scores['failed']:
                print(f"  Error: {scores.get('error', 'Desconocido')}")
                continue

            if 'fallback' in scores and scores['fallback']:
                print(f"  ⚠️ Usando modelo alternativo")

            scores['Modelo'] = name
            scores['Tiempo (s)'] = elapsed
            results.append(scores)
            print(
                f"  Exactitud: {scores.get('accuracy', scores.get('Exactitud', 0)):.4f}, " +
                f"Tiempo: {elapsed:.2f} segundos"
            )
        except Exception as e:
            print(f"  Error evaluando {name}: {e}")
            traceback.print_exc()
            continue

    # Procesar y guardar resultados
    if results:
        # Estandarizar nombres de métricas
        for result in results:
            if 'accuracy' in result:
                result['Exactitud'] = result.pop('accuracy')
            if 'precision' in result:
                result['Precisión'] = result.pop('precision')
            if 'recall' in result:
                result['Recall'] = result.pop('recall')
            if 'f1' in result:
                result['F1'] = result.pop('f1')
            if 'roc_auc' in result:
                result['AUC'] = result.pop('roc_auc')
            # Eliminar marcadores internos si existen
            if 'fallback' in result:
                result.pop('fallback')
            if 'error' in result:
                result.pop('error')
            if 'failed' in result:
                result.pop('failed')

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='Exactitud', ascending=False)

        print(f"\nResultados completos para dataset {dataset_name}:")
        print(results_df)

        # Guardar resultados en CSV
        print(f"\nGuardando resultados métricos en {results_dir}...")
        try:
            save_results(results, filename=os.path.join(results_dir, "metrics_results.csv"))
        except Exception as e:
            print(f"Error al guardar resultados métricos: {e}")
            # Alternativa: mostrar tabla en formato CSV
            print("\nResultados en formato CSV:")
            print(results_df.to_csv(index=False))

        # Generar y guardar visualizaciones
        if not no_visualizations:
            print(f"\nCreando visualizaciones en {vis_dir}...")
            try:
                create_visualizations(classifiers, X, y, results_df, output_dir=vis_dir)
            except Exception as e:
                print(f"Error al crear visualizaciones: {e}")
                traceback.print_exc()
        else:
            print("\nVisualizaciones omitidas según argumentos.")

        return results_df
    else:
        print(f"No se pudieron obtener resultados válidos para dataset {dataset_name}.")
        return None


def main():
    # Procesar argumentos de línea de comandos
    args = parse_arguments()

    # Obtener clasificadores
    all_classifiers = get_classifiers(args.fuzzy_bins)
    classifiers = filter_classifiers(all_classifiers, args)

    if not classifiers:
        print("Error: No hay clasificadores seleccionados. Todos han sido omitidos.")
        return

    # Determinar qué datasets ejecutar
    if args.both_datasets:
        # Ejecutar con ambos datasets
        print("Ejecutando experimentos con ambos datasets...")

        # Dataset original
        print("Cargando dataset original...")
        from data.breast_cancer_dataset import load_full_dataset
        X_original, y_original = load_full_dataset()

        # Dataset preparado
        print("Cargando dataset preparado...")
        from data.breast_cancer_dataset import load_prepared_dataset
        X_prepared, y_prepared = load_prepared_dataset()

        # Crear directorios
        results_original = args.output_dir
        results_prepared = args.output_dir + "_reduced"
        vis_original = args.visualizations_dir
        vis_prepared = args.visualizations_dir + "_reduced"

        for dir_path in [results_original, results_prepared, vis_original, vis_prepared]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        # Ejecutar experimentos
        results_df_original = run_experiment(
            "original", X_original, y_original,
            get_classifiers(args.fuzzy_bins), results_original, vis_original, args.no_visualizations
        )

        results_df_prepared = run_experiment(
            "preparado", X_prepared, y_prepared,
            filter_classifiers(get_classifiers(args.fuzzy_bins), args),
            results_prepared, vis_prepared, args.no_visualizations
        )

        # Crear tabla comparativa
        if results_df_original is not None and results_df_prepared is not None:
            print("\n" + "=" * 80)
            print("COMPARACIÓN ENTRE DATASETS")
            print("=" * 80)

            # Merge de resultados para comparación
            comparison_df = results_df_original[['Modelo', 'Exactitud']].merge(
                results_df_prepared[['Modelo', 'Exactitud']],
                on='Modelo',
                suffixes=('_Original', '_Preparado')
            )
            comparison_df['Cambio'] = comparison_df['Exactitud_Preparado'] - comparison_df['Exactitud_Original']
            comparison_df['Cambio_Porcentual'] = (comparison_df['Cambio'] / comparison_df['Exactitud_Original']) * 100

            print(comparison_df)

            # Guardar comparación
            # Crear directorio de comparación y guardar
            comparison_dir = args.output_dir + "_comparison"
            if not os.path.exists(comparison_dir):
                os.makedirs(comparison_dir)

            # Guardar comparación
            comparison_df.to_csv(os.path.join(comparison_dir, "comparison_results.csv"), index=False)
            print(f"Tabla comparativa guardada en {comparison_dir}/comparison_results.csv")

    elif args.dataset_prepared:
        # Solo dataset preparado
        print("Cargando dataset preparado...")
        from data.breast_cancer_dataset import load_prepared_dataset
        X, y = load_prepared_dataset()
        dataset_type = "preparado con selección de características"

        # Crear directorios
        results_dir = args.output_dir
        vis_dir = args.visualizations_dir
        for dir_path in [results_dir, vis_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        run_experiment(dataset_type, X, y, classifiers, results_dir, vis_dir, args.no_visualizations)

    else:
        # Solo dataset original (por defecto)
        print("Cargando dataset original...")
        from data.breast_cancer_dataset import load_full_dataset
        X, y = load_full_dataset()
        dataset_type = "original completo"

        # Crear directorios
        results_dir = args.output_dir
        vis_dir = args.visualizations_dir
        for dir_path in [results_dir, vis_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        run_experiment(dataset_type, X, y, classifiers, results_dir, vis_dir, args.no_visualizations)

    print("\nProceso finalizado.")


if __name__ == "__main__":
    main()

    # Solo dataset original
    # python main.py -dso --fuzzy-bins 5

    # Solo dataset preparado
    # python main.py -dsp --fuzzy-bins 5

    # Ambos datasets, excluyendo convex hull
    # python main.py --both-datasets -ch --fuzzy-bins 5

    # Ambos datasets, sin visualizaciones
    # python main.py --both-datasets -nv --fuzzy-bins 5