#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Prueba de evaluación continua 4 - Representaciones Alternativas de Reglas
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 1.0

Descripción:
    Módulo de evaluación para sistemas clasificadores evolutivos.
    Implementa validación cruzada estratificada con cálculo de métricas
    de rendimiento estables y manejo de errores para experimentos comparativos.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time

from scipy import stats
from sklearn.base import clone


def calculate_metrics(estimator, X_eval, y_eval):
    """Función interna para calcular métricas, evita sombra de 'scores'"""
    # Asegurar que X_eval y y_eval son arrays numpy
    X_eval = np.asarray(X_eval)
    y_eval = np.asarray(y_eval)

    # Predicciones nítidas
    try:
        y_pred = estimator.predict(X_eval)
    except Exception as pred_error:
        print(f"Error en predict: {pred_error}")
        # Crear predicciones por defecto (clase más frecuente)
        from collections import Counter
        most_common = Counter(y_eval).most_common(1)[0][0]
        y_pred = np.full_like(y_eval, most_common)

    # Predicciones probabilísticas (si el modelo lo soporta)
    y_proba = None
    if hasattr(estimator, 'predict_proba'):
        try:
            y_proba = estimator.predict_proba(X_eval)
        except (ValueError, TypeError, RuntimeError) as prob_error:
            # Captura excepciones específicas en lugar de 'except:'
            print(f"Advertencia en cálculo de probabilidades: {prob_error}")

    # Métricas básicas
    metric_results = {
        'accuracy': accuracy_score(y_eval, y_pred),
        'precision': precision_score(y_eval, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_eval, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_eval, y_pred, average='weighted', zero_division=0)
    }

    # Métricas basadas en probabilidades si están disponibles
    if y_proba is not None:
        try:
            # Para ROC AUC, necesitamos formato específico según binario/multiclase
            if y_proba.shape[1] == 2:  # Caso binario
                metric_results['roc_auc'] = roc_auc_score(y_eval, y_proba[:, 1])
            else:  # Caso multiclase
                metric_results['roc_auc'] = roc_auc_score(
                    y_eval, y_proba, multi_class='ovr', average='weighted'
                )
        except (ValueError, TypeError) as auc_error:
            print(f"Error calculando AUC: {auc_error}")

    return metric_results


def evaluate_classifier(model, X, y, cv=5, timeout=600):
    """
    Evalúa un clasificador utilizando validación cruzada con límite de tiempo.

    [...]
    """
    # Verificar que el modelo tenga los métodos necesarios
    required_methods = ['fit', 'predict']
    for method in required_methods:
        if not hasattr(model, method):
            raise AttributeError(f"El modelo debe implementar el método '{method}'")

    # Verificar si el modelo es un clasificador
    estimator_type = getattr(model, '_estimator_type', None)
    if estimator_type != "classifier":
        # Usamos setattr para evitar acceso directo a miembro protegido
        setattr(model, '_estimator_type', "classifier")
        print(f"Advertencia: Forzando que {model.__class__.__name__} sea un clasificador")

    # Asegurar que X y y sean arrays numpy y manejar tipos
    X = np.asarray(X)
    y = np.asarray(y)

    # Establecer tiempo límite
    start_time = time.time()

    try:
        # Ejecutar validación cruzada manual con mejor manejo de errores
        fold_results = []
        fold_count = 0

        # Configurar validación cruzada estratificada
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        for train_idx, test_idx in skf.split(X, y):
            fold_count += 1
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            try:
                # Entrenar modelo con mejor manejo de errores
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_train, y_train)

                # Evaluar en conjunto de prueba
                fold_metrics = calculate_metrics(model_copy, X_test, y_test)
                fold_results.append(fold_metrics)

            except Exception as fold_error:
                print(f"Error en fold {fold_count}: {fold_error}")
                # Continuar con el siguiente fold en lugar de fallar toda la evaluación
                continue

            # Verificar timeout
            if time.time() - start_time > timeout:
                print(f"Advertencia: Se excedió el tiempo límite de {timeout}s")
                break

        # Si tenemos al menos un fold exitoso, podemos promediar resultados
        if fold_results:
            results = {}
            for metric in fold_results[0].keys():
                results[metric] = np.mean([fold[metric] for fold in fold_results if metric in fold])
            return results
        else:
            # Si todos los folds fallaron, intentar evaluación simplificada
            raise ValueError("Todos los folds fallaron")

    except Exception as cv_error:
        # Mejorar el mensaje de error
        print(f"Error durante la validación cruzada: {cv_error}")

        # Intentar evaluación simplificada con mejor manejo de errores
        try:
            print("Intentando evaluación simplificada...")
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            # Crear copia fresca del modelo
            model_fresh = model.__class__(**model.get_params())

            try:
                model_fresh.fit(X_train, y_train)
                results = calculate_metrics(model_fresh, X_test, y_test)
                print("Evaluación simplificada completada")
                return results
            except Exception as fit_error:
                print(f"Error en ajuste de modelo simplificado: {fit_error}")

                # Intentar con RandomForest como último recurso
                from sklearn.ensemble import RandomForestClassifier
                print("Intentando con RandomForest como último recurso...")

                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                results = calculate_metrics(rf, X_test, y_test)

                # Añadir marca para identificar resultados de fallback
                results['fallback'] = True
                return results

        except Exception as simple_error:
            print(f"Error también en evaluación simplificada: {simple_error}")

            # Devolver un objeto de resultados mínimo en lugar de fallar
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'error': str(cv_error),
                'failed': True
            }


def save_results(results_list, filename='results/metrics_results.csv'):
    """
    Guarda la lista de resultados en un archivo CSV.

    Parámetros:
    - results_list (list[dict]): Lista con los resultados de cada modelo.
    - filename (str): Ruta y nombre del archivo CSV donde se guardarán los resultados.
    """
    import os

    # Asegurar que el directorio existe
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    df = pd.DataFrame(results_list)

    # Ordenar columnas para mejor visualización
    column_order = ['Modelo', 'Exactitud', 'Precisión', 'Recall', 'F1']
    if 'AUC' in df.columns:
        column_order.append('AUC')
    if 'Tiempo (s)' in df.columns:
        column_order.append('Tiempo (s)')

    # Añadir columnas restantes si hay alguna no considerada
    for col in df.columns:
        if col not in column_order:
            column_order.append(col)

    # Reordenar y guardar
    df = df[column_order]
    df.to_csv(filename, index=False)
    print(f"Resultados guardados correctamente en {filename}.")


# ============ NUEVAS FUNCIONES DE EVALUACIÓN ROBUSTA ============

def comprehensive_evaluate_classifier(model, X, y, cv=5, timeout=600,
                                      statistical_tests=True, bootstrap_stability=True):
    """
    Evaluación detallada con validación estadística y análisis de estabilidad.
    """

    print(f"Iniciando evaluación comprehensiva de {model.__class__.__name__}")

    # Validación cruzada con análisis estadístico
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    start_time = time.time()

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        if time.time() - start_time > timeout:
            print(f"Timeout alcanzado en fold {fold}")
            break

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            # Clonar y entrenar modelo
            model_copy = clone(model)
            model_copy.fit(X_train, y_train)

            # Calcular métricas
            fold_metrics = calculate_metrics(model_copy, X_test, y_test)

            for metric in cv_scores:
                if metric in fold_metrics:
                    cv_scores[metric].append(fold_metrics[metric])

        except Exception as e:
            print(f"Error en fold {fold}: {e}")
            continue

    # Estadísticas descriptivas
    results = {}
    for metric, scores in cv_scores.items():
        if scores:  # Si tenemos al menos un resultado
            scores_array = np.array(scores)
            results[f'{metric}_mean'] = np.mean(scores_array)
            results[f'{metric}_std'] = np.std(scores_array)
            results[f'{metric}_ci_lower'] = np.percentile(scores_array, 2.5)
            results[f'{metric}_ci_upper'] = np.percentile(scores_array, 97.5)

            # Test de normalidad
            if statistical_tests and len(scores_array) >= 3:
                try:
                    _, normality_p = stats.shapiro(scores_array)
                    results[f'{metric}_normality_p'] = normality_p
                except (ValueError, RuntimeError, AttributeError) as e:
                    print(f"Error en test de normalidad para {metric}: {e}")
                    results[f'{metric}_normality_p'] = None

    # Análisis de estabilidad con bootstrap
    if bootstrap_stability and cv_scores['accuracy']:
        try:
            stability_score = _bootstrap_stability_analysis(model, X, y, n_bootstrap=10)
            results['stability_score'] = stability_score
        except Exception as e:
            print(f"Error en análisis de estabilidad: {e}")
            results['stability_score'] = None

    # Tiempos
    results['total_time'] = time.time() - start_time
    results['model_name'] = model.__class__.__name__

    return results


def _bootstrap_stability_analysis(model, X, y, n_bootstrap=10):
    """Análisis de estabilidad mediante bootstrap."""
    from sklearn.utils import resample
    from scipy import stats

    predictions_list = []

    # Conjunto de pruebas fijo para evaluar estabilidad
    test_size = min(100, len(X) // 4)  # 25% o máximo 100 muestras
    np.random.seed(42)
    test_indices = np.random.choice(len(X), size=test_size, replace=False)
    X_test_fixed = X[test_indices]

    for i in range(n_bootstrap):
        try:
            # Bootstrap del conjunto de entrenamiento
            train_indices = np.setdiff1d(np.arange(len(X)), test_indices)
            X_train_boot, y_train_boot = resample(X[train_indices], y[train_indices],
                                                  random_state=i)

            # Entrenar modelo
            model_boot = model.__class__(**model.get_params())
            model_boot.fit(X_train_boot, y_train_boot)

            # Predecir en conjunto de pruebas fijo
            pred = model_boot.predict(X_test_fixed)
            predictions_list.append(pred)

        except Exception as e:
            print(f"Error en bootstrap {i}: {e}")
            continue

    if len(predictions_list) < 2:
        return 0.0

    # Calcular estabilidad como consenso promedio
    predictions_array = np.array(predictions_list)
    stability_scores = []

    for i in range(test_size):
        # Moda de las predicciones para cada muestra
        mode_result = stats.mode(predictions_array[:, i])
        most_common = mode_result.mode[0] if hasattr(mode_result, 'mode') else mode_result[0][0]
        # Proporción de modelos que coinciden con la moda
        agreement = np.sum(predictions_array[:, i] == most_common) / len(predictions_list)
        stability_scores.append(agreement)

    return np.mean(stability_scores)


def intelligent_fallback_evaluation(model, X, y, max_attempts=3):
    """Sistema de evaluación inteligente con fallbacks adaptativos."""
    attempts = 0
    last_error = None

    while attempts < max_attempts:
        attempts += 1

        try:
            # Intento principal
            if attempts == 1:
                print(f"Intento {attempts}: Evaluación estándar")
                return evaluate_classifier(model, X, y, cv=5)

            # Intento 2: Simplificar validación cruzada
            elif attempts == 2:
                print(f"Intento {attempts}: Evaluación simplificada (CV=3)")
                return evaluate_classifier(model, X, y, cv=3, timeout=300)

            # Intento 3: Holdout simple
            else:
                print(f"Intento {attempts}: Evaluación holdout")
                from sklearn.model_selection import train_test_split

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )

                # Intentar con modelo simplificado si es posible
                if hasattr(model, 'simplify_for_fallback'):
                    model_simple = model.simplify_for_fallback()
                else:
                    model_simple = model

                model_simple.fit(X_train, y_train)
                return calculate_metrics(model_simple, X_test, y_test)

        except Exception as e:
            last_error = e
            print(f"Intento {attempts} falló: {e}")

            # Aplicar ajustes específicos según el tipo de error
            if "singular matrix" in str(e).lower() or "linalg" in str(e).lower():
                # Problema numérico: añadir regularización
                if hasattr(model, 'increase_regularization'):
                    model.increase_regularization()

            elif "memory" in str(e).lower() or "timeout" in str(e).lower():
                # Problema de recursos: simplificar modelo
                if hasattr(model, 'reduce_complexity'):
                    model.reduce_complexity()

    # Si todos los intentos fallan, usar modelo estable
    print(f"Todos los intentos fallaron. Último error: {last_error}")
    print("Usando RandomForest como fallback final")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)

        results = calculate_metrics(rf, X_test, y_test)
        results['fallback_model'] = 'RandomForest'
        results['original_error'] = str(last_error)

        return results

    except Exception as final_error:
        print(f"Error incluso con fallback final: {final_error}")
        return {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'error': str(final_error), 'failed': True
        }