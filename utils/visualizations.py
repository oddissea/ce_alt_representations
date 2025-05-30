#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualizations.py

COMPUTACIÓN EVOLUTIVA
PEC 4: Prueba de evaluación continua 4 - Representaciones Alternativas de Reglas
Máster en Investigación en Inteligencia Artificial
Universidad Nacional de Educación a Distancia (UNED)

Autor: Fernando H. Nasser-Eddine López
Email: fnassered1@alumno.uned.es
Fecha: 31/05/2025
Versión: 1.0

Descripción:
    Módulo de visualización para análisis comparativo de representaciones.
    Genera gráficas de rendimiento, matrices de confusión y las curvas ROC,
    fronteras de decisión y visualizaciones específicas de reglas por
    tipo de representación en sistemas clasificadores evolutivos.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
import os
from matplotlib.patches import Rectangle, Circle, Ellipse  # Para visualización de reglas


def create_visualizations(classifiers, X, y, results_df, output_dir="visualizations"):
    """
    Crea y guarda visualizaciones para los resultados de los clasificadores.
    """
    # Crear directorio si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Asegurarse de que X es 2D y y es 1D
    X = np.asarray(X)
    y = np.asarray(y)

    # Si X es 1D, convertirlo a 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Primero reentrenar todos los modelos para visualización con mejor manejo de errores
    print("Re-entrenando modelos para visualización...")
    trained_classifiers = {}
    for name, clf in classifiers.items():
        try:
            # Crear copia fresca del clasificador
            clf_copy = clf.__class__(**clf.get_params())
            clf_copy.fit(X, y)
            trained_classifiers[name] = clf_copy
            print(f"  ✓ {name} re-entrenado")
        except Exception as e:
            print(f"  ✗ Error re-entrenando {name}: {e}")

            # Intentar con RandomForest como fallback
            try:
                from sklearn.ensemble import RandomForestClassifier
                fallback = RandomForestClassifier(n_estimators=100, random_state=42)
                fallback.fit(X, y)
                trained_classifiers[name] = fallback
                print(f"  ⚠️ {name} usando RandomForest (fallback)")
            except Exception as fallback_error:
                print(f"  ✗✗ Error con fallback para {name}: {fallback_error}")
                # Si falla todo, ignorar este clasificador
                continue

    # Filtrar resultados_df para incluir solo clasificadores entrenados
    available_models = list(trained_classifiers.keys())
    filtered_results_df = results_df[results_df['Modelo'].isin(available_models)]

    # Ejecutar todas las visualizaciones con manejo mejorado de errores
    try:
        plot_accuracy_comparison(filtered_results_df, output_dir)
    except Exception as e:
        print(f"Error en gráfica de comparación de exactitud: {e}")

    # try:
    #     plot_confusion_matrices(trained_classifiers, X, y, output_dir)
    # except Exception as e:
    #     print(f"Error en matrices de confusión: {e}")

    try:
        plot_roc_curves(trained_classifiers, X, y, output_dir)
    except Exception as e:
        print(f"Error en curvas ROC: {e}")

    # try:
    #     plot_time_vs_accuracy(filtered_results_df, output_dir)
    # except Exception as e:
    #     print(f"Error en gráfica tiempo vs exactitud: {e}")
    #
    # try:
    #     plot_decision_boundaries(trained_classifiers, X, y, output_dir)
    # except Exception as e:
    #     print(f"Error en fronteras de decisión: {e}")
    #
    # try:
    #     plot_rules_visualization(trained_classifiers, X, y, output_dir)
    # except Exception as e:
    #     print(f"Error en visualización de reglas: {e}")

    print(f"Visualizaciones guardadas en '{output_dir}'")


def plot_accuracy_comparison(results_df, output_dir):
    """Crea gráfico comparativo de exactitud para cada clasificador."""
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Ordenar por exactitud descendente
    sorted_results = results_df.sort_values('Exactitud', ascending=False)

    # Crear paleta de colores
    colors = sns.color_palette("viridis", len(sorted_results))

    # Crear gráfico de barras con texto integrado
    bars = plt.bar(sorted_results['Modelo'], sorted_results['Exactitud'], color=colors)

    # Añadir etiquetas con valores de exactitud
    for bar, accuracy in zip(bars, sorted_results['Exactitud']):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 accuracy + 0.01,
                 f'{accuracy:.3f}',
                 ha='center')

    plt.title('Comparación de Exactitud por Tipo de Representación', fontsize=15)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)  # Ajustar límites del eje y
    plt.tight_layout()

    # Guardar figura
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300)
    plt.close()


def plot_confusion_matrices(classifiers, X, y, output_dir):
    """Crea matrices de confusión para cada clasificador."""
    # Número de clasificadores
    n_classifiers = len(classifiers)
    n_cols = 2
    n_rows = (n_classifiers + 1) // n_cols

    plt.figure(figsize=(15, 5 * n_rows))

    # Iterar sobre cada clasificador
    for i, (name, clf) in enumerate(classifiers.items()):
        try:
            # Predecir clases
            y_pred = clf.predict(X)

            # Calcular matriz de confusión
            cm = confusion_matrix(y, y_pred)

            # Crear subplot
            plt.subplot(n_rows, n_cols, i + 1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Matriz de Confusión: {name}')
            plt.xlabel('Predicción')
            plt.ylabel('Real')
        except Exception as e:
            print(f"Error al crear matriz de confusión para {name}: {e}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300)
    plt.close()


def plot_roc_curves(classifiers, X, y, output_dir):
    """Crea las curvas ROC para clasificadores que generan probabilidades."""
    plt.figure(figsize=(10, 8))

    # Para cada clasificador
    for name, clf in classifiers.items():
        try:
            if hasattr(clf, 'predict_proba'):
                # Calcular probabilidades
                y_proba = clf.predict_proba(X)

                # Para problemas binarios
                if y_proba.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(y, y_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        except Exception as e:
            print(f"Error al crear curva ROC para {name}: {e}")

    # Línea de referencia (clasificador aleatorio)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Aleatorio')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curvas ROC por Representación')

    # Incluir leyenda - siempre tendremos al menos la línea de referencia
    plt.legend(loc="lower right")

    plt.grid(True)

    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300)
    plt.close()

def plot_time_vs_accuracy(results_df, output_dir):
    """Crea gráfico de dispersión tiempo vs. exactitud."""
    plt.figure(figsize=(10, 8))

    # Crear gráfico de dispersión
    plt.scatter(
        results_df['Tiempo (s)'],
        results_df['Exactitud'],
        c=np.arange(len(results_df)),
        cmap='viridis',
        s=100,
        alpha=0.7
    )

    # Añadir etiquetas para cada punto
    for i, model_name in enumerate(results_df['Modelo']):
        time_val = results_df['Tiempo (s)'].iloc[i]
        acc_val = results_df['Exactitud'].iloc[i]
        plt.annotate(model_name,
                     (time_val, acc_val),
                     xytext=(10, 5),
                     textcoords='offset points',
                     fontsize=9,
                     alpha=0.8)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Tiempo de ejecución (segundos)', fontsize=12)
    plt.ylabel('Exactitud', fontsize=12)
    plt.title('Compromiso entre Exactitud y Tiempo de Ejecución', fontsize=15)

    # Ajustar escala logarítmica si hay grandes diferencias en tiempos
    if results_df['Tiempo (s)'].max() / results_df['Tiempo (s)'].min() > 50:
        plt.xscale('log')
        plt.xlabel('Tiempo de ejecución (segundos, escala log)', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_vs_accuracy.png'), dpi=300)
    plt.close()


def plot_decision_boundaries(classifiers, X, y, output_dir, sample_size=500):
    """
    Visualiza fronteras de decisión después de reducir dimensionalidad.

    Parámetros:
    - classifiers: Diccionario con los clasificadores evaluados {nombre: instancia}
    - X: Datos de características originales
    - y: Etiquetas originales
    - output_dir: Directorio donde guardar las visualizaciones
    - sample_size: Tamaño máximo de la muestra para visualización
    """
    # Asegurarse de que X es 2D y y es 1D
    X = np.asarray(X)
    y = np.asarray(y)

    # Si X es 1D, convertirlo a 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Reducir dimensionalidad con PCA solo si tenemos más de 2 dimensiones
    if X.shape[1] > 2:
        pca = PCA(n_components=2)

        # Tomar muestra aleatoria si el conjunto es grande
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y

        # Aplicar PCA
        X_pca = pca.fit_transform(X_sample)
    else:
        # Si ya tenemos 2 o menos dimensiones, no aplicamos PCA
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_pca = X[indices]
            y_sample = y[indices]
        else:
            X_pca = X
            y_sample = y

    # Crear malla para visualizar fronteras
    x_min: float = float(X_pca[:, 0].min() - 1)
    x_max: float = float(X_pca[:, 0].max() + 1)
    y_min: float = float(X_pca[:, 1].min() - 1)
    y_max: float = float(X_pca[:, 1].max() + 1)
    h: float = 0.1

    xx: np.ndarray = np.arange(x_min, x_max, h)
    yy: np.ndarray = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xx, yy)

    # Para cada clasificador
    for name, clf in classifiers.items():
        try:
            plt.figure(figsize=(10, 8))

            # Entrenar modelo con datos reducidos
            clf_copy = clf.__class__(**clf.get_params())

            # Asegurarnos de que los datos están en el formato correcto para el entrenamiento
            if X_pca.ndim == 1:
                X_pca_2d = X_pca.reshape(-1, 1)
            else:
                X_pca_2d = X_pca

            clf_copy.fit(X_pca_2d, y_sample)

            # Predecir en la malla
            mesh_points: np.ndarray = np.c_[xx.ravel(), yy.ravel()]
            Z: np.ndarray = clf_copy.predict(mesh_points)
            Z = Z.reshape(xx.shape)

            # Dibujar frontera
            plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

            # Dibujar puntos
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_sample,
                                  edgecolors='k', s=50, cmap='viridis')

            # Añadir colorbar
            plt.colorbar(scatter, label='Clase')

            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.title(f'Frontera de decisión: {name}')
            plt.xlabel('Componente Principal 1' if X.shape[1] > 2 else 'Característica 1')
            plt.ylabel('Componente Principal 2' if X.shape[1] > 2 else 'Característica 2')

            plt.savefig(os.path.join(output_dir, f'decision_boundary_{name}.png'), dpi=300)
            plt.close()

        except Exception as e:
            print(f"Error al crear frontera de decisión para {name}: {e}")
            # Opcionalmente, grabar un error log o una imagen de error
            try:
                plt.figure(figsize=(10, 8))
                plt.text(0.5, 0.5, f"Error al crear frontera de decisión:\n{str(e)}",
                         ha='center', va='center', fontsize=12,
                         bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, f'decision_boundary_error_{name}.png'), dpi=300)
                plt.close()
            except Exception as e:
                print(f"Error al crear frontera de error para {name}: {e}")
                pass  # En caso de que ni siquiera podamos hacer esto


def plot_rules_visualization(classifiers, X, y, output_dir):
    """
    Visualiza reglas para clasificadores interpretables.
    """
    # Mapeo de tipos de clasificadores a funciones de visualización específicas
    visualizers = {
        'IntervalClassifier': _visualize_interval_rules,  # Usa X, y
        'HyperEllipsoidClassifier': _visualize_hyperellipsoid_rules,  # Usa X, y
        'FuzzyLogicClassifier': lambda c, data, labels, n, dir_path: _visualize_fuzzy_rules(c, data, n, dir_path),
        'UnorderedClassifier': lambda c, data, labels, n, dir_path: _visualize_unordered_rules(c, n, dir_path),
        'ExpressionsSClassifier': lambda c, data, labels, n, dir_path: _visualize_expression_rules(c, n, dir_path),
        'GeneExpressionsClassifier': lambda c, data, labels, n, dir_path: _visualize_expression_rules(c, n, dir_path)
    }

    # Para cada clasificador
    for name, clf in classifiers.items():
        clf_type = clf.__class__.__name__

        # Si tenemos un visualizador específico para este tipo
        if clf_type in visualizers:
            try:
                visualizers[clf_type](clf, X, y, name, output_dir)
            except Exception as e:
                print(f"Error al visualizar reglas para {name}: {e}")


# Función auxiliar para dibujar puntos de datos por clase
def _plot_data_points_by_class(X, y, top_features):
    """
    Dibuja los puntos de datos coloreados por clase en un gráfico.

    Parámetros:
    - X: Datos de características
    - y: Etiquetas de clase
    - top_features: Lista o array con los índices de las dos características a visualizar
    """
    # Dibujar puntos de datos por clase
    for class_val in np.unique(y):
        plt.scatter(X[y == class_val, top_features[0]],
                    X[y == class_val, top_features[1]],
                    alpha=0.5, label=f'Clase {class_val}')


def _save_figure(output_dir, filename, dpi=300):
    """
    Guarda la figura actual y la cierra.

    Parámetros:
    - output_dir: Directorio donde guardar la figura
    - filename: Nombre base del archivo (sin extensión)
    - dpi: Resolución de la imagen
    """
    plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=dpi)
    plt.close('all')  # Cerrar todas las figuras para evitar warnings de memoria

def _visualize_interval_rules(clf, X, y, name, output_dir):
    """Visualiza reglas basadas en intervalos."""
    if not hasattr(clf, 'rules_') or not clf.rules_:
        return

    # Seleccionar dos características más importantes
    variance = np.var(X, axis=0)
    top_features = np.argsort(variance)[-2:]

    plt.figure(figsize=(10, 8))

    # Usar la función auxiliar para dibujar puntos
    _plot_data_points_by_class(X, y, top_features)

    # Dibujar intervalos para cada clase
    for i, rule in enumerate(clf.rules_):
        # Dibujar rectángulo directamente con coordenadas
        x_min = rule['lower'][top_features[0]]
        y_min = rule['lower'][top_features[1]]
        width = rule['upper'][top_features[0]] - x_min
        height = rule['upper'][top_features[1]] - y_min

        rect = Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor=f'C{i}', facecolor='none',
            label=f"Regla Clase {rule['class']}"
        )
        plt.gca().add_patch(rect)

    plt.xlabel(f'Característica {top_features[0]}')
    plt.ylabel(f'Característica {top_features[1]}')
    plt.title(f'Visualización de Reglas por Intervalos - {name}')
    plt.legend()

    # Guardar y cerrar
    _save_figure(output_dir, f'intervals_rules_{name}')


def _visualize_hyperellipsoid_rules(clf, X, y, name, output_dir):
    """Visualiza reglas basadas en hiperelipsoides."""
    if not hasattr(clf, 'ellipsoids_') or not clf.ellipsoids_:
        return

    # Seleccionar dos características más importantes
    variance = np.var(X, axis=0)
    top_features = np.argsort(variance)[-2:]

    plt.figure(figsize=(10, 8))

    # Usar la función auxiliar para dibujar puntos
    _plot_data_points_by_class(X, y, top_features)

    # Dibujar elipses para cada regla
    for i, ellipsoid in enumerate(clf.ellipsoids_):
        center = ellipsoid['center'][[top_features[0], top_features[1]]]

        # Si es fallback (esfera), dibujar círculo
        if 'fallback' in ellipsoid and ellipsoid['fallback']:
            circle = Circle(
                center,
                ellipsoid['threshold'],
                fill=False,
                edgecolor=f'C{i}',
                linewidth=2,
                label=f"Regla Clase {ellipsoid['class']}"
            )
            plt.gca().add_artist(circle)
        else:
            # Extraer submatriz 2x2 para las características seleccionadas
            inv_cov_2d = ellipsoid['inv_cov'][np.ix_(top_features, top_features)]

            # Calcular eigenvectores y eigenvalores
            eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(inv_cov_2d))

            # Calcular ángulo con manejo de errores
            try:
                angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
            except (IndexError, TypeError):
                angle = 0.0  # Ángulo por defecto

            # Crear elipse
            ellipse = Ellipse(
                xy=center,
                width=float(2 * np.sqrt(eigenvalues[0]) * ellipsoid['threshold']),
                height=float(2 * np.sqrt(eigenvalues[1]) * ellipsoid['threshold']),
                angle=angle,
                fill=False,
                edgecolor=f'C{i}',
                linewidth=2,
                label=f"Regla Clase {ellipsoid['class']}"
            )

            plt.gca().add_artist(ellipse)

    plt.xlabel(f'Característica {top_features[0]}')
    plt.ylabel(f'Característica {top_features[1]}')
    plt.title(f'Visualización de Reglas por Hiperelipsoides - {name}')
    plt.legend()

    # Guardar y cerrar
    _save_figure(output_dir, f'hyperellipsoid_rules_{name}')


def _visualize_fuzzy_rules(clf, X, name, output_dir):
    """
    Visualiza reglas basadas en lógica difusa.

    Parámetros:
    - clf: El clasificador con atributo bin_edges_
    - X: Datos de características (necesarios para calcular varianza)
    - name: Nombre del clasificador
    - output_dir: Directorio de salida para guardar visualizaciones
    """
    if not hasattr(clf, 'bin_edges_') or not clf.bin_edges_:
        return

    # Visualizar pertenencia difusa para características importantes
    variance = np.var(X, axis=0)
    top_feature = np.argmax(variance)

    plt.figure(figsize=(10, 6))

    # Crear rango x
    x_range = np.linspace(
        np.min(X[:, top_feature]) - 0.1,
        np.max(X[:, top_feature]) + 0.1,
        1000
    )

    # Para cada conjunto difuso
    edges = clf.bin_edges_[top_feature]
    for j in range(clf.n_bins):
        # Usar el método de la clase para obtener parámetros
        a, b, c = clf.get_fuzzy_parameters(edges, j)

        # Calcular pertenencia para cada x usando el método de la clase
        y_vals = [clf.calculate_membership(xi, a, b, c) for xi in x_range]

        plt.plot(x_range, y_vals, label=f'Conjunto difuso {j + 1}')

    # Histograma de datos reales
    plt.hist(X[:, top_feature], bins=15, alpha=0.2, density=True, color='gray')

    plt.xlabel(f'Característica {top_feature}')
    plt.ylabel('Grado de pertenencia')
    plt.title(f'Funciones de pertenencia difusa - {name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Guardar y cerrar figura
    _save_figure(output_dir, f'fuzzy_rules_{name}')


def _visualize_expression_rules(clf, name, output_dir):
    """
    Visualiza reglas basadas en expresiones simbólicas o de genes.

    Parámetros:
    - clf: El clasificador con atributo program_str
    - name: Nombre del clasificador
    - output_dir: Directorio de salida para guardar visualizaciones
    """
    if not hasattr(clf, 'program_str') or not clf.program_str:
        return

    # Crear visualización textual de la expresión
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, clf.program_str, fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.axis('off')
    plt.title(f'Expresión simbólica - {name}')

    # Guardar y cerrar
    _save_figure(output_dir, f'expression_rules_{name}')


def _visualize_unordered_rules(clf, name, output_dir):
    """Visualiza reglas desordenadas evolutivas."""

    # Verificar si es implementación evolutiva o fallback
    if hasattr(clf, 'using_fallback') and clf.using_fallback:
        # Comportamiento para fallback RandomForest
        if hasattr(clf.evolution_engine, 'feature_importances_'):
            importances = clf.evolution_engine.feature_importances_
        else:
            return
    else:
        # Comportamiento para cromosomas desordenados evolutivos
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        else:
            return

    # Visualizar importancias (código existente compatible)
    plt.figure(figsize=(12, 6))
    indices = np.argsort(importances)[::-1]
    n_features = min(10, len(indices))

    plt.bar(range(n_features), importances[indices[:n_features]])
    plt.xticks(range(n_features), indices[:n_features])
    plt.xlabel('Índice de característica')
    plt.ylabel('Frecuencia en cromosomas')
    plt.title(f'Genes más frecuentes en cromosomas desordenados - {name}')
    plt.grid(True, linestyle='--', alpha=0.7)

    _save_figure(output_dir, f'unordered_rules_{name}')

    # Visualización adicional: estructura de cromosomas
    if hasattr(clf, 'class_rules') and not clf.using_fallback:
        plt.figure(figsize=(14, 8))

        class_labels = list(clf.class_rules.keys())
        chromosome_lengths = [rule.length() for rule in clf.class_rules.values()]
        fitness_scores = [rule.fitness for rule in clf.class_rules.values()]

        # Subplot 1: Longitudes de cromosomas
        plt.subplot(2, 1, 1)
        plt.bar(range(len(class_labels)), chromosome_lengths, color='skyblue')
        plt.xticks(range(len(class_labels)), [f'Clase {c}' for c in class_labels])
        plt.ylabel('Número de genes')
        plt.title('Longitud de cromosomas desordenados por clase')
        plt.grid(True, alpha=0.3)

        # Subplot 2: Fitness por clase
        plt.subplot(2, 1, 2)
        plt.bar(range(len(class_labels)), fitness_scores, color='lightcoral')
        plt.xticks(range(len(class_labels)), [f'Clase {c}' for c in class_labels])
        plt.ylabel('Fitness evolutivo')
        plt.title('Fitness de cromosomas evolucionados')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        _save_figure(output_dir, f'unordered_chromosomes_{name}')