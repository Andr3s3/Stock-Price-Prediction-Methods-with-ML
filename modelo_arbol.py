#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modelo_arbol.py
===============
Árbol de Decisión: dos árboles en el mismo modelo.

  1. DecisionTreeRegressor (multi-salida) para el pronóstico: predice el
     rendimiento de cada día del horizonte (1..N) a partir de las features.
  2. DecisionTreeClassifier de 3 clases para las señales:
         BAJA (< -3 %)   MANTENER   SUBE (> +3 %)
     De sus probabilidades salen p_sube y p_baja -> COMPRAR / MANTENER / VENDER.

Por qué no un solo regresor para todo: el regresor predice el rendimiento
PROMEDIO de las situaciones parecidas, que casi nunca cruza el +-3 %. El
clasificador aprende directamente "en qué situaciones subió o bajó más de 3 %"
y da probabilidades.

Diferencias respecto al notebook RLineal-DTree.ipynb:
  - Se predice rendimiento, no precio: un árbol nunca puede predecir un valor
    mayor al máximo que vio en el entrenamiento; con rendimientos sí funciona
    aunque el precio actual sea récord.
  - 47 features (todos los indicadores como proporciones) en lugar de solo Close.
  - Validación walk-forward en orden cronológico (train_test_split mezclaba días
    y el árbol entrenaba con el futuro).
  - Profundidad y tamaño mínimo de hoja limitados y elegidos en validación
    (un árbol sin límite memoriza el entrenamiento).

No necesita escalado: los árboles comparan cada variable contra un umbral.

Para configurar los árboles edita la SECCIÓN 1 (modo manual o automático,
parámetros de cada árbol, variables a usar y niveles a dibujar).

Uso:
    python3 modelo_arbol.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree, export_text

from motor_prediccion import ejecutar_modelo, FEATURES_EXTENDIDAS, FEATURES_MODELO, SEED

# =====================================================================
# SECCIÓN 1. CONFIGURACIÓN  <-- AQUÍ SE EDITAN LOS ÁRBOLES
# =====================================================================
#
# MODO_CONFIGURACION
#   'manual' : se usan exactamente ARBOL_REGRESOR y ARBOL_CLASIFICADOR.
#   'auto'   : el motor prueba todas las combinaciones de OPCIONES_REGRESOR y
#              OPCIONES_CLASIFICADOR en el conjunto de validación de cada
#              periodo y se queda con la mejor (menor MAE / mayor AUC).
MODO_CONFIGURACION = 'manual'

# ---------------------------------------------------------------------
# 1A. CONFIGURACIÓN MANUAL
# ---------------------------------------------------------------------
# Árbol REGRESOR (pronóstico del precio)
ARBOL_REGRESOR = {
    'max_depth': 5,               # niveles de preguntas (None = sin límite)
    'min_samples_leaf': 50,       # mínimo de días por hoja
    'min_samples_split': 2,       # mínimo de días para dividir un nodo
    'max_leaf_nodes': None,       # máximo de hojas (None = sin límite)
    'max_features': None,         # variables por división: None (todas), 'sqrt', 0.5
    'ccp_alpha': 0.0,             # poda: > 0 elimina ramas que aportan poco (ej. 0.0005)
    'criterion': 'squared_error', # 'squared_error' o 'absolute_error' (más lento)
}

# Árbol CLASIFICADOR (señales COMPRAR / MANTENER / VENDER)
ARBOL_CLASIFICADOR = {
    'max_depth': 4,
    'min_samples_leaf': 50,
    'min_samples_split': 2,
    'max_leaf_nodes': None,
    'max_features': None,
    'ccp_alpha': 0.0,
    'criterion': 'gini',          # 'gini', 'entropy' o 'log_loss'
    'class_weight': 'balanced',   # 'balanced' compensa clases desiguales; None = sin compensar
}

# ---------------------------------------------------------------------
# 1B. CONFIGURACIÓN AUTOMÁTICA (solo si MODO_CONFIGURACION = 'auto')
# ---------------------------------------------------------------------
# Cada diccionario es una configuración candidata. Los parámetros que no se
# indiquen toman el valor de ARBOL_REGRESOR / ARBOL_CLASIFICADOR.
OPCIONES_REGRESOR = [{'max_depth': d, 'min_samples_leaf': h}
                     for d in (3, 5, 8) for h in (20, 50, 100)]
OPCIONES_CLASIFICADOR = [{'max_depth': d, 'min_samples_leaf': h}
                         for d in (3, 4, 6) for h in (50, 100)]

# ---------------------------------------------------------------------
# 1C. VARIABLES Y GRÁFICAS
# ---------------------------------------------------------------------
# FEATURES_EXTENDIDAS (47 variables), FEATURES_MODELO (22) o tu propia lista,
# por ejemplo: ['Dist_MA200', 'MA50_vs_MA200', 'RSI', 'ATR_pct', 'Volatilidad20']
FEATURES = FEATURES_EXTENDIDAS

NIVELES_DIBUJO = 3                # niveles del árbol que se dibujan en la Gráfica 7

CLASES = ['BAJA', 'MANTENER', 'SUBE']      # etiquetas -1, 0, 1


def _opciones(base, candidatas):
    """Lista de configuraciones que recibe el motor según el modo elegido."""
    if MODO_CONFIGURACION == 'manual':
        return [dict(base)]
    if MODO_CONFIGURACION == 'auto':
        return [{**base, **c} for c in candidatas]
    raise ValueError("MODO_CONFIGURACION debe ser 'manual' o 'auto'")


# =====================================================================
# SECCIÓN 2. ÁRBOL REGRESOR (pronóstico de precio)
# =====================================================================
# Predice el rendimiento logarítmico de cada día del horizonte (1..N) con un
# solo árbol multi-salida. El motor convierte rendimiento -> precio.
def entrenar_regresor(X, Y_dias, params):
    return DecisionTreeRegressor(random_state=SEED, **params).fit(X, Y_dias)


def predecir_regresor(regresor, X):
    """Devuelve un array (filas, días) con el rendimiento de cada día."""
    return regresor.predict(X).reshape(len(X), -1)     # reshape: también si N = 1


# =====================================================================
# SECCIÓN 3. ÁRBOL CLASIFICADOR (señales COMPRAR / MANTENER / VENDER)
# =====================================================================
# Un solo árbol de 3 clases: BAJA (< -3 %), MANTENER, SUBE (> +3 %).
# De sus probabilidades salen p_sube y p_baja; el motor decide la señal.
def entrenar_clasificador(X, y_sube, y_baja, params):
    y3 = (y_sube - y_baja).astype(int)                 # -1 baja, 0 se queda, 1 sube
    if y3.nunique() < 2:
        return None
    return DecisionTreeClassifier(random_state=SEED, **params).fit(X, y3)


def predecir_clasificador(clasificador, X):
    """Devuelve (p_sube, p_baja), cada uno un array de longitud len(X)."""
    if clasificador is None:
        nan = np.full(len(X), np.nan)
        return nan, nan
    proba = clasificador.predict_proba(X)
    clases = list(clasificador.classes_)
    p_sube = proba[:, clases.index(1)] if 1 in clases else np.zeros(len(X))
    p_baja = proba[:, clases.index(-1)] if -1 in clases else np.zeros(len(X))
    return p_sube, p_baja


# =====================================================================
# SECCIÓN 4. CONEXIÓN CON EL MOTOR (une regresor + clasificador)
# =====================================================================
def entrenar(X, Y_dias, y_sube, y_baja, p_precio, p_senal):
    return {'regresor': entrenar_regresor(X, Y_dias, p_precio),
            'clasificador': entrenar_clasificador(X, y_sube, y_baja, p_senal)}


def predecir(modelo, X):
    rend = predecir_regresor(modelo['regresor'], X)
    p_sube, p_baja = predecir_clasificador(modelo['clasificador'], X)
    return rend, p_sube, p_baja


# =====================================================================
# SECCIÓN 5. GRÁFICAS PROPIAS DEL ÁRBOL
# =====================================================================
def imprimir_parametros(modelo):
    """Parámetros con los que se entrenaron los árboles finales y su tamaño real."""
    claves = ['max_depth', 'min_samples_leaf', 'min_samples_split', 'max_leaf_nodes',
              'max_features', 'ccp_alpha', 'criterion', 'class_weight']
    print("PARÁMETROS DE LOS ÁRBOLES FINALES (los que generan el pronóstico):")
    for nombre, arbol in [('Regresor', modelo['regresor']), ('Clasificador', modelo['clasificador'])]:
        if arbol is None:
            print(f"  {nombre}: no se entrenó (una sola clase en los datos)")
            continue
        params = arbol.get_params()
        print(f"\n  {nombre}:")
        for k in claves:
            if k in params:
                print(f"    {k:<18} {params[k]}")
        print(f"    {'profundidad real':<18} {arbol.get_depth()}")
        print(f"    {'hojas':<18} {arbol.get_n_leaves()}")
    print()


def graficas_propias(modelo, ctx):
    reg, clf = modelo['regresor'], modelo['clasificador']
    imprimir_parametros(modelo)

    # ---------- Gráfica 6: importancia de variables ----------
    print(f"GRÁFICA 6: {ctx.ticker} - Importancia de variables")
    imp = pd.DataFrame({
        'Regresor (pronóstico)': reg.feature_importances_,
        'Clasificador (señales)': clf.feature_importances_ if clf is not None else np.nan,
    }, index=ctx.features)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    for ax, col, color in zip(axes, imp.columns, ['steelblue', 'darkorange']):
        top = imp[col].sort_values(ascending=False).head(15)[::-1]
        ax.barh(top.index, top.values, color=color, alpha=0.8)
        for y, v in enumerate(top.values):
            ax.text(v, y, f' {v:.3f}', va='center', fontsize=9)
        ax.set_title(f'{col}: 15 variables más importantes')
        ax.set_xlabel('Importancia (suma = 1)')
        ax.grid(True, alpha=0.3, axis='x')
    plt.suptitle(f'{ctx.ticker} - Importancia de variables en los árboles', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.show(block=False); plt.pause(0.1)

    sin_uso = imp.index[(imp.fillna(0) == 0).all(axis=1)].tolist()
    print(f"  Variables usadas por el regresor: {(reg.feature_importances_ > 0).sum()} de {len(ctx.features)}")
    if clf is not None:
        print(f"  Variables usadas por el clasificador: {(clf.feature_importances_ > 0).sum()} de {len(ctx.features)}")
    if sin_uso:
        print(f"  Sin uso en ningún árbol: {', '.join(sin_uso)}")

    # ---------- Gráfica 7: árbol de señales ----------
    if clf is None:
        return
    print(f"\nGRÁFICA 7: {ctx.ticker} - Árbol de decisión de señales (primeros {NIVELES_DIBUJO} niveles)")
    nombres = [CLASES[c + 1] for c in clf.classes_]
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(clf, max_depth=NIVELES_DIBUJO, feature_names=ctx.features, class_names=nombres, filled=True,
              rounded=True, impurity=False, proportion=True, precision=3, fontsize=9, ax=ax)
    ax.set_title(f'{ctx.ticker} - Árbol de señales (profundidad total {clf.get_depth()}, '
                 f'{clf.get_n_leaves()} hojas; se muestran {NIVELES_DIBUJO} niveles)', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.show(block=False); plt.pause(0.1)

    print(f"\nREGLAS DEL ÁRBOL DE SEÑALES (primeros {NIVELES_DIBUJO} niveles):")
    try:
        reglas = export_text(clf, feature_names=list(ctx.features), max_depth=NIVELES_DIBUJO, decimals=3,
                             class_names=nombres)
    except TypeError:                       # scikit-learn < 1.3 no acepta class_names
        reglas = export_text(clf, feature_names=list(ctx.features), max_depth=NIVELES_DIBUJO, decimals=3)
    print(reglas)


# =====================================================================
# SECCIÓN 6. EJECUCIÓN
# =====================================================================
if __name__ == '__main__':
    ejecutar_modelo(
        nombre="Árbol de Decisión",
        entrenar=entrenar,
        predecir=predecir,
        nombre_precio="Árbol de Decisión Regresor",
        nombre_senal="Árbol de Decisión Clasificador",
        opciones_precio=_opciones(ARBOL_REGRESOR, OPCIONES_REGRESOR),
        opciones_senal=_opciones(ARBOL_CLASIFICADOR, OPCIONES_CLASIFICADOR),
        graficas_propias=graficas_propias,
        features=FEATURES,
        tres_clases=True,          # señal = clase más probable de BAJA / MANTENER / SUBE
    )
