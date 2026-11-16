#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modelo_random_forest.py
=======================
Random Forest: dos bosques en el mismo modelo.

  1. RandomForestRegressor (multi-salida) para el pronóstico: predice el
     rendimiento de cada día del horizonte (1..N). Cada árbol da su predicción
     y el bosque las promedia.
  2. RandomForestClassifier de 3 clases para las señales:
         BAJA (< -3 %)   MANTENER   SUBE (> +3 %)
     Cada árbol vota; las probabilidades son la proporción de votos.

Diferencia con el árbol de decisión: en lugar de un solo árbol se entrenan
cientos, cada uno con una muestra distinta de días (max_samples) y, en cada
división, con un subconjunto de variables (max_features). El promedio de
todos es mucho más estable que un árbol individual.

No se usa oob_score para elegir configuraciones: los días consecutivos tienen
respuestas casi idénticas y el error fuera de bolsa sale demasiado optimista.

Para configurar los bosques edita la SECCIÓN 1.

Uso:
    python3 modelo_random_forest.py
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, balanced_accuracy_score, roc_auc_score

from motor_prediccion import (ejecutar_modelo, FEATURES_EXTENDIDAS, FEATURES_MODELO, SEED,
                              decidir_senal)


# =====================================================================
# SECCIÓN 1. CONFIGURACIÓN  <-- AQUÍ SE EDITAN LOS BOSQUES
# =====================================================================
#
# MODO_CONFIGURACION
#   'manual' : se usan exactamente BOSQUE_REGRESOR y BOSQUE_CLASIFICADOR.
#   'auto'   : el motor prueba las combinaciones de OPCIONES_REGRESOR y
#              OPCIONES_CLASIFICADOR en validación y se queda con la mejor.
#              Tarda más: cada combinación se entrena en cada periodo.
MODO_CONFIGURACION = 'manual'

# ---------------------------------------------------------------------
# 1A. CONFIGURACIÓN MANUAL
# ---------------------------------------------------------------------
# Bosque REGRESOR (pronóstico del precio)
BOSQUE_REGRESOR = {
    'n_estimators': 300,          # número de árboles
    'max_depth': 8,               # niveles de cada árbol (None = sin límite)
    'min_samples_leaf': 40,       # mínimo de días por hoja
    'min_samples_split': 2,       # mínimo de días para dividir un nodo
    'max_features': 0.3,          # % de variables por división: 0.3, 0.5, 'sqrt', None (todas)
    'max_samples': 0.5,           # % de días que ve cada árbol (None = todos, con reemplazo)
    'bootstrap': True,            # muestreo con reemplazo (necesario para max_samples)
    'ccp_alpha': 0.0,             # poda de cada árbol
    'criterion': 'squared_error',
    'n_jobs': -1,                 # núcleos del procesador (-1 = todos)
}

# Bosque CLASIFICADOR (señales COMPRAR / MANTENER / VENDER)
BOSQUE_CLASIFICADOR = {
    'n_estimators': 300,
    'max_depth': 6,
    'min_samples_leaf': 40,
    'min_samples_split': 2,
    'max_features': 0.3,
    'max_samples': 0.5,
    'bootstrap': True,
    'ccp_alpha': 0.0,
    'criterion': 'gini',          # 'gini', 'entropy' o 'log_loss'
    'class_weight': 'balanced_subsample',   # compensa clases dentro de la muestra de cada árbol
    'n_jobs': -1,
}

# ---------------------------------------------------------------------
# 1B. CONFIGURACIÓN AUTOMÁTICA (solo si MODO_CONFIGURACION = 'auto')
# ---------------------------------------------------------------------
# Los parámetros que no se indiquen toman el valor de 1A. Cuadrícula pequeña
# a propósito: 4 combinaciones por bosque ya tardan 1-3 minutos.
OPCIONES_REGRESOR = [{'max_depth': d, 'max_features': f}
                     for d in (6, 10) for f in (0.3, 0.5)]
OPCIONES_CLASIFICADOR = [{'max_depth': d, 'min_samples_leaf': h}
                         for d in (4, 8) for h in (30, 60)]

# ---------------------------------------------------------------------
# 1C. VARIABLES Y GRÁFICAS
# ---------------------------------------------------------------------
FEATURES = FEATURES_EXTENDIDAS    # 47 variables; FEATURES_MODELO = 22; o una lista propia

PERMUTACION_REPETICIONES = 5      # veces que se revuelve cada variable (más = más estable, más lento)
ARBOLES_CURVA = [1, 5, 10, 25, 50, 100, 200, 300]   # puntos de la gráfica de número de árboles

CLASES = ['BAJA', 'MANTENER', 'SUBE']      # etiquetas -1, 0, 1


def _opciones(base, candidatas):
    """Lista de configuraciones que recibe el motor según el modo elegido."""
    if MODO_CONFIGURACION == 'manual':
        return [dict(base)]
    if MODO_CONFIGURACION == 'auto':
        return [{**base, **c} for c in candidatas]
    raise ValueError("MODO_CONFIGURACION debe ser 'manual' o 'auto'")


# =====================================================================
# SECCIÓN 2. BOSQUE REGRESOR (pronóstico de precio)
# =====================================================================
def entrenar_regresor(X, Y_dias, params):
    return RandomForestRegressor(random_state=SEED, **params).fit(X, Y_dias)


def predecir_regresor(regresor, X):
    """Devuelve un array (filas, días) con el rendimiento de cada día."""
    return regresor.predict(X).reshape(len(X), -1)


# =====================================================================
# SECCIÓN 3. BOSQUE CLASIFICADOR (señales COMPRAR / MANTENER / VENDER)
# =====================================================================
def entrenar_clasificador(X, y_sube, y_baja, params):
    y3 = (y_sube - y_baja).astype(int)                 # -1 baja, 0 se queda, 1 sube
    if y3.nunique() < 2:
        return None
    return RandomForestClassifier(random_state=SEED, **params).fit(X, y3)


def predecir_clasificador(clasificador, X):
    """Devuelve (p_sube, p_baja): proporción de árboles que votan SUBE / BAJA."""
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
            'clasificador': entrenar_clasificador(X, y_sube, y_baja, p_senal),
            'p_precio': p_precio, 'p_senal': p_senal}


def predecir(modelo, X):
    rend = predecir_regresor(modelo['regresor'], X)
    p_sube, p_baja = predecir_clasificador(modelo['clasificador'], X)
    return rend, p_sube, p_baja


# =====================================================================
# SECCIÓN 5. GRÁFICAS PROPIAS DEL BOSQUE
# =====================================================================
def imprimir_parametros(modelo):
    """Parámetros con los que se entrenaron los bosques finales y su tamaño real."""
    claves = ['n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split',
              'max_features', 'max_samples', 'bootstrap', 'ccp_alpha', 'criterion', 'class_weight']
    print("PARÁMETROS DE LOS BOSQUES FINALES (los que generan el pronóstico):")
    for nombre, bosque in [('Regresor', modelo['regresor']), ('Clasificador', modelo['clasificador'])]:
        if bosque is None:
            print(f"  {nombre}: no se entrenó (una sola clase en los datos)")
            continue
        params = bosque.get_params()
        print(f"\n  {nombre}:")
        for k in claves:
            if k in params:
                print(f"    {k:<20} {params[k]}")
        prof = [a.get_depth() for a in bosque.estimators_]
        hojas = [a.get_n_leaves() for a in bosque.estimators_]
        print(f"    {'profundidad real':<20} promedio {np.mean(prof):.1f} (mín {min(prof)}, máx {max(prof)})")
        print(f"    {'hojas por árbol':<20} promedio {np.mean(hojas):.1f}")
    print()


def _modelo_periodo_reciente(modelo, ctx):
    """Bosques entrenados solo con el entrenamiento del periodo de prueba más
    reciente, con la configuración final. Se usan para la importancia por
    permutación y la curva de número de árboles: evaluar el modelo final en esos
    días sería hacer trampa, porque los vio al entrenar."""
    d, f_tr = ctx.d, ctx.f_train
    return entrenar(d.X.loc[f_tr], d.Y.loc[f_tr], d.y_sube.loc[f_tr], d.y_baja.loc[f_tr],
                    modelo['p_precio'], modelo['p_senal'])


def _grafica_importancia(modelo, m_rec, ctx):
    print(f"GRÁFICA 6: {ctx.ticker} - Importancia de variables")
    d, f_te = ctx.d, ctx.f_test
    X_te = d.X.loc[f_te]
    y_rend = d.Y.loc[f_te].values[:, -1]
    y3 = (d.y_sube.loc[f_te] - d.y_baja.loc[f_te]).values

    # Puntuaciones: más alto = mejor (el MAE va con signo negativo)
    def score_reg(est, X, y):
        return -mean_absolute_error(y, est.predict(X).reshape(len(X), -1)[:, -1])

    def score_clf(est, X, y):
        return balanced_accuracy_score(y, est.predict(X))

    t0 = time.time()
    perm_reg = permutation_importance(m_rec['regresor'], X_te, y_rend, scoring=score_reg,
                                      n_repeats=PERMUTACION_REPETICIONES, random_state=SEED, n_jobs=1)
    perm_clf = None
    if m_rec['clasificador'] is not None and len(np.unique(y3)) >= 2:
        perm_clf = permutation_importance(m_rec['clasificador'], X_te, y3, scoring=score_clf,
                                          n_repeats=PERMUTACION_REPETICIONES, random_state=SEED, n_jobs=1)
    print(f"  Importancia por permutación calculada en {time.time() - t0:.1f} s "
          f"(periodo {f_te[0].date()} a {f_te[-1].date()})")

    paneles = [
        ('Regresor: importancia del bosque', modelo['regresor'].feature_importances_, None, 'steelblue'),
        ('Regresor: permutación (aumento del MAE)', perm_reg.importances_mean, perm_reg.importances_std, 'navy'),
    ]
    if modelo['clasificador'] is not None:
        paneles.append(('Clasificador: importancia del bosque',
                        modelo['clasificador'].feature_importances_, None, 'darkorange'))
    if perm_clf is not None:
        paneles.append(('Clasificador: permutación (caída del accuracy balanceado)',
                        perm_clf.importances_mean, perm_clf.importances_std, 'saddlebrown'))

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    for ax in axes.flat[len(paneles):]:
        ax.axis('off')
    for ax, (titulo, valores, error, color) in zip(axes.flat, paneles):
        serie = pd.Series(valores, index=ctx.features)
        top = serie.sort_values(ascending=False).head(15)[::-1]
        err = pd.Series(error, index=ctx.features).loc[top.index] if error is not None else None
        ax.barh(top.index, top.values, xerr=err, color=color, alpha=0.8, capsize=3)
        for y, v in enumerate(top.values):
            ax.text(v, y, f' {v:.4f}', va='center', fontsize=8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(titulo, fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
    plt.suptitle(f'{ctx.ticker} - Importancia de variables (15 principales)\n'
                 f'Permutación: cuánto empeora el modelo al revolver cada variable en el periodo de prueba '
                 f'más reciente. Valores cerca de 0 o negativos = la variable no ayuda.',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.show(block=False); plt.pause(0.1)

    utiles = (pd.Series(perm_reg.importances_mean, index=ctx.features) > 0).sum()
    print(f"  Variables que mejoran el pronóstico fuera del entrenamiento (permutación > 0): "
          f"{utiles} de {len(ctx.features)}")
    if perm_clf is not None:
        utiles_c = (pd.Series(perm_clf.importances_mean, index=ctx.features) > 0).sum()
        print(f"  Variables que mejoran las señales fuera del entrenamiento: {utiles_c} de {len(ctx.features)}")


def _grafica_acuerdo(modelo, ctx):
    print(f"\nGRÁFICA 7: {ctx.ticker} - Acuerdo entre árboles para la predicción de hoy")
    df = ctx.d.df
    ultima = df[ctx.features].dropna().iloc[[-1]]
    fecha = ultima.index[0]
    precio_actual = float(df.loc[fecha, 'Close'])
    Xu = ultima.values

    reg = modelo['regresor']
    rend_arboles = np.array([a.predict(Xu).reshape(1, -1)[0, -1] for a in reg.estimators_])
    precios = precio_actual * np.exp(rend_arboles)
    precio_bosque = precio_actual * np.exp(predecir_regresor(reg, ultima)[0, -1])
    p10, p50, p90 = np.percentile(precios, [10, 50, 90])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})
    ax = axes[0]
    ax.hist(precios, bins=30, color='steelblue', alpha=0.75, edgecolor='black')
    ax.axvline(precio_actual, color='gray', linestyle='--', linewidth=2, label=f'Precio actual: {precio_actual:.2f}')
    ax.axvline(precio_bosque, color='red', linewidth=2.5, label=f'Predicción del bosque: {precio_bosque:.2f}')
    ax.axvspan(p10, p90, color='orange', alpha=0.15, label=f'80 % de los árboles: {p10:.2f} a {p90:.2f}')
    ax.set_title(f'Precio final que predice cada uno de los {len(precios)} árboles ({ctx.dias} días)')
    ax.set_xlabel(f'Precio ({ctx.moneda})'); ax.set_ylabel('Número de árboles')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    clf = modelo['clasificador']
    if clf is not None:
        clases = list(clf.classes_)
        votos = np.array([clases[int(np.argmax(a.predict_proba(Xu)[0]))] for a in clf.estimators_])
        conteo = {c: int((votos == v).sum()) for c, v in zip(CLASES, (-1, 0, 1))}
        total = len(votos)
        colores = ['red', 'gray', 'green']
        barras = ax.bar(list(conteo.keys()), [v / total * 100 for v in conteo.values()], color=colores, alpha=0.75)
        for b, v in zip(barras, conteo.values()):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f'{v / total:.0%}\n({v})',
                    ha='center', va='bottom', fontsize=10)
        ax.set_ylim(0, 115); ax.set_ylabel('% de árboles')
        p_sube, p_baja = predecir_clasificador(clf, ultima)
        senal = decidir_senal(float(p_sube[0]), float(p_baja[0]), tres_clases=True)
        mayor = max(conteo.values()) / total
        firmeza = 'firme' if mayor >= 0.7 else 'moderada' if mayor >= 0.5 else 'débil'
        ax.set_title(f'Votos del clasificador ({total} árboles)\nSeñal: {senal} (acuerdo {firmeza})')
        ax.grid(True, alpha=0.3, axis='y')
        print(f"  Votos: SUBE {conteo['SUBE']/total:.0%} | MANTENER {conteo['MANTENER']/total:.0%} | "
              f"BAJA {conteo['BAJA']/total:.0%}  -> acuerdo {firmeza}")
    else:
        ax.axis('off')

    plt.suptitle(f'{ctx.ticker} - Acuerdo entre árboles (datos al {fecha.date()})', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.show(block=False); plt.pause(0.1)
    print(f"  Precio final por árbol: mediana {p50:.2f}, 80 % entre {p10:.2f} y {p90:.2f} {ctx.moneda}")


def _grafica_numero_arboles(m_rec, ctx):
    print(f"\nGRÁFICA 8: {ctx.ticker} - Desempeño contra número de árboles")
    d, f_te = ctx.d, ctx.f_test
    X_te = d.X.loc[f_te].values
    precio_hoy = d.close.loc[f_te].values
    precio_real = d.precio_fut.loc[f_te].values
    mae_base = mean_absolute_error(precio_real, precio_hoy)

    reg = m_rec['regresor']
    pred_arboles = np.array([a.predict(X_te).reshape(len(X_te), -1)[:, -1] for a in reg.estimators_])
    ks = [k for k in ARBOLES_CURVA if k <= len(reg.estimators_)]
    maes = [mean_absolute_error(precio_real, precio_hoy * np.exp(pred_arboles[:k].mean(axis=0))) for k in ks]

    aucs = []
    clf = m_rec['clasificador']
    ys = d.y_sube.loc[f_te].values
    if clf is not None and 1 in clf.classes_ and len(np.unique(ys)) == 2:
        i_sube = list(clf.classes_).index(1)
        proba_arboles = np.array([a.predict_proba(X_te)[:, i_sube] for a in clf.estimators_])
        aucs = [roc_auc_score(ys, proba_arboles[:k].mean(axis=0)) for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(ks, maes, 'o-', color='steelblue', linewidth=2, label='MAE del bosque')
    axes[0].axhline(mae_base, color='gray', linestyle='--', label=f'Línea base: {mae_base:.2f}')
    axes[0].set_xscale('log'); axes[0].set_xticks(ks); axes[0].set_xticklabels(ks)
    axes[0].set_title('Regresor: MAE en el periodo de prueba más reciente')
    axes[0].set_xlabel('Número de árboles'); axes[0].set_ylabel(f'MAE ({ctx.moneda})')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    if aucs:
        axes[1].plot(ks, aucs, 'o-', color='purple', linewidth=2, label='AUC subida')
        axes[1].axhline(0.5, color='red', linestyle='--', label='Azar (0.5)')
        axes[1].set_xscale('log'); axes[1].set_xticks(ks); axes[1].set_xticklabels(ks)
        axes[1].set_title('Clasificador: AUC de subida en el periodo de prueba más reciente')
        axes[1].set_xlabel('Número de árboles'); axes[1].set_ylabel('ROC-AUC')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Sin subidas y no-subidas en el periodo:\nel AUC no se puede calcular',
                     ha='center', va='center', color='gray', fontsize=11)
        axes[1].axis('off')
    plt.suptitle(f'{ctx.ticker} - ¿Cuántos árboles hacen falta? (si la curva se aplana, más árboles no ayudan)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.show(block=False); plt.pause(0.1)

    print("  Árboles | MAE" + ("     | AUC subida" if aucs else ""))
    for i, k in enumerate(ks):
        print(f"  {k:>7} | {maes[i]:<7.2f}" + (f" | {aucs[i]:.3f}" if aucs else ""))


def graficas_propias(modelo, ctx):
    imprimir_parametros(modelo)
    print("Entrenando bosques del periodo de prueba más reciente para evaluar variables y árboles...")
    m_rec = _modelo_periodo_reciente(modelo, ctx)
    _grafica_importancia(modelo, m_rec, ctx)
    _grafica_acuerdo(modelo, ctx)
    _grafica_numero_arboles(m_rec, ctx)


# =====================================================================
# SECCIÓN 6. EJECUCIÓN
# =====================================================================
if __name__ == '__main__':
    ejecutar_modelo(
        nombre="Random Forest",
        entrenar=entrenar,
        predecir=predecir,
        nombre_precio="Random Forest Regresor",
        nombre_senal="Random Forest Clasificador",
        opciones_precio=_opciones(BOSQUE_REGRESOR, OPCIONES_REGRESOR),
        opciones_senal=_opciones(BOSQUE_CLASIFICADOR, OPCIONES_CLASIFICADOR),
        graficas_propias=graficas_propias,
        features=FEATURES,
        tres_clases=True,          # señal = clase más probable de BAJA / MANTENER / SUBE
    )
