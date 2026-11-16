#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modelo_xgboost.py
=================
XGBoost: dos modelos de boosting en el mismo archivo.

  1. XGBRegressor para el pronóstico: predice el rendimiento logarítmico al
     final del horizonte (día N). La trayectoria día por día (la línea oculta
     "desde hoy" de la Gráfica 5) se traza en línea recta hasta ese valor.
     El pronóstico rodante, la validación y las señales solo usan el día N,
     así que esto no cambia ninguna métrica y es mucho más rápido que entrenar
     un modelo por cada día del horizonte.
  2. XGBClassifier de 3 clases para las señales: BAJA (< -3 %), MANTENER,
     SUBE (> +3 %). La señal es la clase más probable.

Cómo funciona: a diferencia del Random Forest (árboles independientes que se
promedian), XGBoost entrena los árboles en cadena; cada árbol corrige los
errores de los anteriores. Es muy preciso pero memoriza con facilidad, por eso:
  - Parada temprana: se reserva el final del entrenamiento como validación
    (con un hueco de N días contra la fuga de datos). Se agregan árboles hasta
    que el error en validación deja de bajar durante EARLY_STOPPING rondas y
    se reentrena con todos los datos usando ese número de árboles.
  - Pesos por clase en el clasificador (equivalente a class_weight='balanced').

Requiere: xgboost >= 2.0   (conda install -c conda-forge xgboost)

Para configurar los modelos edita la SECCIÓN 1.

Uso:
    python3 modelo_xgboost.py
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from motor_prediccion import ejecutar_modelo, FEATURES_EXTENDIDAS, FEATURES_MODELO, SEED


# =====================================================================
# SECCIÓN 1. CONFIGURACIÓN  <-- AQUÍ SE EDITAN LOS MODELOS
# =====================================================================
#
# MODO_CONFIGURACION
#   'manual' : se usan exactamente XGB_REGRESOR y XGB_CLASIFICADOR.
#   'auto'   : el motor prueba las combinaciones de OPCIONES_REGRESOR y
#              OPCIONES_CLASIFICADOR en validación y se queda con la mejor.
MODO_CONFIGURACION = 'manual'

# ---------------------------------------------------------------------
# 1A. CONFIGURACIÓN MANUAL
# ---------------------------------------------------------------------
# REGRESOR (pronóstico del precio)
XGB_REGRESOR = {
    'n_estimators': 1000,         # máximo de árboles; la parada temprana elige cuántos usar
    'learning_rate': 0.03,        # cuánto corrige cada árbol (más bajo = más lento, memoriza menos)
    'max_depth': 4,               # niveles de cada árbol (en boosting se usan árboles poco profundos)
    'min_child_weight': 20,       # similar al mínimo de días por hoja
    'subsample': 0.7,             # % de días que ve cada árbol
    'colsample_bytree': 0.5,      # % de variables que ve cada árbol
    'reg_lambda': 5.0,            # penaliza valores extremos en las hojas (L2)
    'reg_alpha': 0.0,             # penalización L1 (puede anular variables poco útiles)
    'gamma': 0.0,                 # mejora mínima exigida para dividir un nodo
}

# CLASIFICADOR (señales COMPRAR / MANTENER / VENDER)
XGB_CLASIFICADOR = {
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'max_depth': 3,
    'min_child_weight': 20,
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'reg_lambda': 5.0,
    'reg_alpha': 0.0,
    'gamma': 0.0,
    'balancear_clases': True,     # pesos por clase (equivalente a class_weight='balanced')
}

# Parada temprana (aplica a ambos modelos)
EARLY_STOPPING = 50               # rondas sin mejora en validación antes de parar
FRACCION_VALIDACION_ES = 0.15     # % final del entrenamiento usado para la parada temprana
N_JOBS = -1                       # núcleos del procesador (-1 = todos)

# ---------------------------------------------------------------------
# 1B. CONFIGURACIÓN AUTOMÁTICA (solo si MODO_CONFIGURACION = 'auto')
# ---------------------------------------------------------------------
OPCIONES_REGRESOR = [{'max_depth': d, 'learning_rate': lr}
                     for d in (3, 5) for lr in (0.03, 0.1)]
OPCIONES_CLASIFICADOR = [{'max_depth': d, 'learning_rate': lr}
                         for d in (3, 5) for lr in (0.03, 0.1)]

# ---------------------------------------------------------------------
# 1C. VARIABLES Y GRÁFICAS
# ---------------------------------------------------------------------
FEATURES = FEATURES_EXTENDIDAS    # 47 variables; FEATURES_MODELO = 22; o una lista propia

PERMUTACION_REPETICIONES = 5      # veces que se revuelve cada variable
N_CONTRIBUCIONES = 15             # variables mostradas en la explicación de la predicción de hoy

CLASES = ['BAJA', 'MANTENER', 'SUBE']      # etiquetas -1, 0, 1


def _opciones(base, candidatas):
    """Lista de configuraciones que recibe el motor según el modo elegido."""
    if MODO_CONFIGURACION == 'manual':
        return [dict(base)]
    if MODO_CONFIGURACION == 'auto':
        return [{**base, **c} for c in candidatas]
    raise ValueError("MODO_CONFIGURACION debe ser 'manual' o 'auto'")


# =====================================================================
# PARADA TEMPRANA (común a regresor y clasificador)
# =====================================================================
def _particion_parada(n_filas, hueco):
    """Índices (entrenamiento interno, validación) para la parada temprana.
    Entre ambos se deja un hueco de `hueco` filas (el horizonte en días
    hábiles): la respuesta de esas filas cae dentro de la validación."""
    n_val = max(20, int(n_filas * FRACCION_VALIDACION_ES))
    fin_tr = n_filas - n_val - hueco
    if fin_tr < 100:
        return None
    return np.arange(fin_tr), np.arange(n_filas - n_val, n_filas)


def _ajustar_con_parada(crear, X, y, hueco, peso=None, clases_ok=None):
    """1) Entrena con parada temprana sobre una validación interna.
       2) Reentrena con TODOS los datos usando el mejor número de árboles.
    Devuelve (modelo_final, info) con las curvas de error para la Gráfica 7."""
    info = {'curvas': None, 'mejor_iteracion': None}
    part = _particion_parada(len(X), hueco)
    n_arboles = None
    if part is not None:
        i_tr, i_val = part
        if clases_ok is not None:          # la validación solo con clases vistas en el entrenamiento interno
            vistas = set(np.unique(y[i_tr]))
            i_val = i_val[np.isin(y[i_val], list(vistas))]
            if len(vistas) < 2 or len(i_val) < 10 or not clases_ok(y[i_tr]):
                part = None
        if part is not None:
            modelo_es = crear(early_stopping_rounds=EARLY_STOPPING)
            w_tr = peso[i_tr] if peso is not None else None
            modelo_es.fit(X.iloc[i_tr], y[i_tr], sample_weight=w_tr,
                          eval_set=[(X.iloc[i_tr], y[i_tr]), (X.iloc[i_val], y[i_val])], verbose=False)
            n_arboles = int(modelo_es.best_iteration) + 1
            info = {'curvas': modelo_es.evals_result(), 'mejor_iteracion': n_arboles - 1}
    final = crear(n_estimators=n_arboles) if n_arboles else crear()
    final.fit(X, y, sample_weight=peso, verbose=False)
    info['n_arboles'] = n_arboles or final.get_params()['n_estimators']
    return final, info


# =====================================================================
# SECCIÓN 2. REGRESOR (pronóstico de precio)
# =====================================================================
def entrenar_regresor(X, Y_dias, params):
    n_dias = Y_dias.shape[1]
    y = Y_dias.iloc[:, -1].values                    # rendimiento al final del horizonte

    def crear(**extra):
        p = {**params, **extra}
        return XGBRegressor(objective='reg:squarederror', eval_metric='mae', tree_method='hist',
                            random_state=SEED, n_jobs=N_JOBS, **p)

    modelo, info = _ajustar_con_parada(crear, X, y, hueco=n_dias)
    return {'modelo': modelo, 'info': info, 'n_dias': n_dias}


def predecir_regresor(regresor, X):
    """Array (filas, días): rendimiento final predicho, trazado en línea recta
    desde 0 hasta el día N (solo se usa para la trayectoria oculta)."""
    rend_final = regresor['modelo'].predict(X)
    fraccion = np.arange(1, regresor['n_dias'] + 1) / regresor['n_dias']
    return np.outer(rend_final, fraccion)


# =====================================================================
# SECCIÓN 3. CLASIFICADOR (señales COMPRAR / MANTENER / VENDER)
# =====================================================================
def entrenar_clasificador(X, y_sube, y_baja, params, n_dias):
    y3 = (y_sube - y_baja).astype(int).values        # -1 baja, 0 se queda, 1 sube
    presentes = sorted(np.unique(y3))
    if len(presentes) < 2:
        return None
    # XGBoost necesita etiquetas 0..k-1 consecutivas
    a_indice = {c: i for i, c in enumerate(presentes)}
    y_idx = np.array([a_indice[c] for c in y3])
    params = dict(params)
    balancear = params.pop('balancear_clases', True)
    peso = compute_sample_weight('balanced', y_idx) if balancear else None
    k = len(presentes)

    def crear(**extra):
        p = {**params, **extra}
        if k == 2:
            return XGBClassifier(objective='binary:logistic', eval_metric='logloss', tree_method='hist',
                                 random_state=SEED, n_jobs=N_JOBS, **p)
        return XGBClassifier(objective='multi:softprob', eval_metric='mlogloss',
                             tree_method='hist', random_state=SEED, n_jobs=N_JOBS, **p)

    modelo, info = _ajustar_con_parada(crear, X, y_idx, hueco=n_dias, peso=peso,
                                       clases_ok=lambda y: len(np.unique(y)) == k)
    return {'modelo': modelo, 'info': info, 'clases': presentes}


def predecir_clasificador(clasificador, X):
    """(p_sube, p_baja): probabilidad de SUBE y de BAJA."""
    if clasificador is None:
        nan = np.full(len(X), np.nan)
        return nan, nan
    proba = clasificador['modelo'].predict_proba(X)
    clases = clasificador['clases']
    p_sube = proba[:, clases.index(1)] if 1 in clases else np.zeros(len(X))
    p_baja = proba[:, clases.index(-1)] if -1 in clases else np.zeros(len(X))
    return p_sube, p_baja


# =====================================================================
# SECCIÓN 4. CONEXIÓN CON EL MOTOR (une regresor + clasificador)
# =====================================================================
def entrenar(X, Y_dias, y_sube, y_baja, p_precio, p_senal):
    return {'regresor': entrenar_regresor(X, Y_dias, p_precio),
            'clasificador': entrenar_clasificador(X, y_sube, y_baja, p_senal, Y_dias.shape[1]),
            'p_precio': p_precio, 'p_senal': p_senal}


def predecir(modelo, X):
    rend = predecir_regresor(modelo['regresor'], X)
    p_sube, p_baja = predecir_clasificador(modelo['clasificador'], X)
    return rend, p_sube, p_baja


# =====================================================================
# SECCIÓN 5. GRÁFICAS PROPIAS DE XGBOOST
# =====================================================================
def imprimir_parametros(modelo):
    claves = ['learning_rate', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree',
              'reg_lambda', 'reg_alpha', 'gamma']
    print("PARÁMETROS DE LOS MODELOS FINALES (los que generan el pronóstico):")
    for nombre, parte in [('Regresor', modelo['regresor']), ('Clasificador', modelo['clasificador'])]:
        if parte is None:
            print(f"  {nombre}: no se entrenó (una sola clase en los datos)")
            continue
        params = parte['modelo'].get_params()
        info = parte['info']
        print(f"\n  {nombre}:")
        for k in claves:
            print(f"    {k:<27} {params.get(k)}")
        if info['mejor_iteracion'] is not None:
            print(f"    {'árboles (parada temprana)':<27} {info['n_arboles']} de {modelo['p_precio' if nombre == 'Regresor' else 'p_senal'].get('n_estimators')} posibles")
        else:
            print(f"    {'árboles':<27} {info['n_arboles']} (sin parada temprana: datos insuficientes)")
        if nombre == 'Clasificador':
            print(f"    {'clases':<27} {', '.join(CLASES[c + 1] for c in parte['clases'])}")
    print()


def _modelo_periodo_reciente(modelo, ctx):
    """Modelos entrenados solo hasta el periodo de prueba más reciente (con la
    configuración final), para evaluar variables sin hacer trampa."""
    d, f_tr = ctx.d, ctx.f_train
    return entrenar(d.X.loc[f_tr], d.Y.loc[f_tr], d.y_sube.loc[f_tr], d.y_baja.loc[f_tr],
                    modelo['p_precio'], modelo['p_senal'])


def _importancia_ganancia(modelo_xgb, features):
    score = modelo_xgb.get_booster().get_score(importance_type='gain')
    s = pd.Series({f: score.get(f, 0.0) for f in features})
    return s / s.sum() if s.sum() > 0 else s


def _grafica_importancia(modelo, m_rec, ctx):
    print(f"GRÁFICA 6: {ctx.ticker} - Importancia de variables")
    d, f_te = ctx.d, ctx.f_test
    X_te = d.X.loc[f_te]
    y_rend = d.Y.loc[f_te].values[:, -1]
    y3 = (d.y_sube.loc[f_te] - d.y_baja.loc[f_te]).values

    def score_reg(est, X, y):
        return -mean_absolute_error(y, est.predict(X))

    t0 = time.time()
    perm_reg = permutation_importance(m_rec['regresor']['modelo'], X_te, y_rend, scoring=score_reg,
                                      n_repeats=PERMUTACION_REPETICIONES, random_state=SEED, n_jobs=1)
    perm_clf = None
    clf_rec = m_rec['clasificador']
    if clf_rec is not None:
        en_modelo = np.isin(y3, clf_rec['clases'])
        y_idx = np.array([clf_rec['clases'].index(c) for c in y3[en_modelo]])
        if len(np.unique(y_idx)) >= 2:
            perm_clf = permutation_importance(
                clf_rec['modelo'], X_te[en_modelo], y_idx,
                scoring=lambda est, X, y: balanced_accuracy_score(y, est.predict(X)),
                n_repeats=PERMUTACION_REPETICIONES, random_state=SEED, n_jobs=1)
    print(f"  Importancia por permutación calculada en {time.time() - t0:.1f} s "
          f"(periodo {f_te[0].date()} a {f_te[-1].date()})")

    paneles = [('Regresor: ganancia (gain)', _importancia_ganancia(modelo['regresor']['modelo'], ctx.features),
                None, 'steelblue'),
               ('Regresor: permutación (aumento del MAE)',
                pd.Series(perm_reg.importances_mean, index=ctx.features),
                pd.Series(perm_reg.importances_std, index=ctx.features), 'navy')]
    if modelo['clasificador'] is not None:
        paneles.append(('Clasificador: ganancia (gain)',
                        _importancia_ganancia(modelo['clasificador']['modelo'], ctx.features), None, 'darkorange'))
    if perm_clf is not None:
        paneles.append(('Clasificador: permutación (caída del accuracy balanceado)',
                        pd.Series(perm_clf.importances_mean, index=ctx.features),
                        pd.Series(perm_clf.importances_std, index=ctx.features), 'saddlebrown'))

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    for ax in axes.flat[len(paneles):]:
        ax.axis('off')
    for ax, (titulo, serie, error, color) in zip(axes.flat, paneles):
        top = serie.sort_values(ascending=False).head(15)[::-1]
        err = error.loc[top.index] if error is not None else None
        ax.barh(top.index, top.values, xerr=err, color=color, alpha=0.8, capsize=3)
        for y, v in enumerate(top.values):
            ax.text(v, y, f' {v:.4f}', va='center', fontsize=8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(titulo, fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
    plt.suptitle(f'{ctx.ticker} - Importancia de variables (15 principales)\n'
                 f'Ganancia: cuánto mejora el modelo cada vez que usa la variable. Permutación: cuánto empeora '
                 f'al revolverla en el periodo de prueba más reciente.', fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.show(block=False); plt.pause(0.1)

    utiles = (perm_reg.importances_mean > 0).sum()
    print(f"  Variables que mejoran el pronóstico fuera del entrenamiento (permutación > 0): "
          f"{utiles} de {len(ctx.features)}")
    if perm_clf is not None:
        print(f"  Variables que mejoran las señales fuera del entrenamiento: "
              f"{(perm_clf.importances_mean > 0).sum()} de {len(ctx.features)}")


def _grafica_aprendizaje(modelo, ctx):
    print(f"\nGRÁFICA 7: {ctx.ticker} - Curva de aprendizaje (parada temprana)")
    partes = [('Regresor', modelo['regresor'], 'mae', 'MAE del rendimiento'),
              ('Clasificador', modelo['clasificador'], None, 'Log-loss')]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (nombre, parte, metrica, etiqueta) in zip(axes, partes):
        if parte is None or parte['info']['curvas'] is None:
            ax.text(0.5, 0.5, f'{nombre}: sin parada temprana\n(datos insuficientes para validación interna)',
                    ha='center', va='center', color='gray'); ax.axis('off')
            continue
        curvas = parte['info']['curvas']
        met = metrica or list(curvas['validation_0'].keys())[0]
        tr, val = curvas['validation_0'][met], curvas['validation_1'][met]
        mejor = parte['info']['mejor_iteracion']
        ax.plot(tr, color='steelblue', label='Entrenamiento')
        ax.plot(val, color='darkorange', label='Validación')
        ax.axvline(mejor, color='red', linestyle='--', label=f'Mejor: {mejor + 1} árboles')
        ax.set_title(f'{nombre}: error por número de árboles')
        ax.set_xlabel('Número de árboles'); ax.set_ylabel(etiqueta)
        ax.legend(); ax.grid(True, alpha=0.3)
        brecha = (val[mejor] - tr[mejor]) / abs(tr[mejor]) * 100 if tr[mejor] else np.nan
        print(f"  {nombre}: se detuvo en {mejor + 1} árboles; error validación {val[mejor]:.4f} vs "
              f"entrenamiento {tr[mejor]:.4f} (brecha {brecha:+.0f} %)")
    plt.suptitle(f'{ctx.ticker} - Curva de aprendizaje\nSi el entrenamiento sigue bajando y la validación sube, '
                 f'el modelo empieza a memorizar: ahí se detiene.', fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.show(block=False); plt.pause(0.1)


def _contribuciones(modelo_xgb, X):
    """Aporte de cada variable a la predicción (valores SHAP calculados por XGBoost).
    Regresor / binario: (filas, variables + 1); multiclase: (filas, clases, variables + 1).
    La última columna es el valor base."""
    return modelo_xgb.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)


def _grafica_explicacion(modelo, ctx):
    print(f"\nGRÁFICA 8: {ctx.ticker} - ¿Por qué el modelo predice esto hoy?")
    df = ctx.d.df
    ultima = df[ctx.features].dropna().iloc[[-1]]
    fecha = ultima.index[0]

    paneles = []
    contrib_reg = _contribuciones(modelo['regresor']['modelo'], ultima)[0]
    rend_total = contrib_reg.sum()
    # aporte en puntos porcentuales del precio esperado (aprox. para valores pequeños)
    paneles.append((f'Precio: aporte al cambio esperado a {ctx.dias} días (puntos %)\n'
                    f'Cambio esperado total: {(np.exp(rend_total) - 1) * 100:+.2f} %',
                    pd.Series(contrib_reg[:-1] * 100, index=ctx.features)))

    clf = modelo['clasificador']
    if clf is not None:
        c = _contribuciones(clf['modelo'], ultima)
        clases = clf['clases']
        if c.ndim == 3:                                   # multiclase
            for etiqueta, valor in [('SUBE', 1), ('BAJA', -1)]:
                if valor in clases:
                    paneles.append((f'Señal: aporte a la clase {etiqueta} (log-odds)',
                                    pd.Series(c[0, clases.index(valor), :-1], index=ctx.features)))
        else:                                             # binario: aporte a la clase de índice 1
            etiqueta = CLASES[clases[1] + 1]
            paneles.append((f'Señal: aporte a la clase {etiqueta} (log-odds)',
                            pd.Series(c[0, :-1], index=ctx.features)))

    fig, axes = plt.subplots(1, len(paneles), figsize=(7 * len(paneles), 8))
    axes = np.atleast_1d(axes)
    for ax, (titulo, serie) in zip(axes, paneles):
        top = serie.reindex(serie.abs().sort_values(ascending=False).index).head(N_CONTRIBUCIONES)[::-1]
        ax.barh(top.index, top.values, color=['green' if v > 0 else 'red' for v in top.values], alpha=0.75)
        for y, v in enumerate(top.values):
            ax.text(v, y, f' {v:+.3f}', va='center', ha='left' if v >= 0 else 'right', fontsize=8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(titulo, fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
    plt.suptitle(f'{ctx.ticker} - Explicación de la predicción del {fecha.date()} '
                 f'(verde: empuja hacia arriba / a esa clase, rojo: en contra)', fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.show(block=False); plt.pause(0.1)

    serie_precio = paneles[0][1]
    orden = serie_precio.sort_values()
    print("  Variables que más SUBEN el precio esperado: " +
          ", ".join(f"{k} ({v:+.2f} pts)" for k, v in orden[::-1].head(5).items()))
    print("  Variables que más BAJAN el precio esperado: " +
          ", ".join(f"{k} ({v:+.2f} pts)" for k, v in orden.head(5).items()))


def graficas_propias(modelo, ctx):
    imprimir_parametros(modelo)
    print("Entrenando modelos del periodo de prueba más reciente para evaluar variables...")
    m_rec = _modelo_periodo_reciente(modelo, ctx)
    _grafica_importancia(modelo, m_rec, ctx)
    _grafica_aprendizaje(modelo, ctx)
    _grafica_explicacion(modelo, ctx)


# =====================================================================
# SECCIÓN 6. EJECUCIÓN
# =====================================================================
if __name__ == '__main__':
    ejecutar_modelo(
        nombre="XGBoost",
        entrenar=entrenar,
        predecir=predecir,
        nombre_precio="XGBoost Regresor",
        nombre_senal="XGBoost Clasificador",
        opciones_precio=_opciones(XGB_REGRESOR, OPCIONES_REGRESOR),
        opciones_senal=_opciones(XGB_CLASIFICADOR, OPCIONES_CLASIFICADOR),
        graficas_propias=graficas_propias,
        features=FEATURES,
        tres_clases=True,          # señal = clase más probable de BAJA / MANTENER / SUBE
    )
