#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modelo_lstm.py
==============
Red LSTM. Por defecto ('arquitectura': 'separadas') son DOS redes independientes,
cada una con su propia parada temprana: una para el precio y otra para la señal.
Con 'arquitectura': 'conjunta' es una sola red con las dos salidas:

             secuencia de VENTANA días x features
                            |
                  LSTM + Dropout (1 o 2 capas)
                            |
             +--------------+---------------+
      Salida "precio"                 Salida "senal"
      N valores: rendimiento          3 clases (softmax):
      de cada día 1..N                BAJA / MANTENER / SUBE

A diferencia de los demás modelos, que ven un solo día, la LSTM recibe la
evolución de las features en los últimos VENTANA días. La salida de precio da
la trayectoria completa del horizonte (no una línea recta).

Diferencias con el notebook LSTM.ipynb:
  - Usa las features como proporciones (no el precio en dinero) y Close como
    referencia (el notebook tomaba la columna High por error: iloc[:, 1:2]).
  - Predice el horizonte elegido, no solo el día siguiente (predecir mañana con
    el precio de hoy da una gráfica casi perfecta pero equivale a copiar el
    último valor).
  - Escala con StandardScaler ajustado solo con el entrenamiento (MinMax sobre
    precios deja fuera de rango los máximos históricos nuevos).
  - Validación walk-forward y parada temprana con hueco anti-fuga.

Requiere: tensorflow / keras (ya instalado en el entorno ML).

Para configurar la red edita la SECCIÓN 1.

Uso:
    python3 modelo_lstm.py
"""
import os
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')     # menos mensajes de TensorFlow

# Usar tarjeta de video (GPU). False = solo procesador (CPU): evita el aviso
# "CUDA error: Failed call to cuInit" cuando no hay GPU compatible configurada.
USAR_GPU = False
if not USAR_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

import motor_prediccion
from motor_prediccion import ejecutar_modelo, FEATURES_EXTENDIDAS, FEATURES_MODELO, SEED


# =====================================================================
# SECCIÓN 1. CONFIGURACIÓN  <-- AQUÍ SE EDITA LA RED
# =====================================================================
#
# MODO_CONFIGURACION
#   'manual' : se usa exactamente LSTM_CONFIG.
#   'auto'   : el motor prueba las combinaciones de OPCIONES_LSTM en
#              validación. Cada combinación entrena la red en cada periodo:
#              puede tardar mucho. Úsalo con pocas opciones.
MODO_CONFIGURACION = 'manual'

# ---------------------------------------------------------------------
# 1A. CONFIGURACIÓN MANUAL
# ---------------------------------------------------------------------
# Valores actuales = red original. La columna "propuesta" son las mejoras
# contra la memorización; pruébalas de una en una y compara el PROMEDIO
# WALK-FORWARD.
LSTM_CONFIG = {                   #                                  propuesta
    'ventana': 60,                # días de historia por predicción      30
    'unidades_lstm': 64,          # tamaño de la memoria de la LSTM      16-32
    'capas_lstm': 1,              # 1 o 2 capas LSTM                     1
    'dropout': 0.2,               # neuronas apagadas al entrenar        0.3-0.4
    'recurrent_dropout': 0.0,     # dropout dentro de la memoria         0.2
    'l2': 0.0,                    # penaliza pesos grandes (0 = no)      0.0001
    'learning_rate': 0.001,       # velocidad de aprendizaje (Adam)      0.0003
    'batch_size': 64,             # secuencias por paso                  64
    'epocas_max': 100,            # máximo de épocas                     100
    'paciencia': 10,              # épocas sin mejora antes de parar     15
    'peso_senal': 1.0,            # peso de la salida de señal           1.0
    'balancear_clases': True,     # pesos por clase en la señal          True
    'salida_precio': 'todos',     # 'todos': un valor por día 1..N       'final'
                                  # 'final': solo el día N (como XGBoost)
    'n_redes': 1,                 # redes promediadas (ensamble)         3-5
    'arquitectura': 'separadas',  # 'separadas': una red para precio y otra para señal,
                                  #   cada una con su propia parada temprana.
                                  # 'conjunta': una sola red con las dos salidas.
}

# Periodos de prueba del walk-forward. 5 es lo normal; con 2-3 las pruebas de
# configuración tardan la mitad (las métricas son menos confiables).
PERIODOS_WALK_FORWARD = 5

# Fracción final del entrenamiento usada como validación para la parada temprana
FRACCION_VALIDACION_ES = 0.15

# Reentrenar con todos los datos (incluida la validación interna) usando el
# mejor número de épocas. Más preciso con los datos recientes, pero duplica el
# tiempo. Con False se usa la red con los mejores pesos de la parada temprana.
REENTRENAR_CON_TODO = False

# ---------------------------------------------------------------------
# 1B. CONFIGURACIÓN AUTOMÁTICA (solo si MODO_CONFIGURACION = 'auto')
# ---------------------------------------------------------------------
# Los parámetros que no se indiquen toman el valor de LSTM_CONFIG.
OPCIONES_LSTM = [{'unidades_lstm': 32}, {'unidades_lstm': 64}]

# ---------------------------------------------------------------------
# 1C. VARIABLES Y GRÁFICAS
# ---------------------------------------------------------------------
# Las 6 variables que más usó la red con las 47 en la corrida de AAPL (max, 365
# días): volatilidad, rango del día, ATR, distancia a la EMA200, rendimiento de
# 20 días y gap de apertura. AVISO: se eligieron mirando un periodo de prueba,
# así que el resultado con ellas puede salir optimista. Confírmalo con otra
# acción u otro horizonte antes de confiar en la mejora.
FEATURES_REDUCIDAS = ['Volatilidad20', 'Rango_dia', 'ATR_pct', 'Dist_EMA200', 'Ret_20', 'Gap_apertura']

FEATURES = FEATURES_MODELO          # 22 variables
# FEATURES = FEATURES_EXTENDIDAS    # 47 variables (todos los indicadores)
# FEATURES = FEATURES_REDUCIDAS     # 6 variables
# FEATURES = ['RSI', 'Dist_MA200', ...]   # o tu propia lista

PERMUTACION_REPETICIONES = 3      # veces que se revuelve cada variable
N_VARIABLES_VENTANA = 5           # variables mostradas en "lo que ve la red hoy"
VERBOSE_ENTRENAMIENTO = 0         # 0 = silencioso, 1 = barra de progreso de Keras

CLASES = ['BAJA', 'MANTENER', 'SUBE']      # salida softmax: índices 0, 1, 2 = etiquetas -1, 0, 1


def _opciones(base, candidatas):
    if MODO_CONFIGURACION == 'manual':
        return [dict(base)]
    if MODO_CONFIGURACION == 'auto':
        return [{**base, **c} for c in candidatas]
    raise ValueError("MODO_CONFIGURACION debe ser 'manual' o 'auto'")


def _keras():
    """Importa Keras/TensorFlow solo al momento de entrenar.

    TensorFlow y curl_cffi (la librería que usa yfinance para descargar de Yahoo)
    traen cada uno su propia versión de las librerías de conexión segura. Si
    TensorFlow se carga antes de la descarga, yfinance puede fallar. Cargándolo
    aquí, la descarga a MySQL ya terminó cuando TensorFlow entra en memoria."""
    import keras
    try:                                    # oculta el aviso "tf.function retracing" (no afecta resultados)
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
    except Exception:
        pass
    return keras


# =====================================================================
# SECCIÓN 2. PREPARACIÓN DE SECUENCIAS
# =====================================================================
def _historia():
    h = motor_prediccion.HISTORIA_FEATURES
    if h is None:
        raise RuntimeError("El motor no cargó el histórico de features (HISTORIA_FEATURES).")
    return h


def armar_secuencias(fechas, scaler, ventana):
    """Para cada fecha devuelve la ventana de `ventana` días que termina en ella
    (inclusive), con las features escaladas. Forma: (fechas, ventana, features).
    Las fechas sin historia suficiente (inicio de los datos) se rellenan
    repitiendo el primer día disponible; se devuelve también una máscara de
    cuáles tienen ventana completa."""
    hist = _historia()
    H = scaler.transform(hist.values).astype('float32')
    pos = hist.index.get_indexer(fechas)
    if (pos < 0).any():
        raise ValueError("Hay fechas sin features en el histórico.")
    completas = pos >= ventana - 1
    relleno = np.vstack([np.repeat(H[:1], ventana - 1, axis=0), H])     # para ventanas incompletas
    idx = pos[:, None] + np.arange(ventana)[None, :]                     # (n, ventana) en coordenadas del relleno
    return relleno[idx], completas


# =====================================================================
# SECCIÓN 3. REDES NEURONALES
# =====================================================================
# tipo = 'conjunta' : una red con dos salidas [precio, señal]
#        'precio'   : red solo de precio (rendimiento escalado)
#        'senal'    : red solo de señal (BAJA / MANTENER / SUBE)
def construir_red(ventana, n_features, n_salidas_precio, p, tipo='conjunta'):
    keras = _keras()
    layers = keras.layers
    reg = keras.regularizers.l2(p['l2']) if p.get('l2', 0) > 0 else None
    entrada = keras.Input(shape=(ventana, n_features), name='secuencia')
    x = entrada
    for i in range(p['capas_lstm']):
        ultima = i == p['capas_lstm'] - 1
        x = layers.LSTM(p['unidades_lstm'], return_sequences=not ultima,
                        recurrent_dropout=p.get('recurrent_dropout', 0.0),
                        kernel_regularizer=reg, recurrent_regularizer=reg, name=f'lstm_{i + 1}')(x)
        x = layers.Dropout(p['dropout'], name=f'dropout_{i + 1}')(x)
    optimizador = keras.optimizers.Adam(learning_rate=p['learning_rate'])

    if tipo == 'precio':
        salida = layers.Dense(n_salidas_precio, kernel_regularizer=reg, name='precio')(x)
        red = keras.Model(entrada, salida, name='lstm_precio')
        red.compile(optimizer=optimizador, loss='mse')
        return red
    if tipo == 'senal':
        salida = layers.Dense(3, activation='softmax', kernel_regularizer=reg, name='senal')(x)
        red = keras.Model(entrada, salida, name='lstm_senal')
        red.compile(optimizer=optimizador, loss='sparse_categorical_crossentropy')
        return red

    salida_precio = layers.Dense(n_salidas_precio, kernel_regularizer=reg, name='precio')(x)
    salida_senal = layers.Dense(3, activation='softmax', kernel_regularizer=reg, name='senal')(x)
    red = keras.Model(entrada, [salida_precio, salida_senal], name='lstm_precio_senal')
    red.compile(optimizer=optimizador,
                loss=['mse', 'sparse_categorical_crossentropy'],      # orden: [precio, senal]
                loss_weights=[1.0, p['peso_senal']])
    return red


def _fit(red, tipo, X, y_precio, y_senal, w_senal, **kwargs):
    """fit según el tipo de red. En la red conjunta, respuestas y pesos van como
    listas en el orden [precio, senal] (formato de Keras 3) y el peso por clase
    se aplica solo a la señal; si esa versión de Keras no acepta pesos por
    salida, se aplica el mismo peso a ambas."""
    if tipo == 'precio':
        return red.fit(X, y_precio, **kwargs)
    if tipo == 'senal':
        return red.fit(X, y_senal, sample_weight=w_senal, **kwargs)
    y = [y_precio, y_senal]
    try:
        return red.fit(X, y, sample_weight=[np.ones(len(X), dtype='float32'), w_senal], **kwargs)
    except (TypeError, ValueError, KeyError):
        return red.fit(X, y, sample_weight=w_senal, **kwargs)


def _validacion(tipo, Y_esc, y_cls):
    if tipo == 'precio':
        return Y_esc
    if tipo == 'senal':
        return y_cls
    return [Y_esc, y_cls]


def _ajustar_red(p, tipo, Xs, Y_esc, y_cls, w_cls, n_dias, semilla=SEED):
    """Entrena una red con parada temprana sobre el final de los datos (con un
    hueco de n_dias filas contra la fuga). Cada red usa su propia pérdida de
    validación para detenerse. Devuelve {'red', 'historial', 'mejor'}."""
    keras = _keras()
    n = len(Xs)
    n_val = max(20, int(n * FRACCION_VALIDACION_ES))
    fin_tr = n - n_val - n_dias
    keras.utils.set_random_seed(semilla)
    red = construir_red(Xs.shape[1], Xs.shape[2], Y_esc.shape[1], p, tipo)

    if fin_tr < 100:        # muy pocos datos: entrenamiento sin validación interna
        hist = _fit(red, tipo, Xs, Y_esc, y_cls, w_cls, epochs=min(20, p['epocas_max']),
                    batch_size=p['batch_size'], verbose=VERBOSE_ENTRENAMIENTO, shuffle=True)
        return {'red': red, 'historial': hist.history, 'mejor': None}

    tr, val = slice(0, fin_tr), slice(n - n_val, n)
    parada = keras.callbacks.EarlyStopping(monitor='val_loss', patience=p['paciencia'],
                                           restore_best_weights=True)
    hist = _fit(red, tipo, Xs[tr], Y_esc[tr], y_cls[tr], w_cls[tr],
                validation_data=(Xs[val], _validacion(tipo, Y_esc[val], y_cls[val])),
                epochs=p['epocas_max'], batch_size=p['batch_size'], callbacks=[parada],
                verbose=VERBOSE_ENTRENAMIENTO, shuffle=True)
    mejor = int(np.argmin(hist.history['val_loss']))

    if REENTRENAR_CON_TODO:
        keras.utils.set_random_seed(semilla)
        red = construir_red(Xs.shape[1], Xs.shape[2], Y_esc.shape[1], p, tipo)
        _fit(red, tipo, Xs, Y_esc, y_cls, w_cls, epochs=mejor + 1, batch_size=p['batch_size'],
             verbose=VERBOSE_ENTRENAMIENTO, shuffle=True)
    return {'red': red, 'historial': hist.history, 'mejor': mejor}


# =====================================================================
# SECCIÓN 4. CONEXIÓN CON EL MOTOR
# =====================================================================
_MODELOS_ENTRENADOS = {}      # se reutilizan en las gráficas (evita reentrenar)


def entrenar(X, Y_dias, y_sube, y_baja, p_precio, p_senal):
    p = {**p_precio}
    t0 = time.time()
    ventana = p['ventana']

    # Escaladores ajustados SOLO con el entrenamiento
    scaler_x = StandardScaler().fit(X.values)
    Xs, completas = armar_secuencias(X.index, scaler_x, ventana)
    Xs = Xs[completas]                                     # solo ventanas completas para entrenar
    Y = Y_dias.values[completas]
    if p.get('salida_precio', 'todos') == 'final':
        Y = Y[:, -1:]                                      # solo el rendimiento del día N
    y_media, y_std = Y.mean(axis=0), Y.std(axis=0) + 1e-8
    Y_esc = ((Y - y_media) / y_std).astype('float32')

    y_cls = ((y_sube - y_baja).values[completas] + 1).astype('int32')     # 0 BAJA, 1 MANTENER, 2 SUBE
    w_cls = (compute_sample_weight('balanced', y_cls) if p['balancear_clases']
             else np.ones(len(y_cls))).astype('float32')

    arquitectura = p.get('arquitectura', 'conjunta')
    tipos = ['precio', 'senal'] if arquitectura == 'separadas' else ['conjunta']
    grupos = {}
    for tipo in tipos:
        grupos[tipo] = [_ajustar_red(p, tipo, Xs, Y_esc, y_cls, w_cls, Y_dias.shape[1], semilla=SEED + k)
                        for k in range(max(1, int(p.get('n_redes', 1))))]

    partes = []
    for tipo, redes in grupos.items():
        epocas = '/'.join(str(len(r['historial']['loss'])) for r in redes)
        mejores = '/'.join(str(r['mejor'] + 1) for r in redes if r['mejor'] is not None)
        etiqueta = {'precio': 'precio', 'senal': 'señal', 'conjunta': 'red conjunta'}[tipo]
        partes.append(f"{etiqueta}: épocas {epocas}{f' (mejor {mejores})' if mejores else ''}")
    n_r = max(1, int(p.get('n_redes', 1)))
    print(f"    LSTM {'separadas' if arquitectura == 'separadas' else 'conjunta'}"
          f"{f' x{n_r}' if n_r > 1 else ''} con {len(Xs)} secuencias | {' | '.join(partes)} "
          f"| {time.time() - t0:.0f} s")

    modelo = {'grupos': grupos, 'arquitectura': arquitectura, 'scaler_x': scaler_x,
              'y_media': y_media, 'y_std': y_std, 'ventana': ventana, 'n_dias': Y_dias.shape[1],
              'p_precio': p_precio, 'p_senal': p_senal}
    _MODELOS_ENTRENADOS[(X.index[0], X.index[-1], len(X), repr(p))] = modelo
    return modelo


def _predecir_secuencias(modelo, Xs):
    """Promedio del ensamble. Devuelve (rendimiento por día (n, N), probabilidades (n, 3))."""
    rends, probas = [], []
    g = modelo['grupos']
    if 'conjunta' in g:
        for r in g['conjunta']:
            pred_esc, proba = r['red'].predict(Xs, batch_size=256, verbose=0)
            rends.append(pred_esc); probas.append(proba)
    else:
        rends = [r['red'].predict(Xs, batch_size=256, verbose=0) for r in g['precio']]
        probas = [r['red'].predict(Xs, batch_size=256, verbose=0) for r in g['senal']]
    rend = np.mean(rends, axis=0) * modelo['y_std'] + modelo['y_media']
    proba = np.mean(probas, axis=0)
    if rend.shape[1] == 1 and modelo['n_dias'] > 1:       # salida 'final': trayectoria en línea recta
        rend = np.outer(rend[:, 0], np.arange(1, modelo['n_dias'] + 1) / modelo['n_dias'])
    return rend, proba


def predecir(modelo, X):
    Xs, _ = armar_secuencias(X.index, modelo['scaler_x'], modelo['ventana'])
    rend, proba = _predecir_secuencias(modelo, Xs)
    return rend, proba[:, 2], proba[:, 0]                 # p_sube, p_baja


# =====================================================================
# SECCIÓN 5. GRÁFICAS PROPIAS DE LA LSTM
# =====================================================================
def imprimir_parametros(modelo):
    print("PARÁMETROS DE LA RED FINAL (la que genera el pronóstico):")
    for k, v in modelo['p_precio'].items():
        print(f"    {k:<26} {v}")
    etiquetas = {'precio': 'red precio', 'senal': 'red señal', 'conjunta': 'red conjunta'}
    hubo_epoca_1 = False
    for tipo, redes in modelo['grupos'].items():
        for i, r in enumerate(redes, 1):
            nombre = etiquetas[tipo] + (f' {i}' if len(redes) > 1 else '')
            m = r['mejor']
            print(f"    {nombre + ': épocas':<26} {len(r['historial']['loss'])}"
                  f"{f' (mejor: {m + 1})' if m is not None else ''}")
            hubo_epoca_1 |= (m == 0)
        print(f"    {etiquetas[tipo] + ': parámetros':<26} {redes[0]['red'].count_params():,}")
    if hubo_epoca_1:
        print("    AVISO: alguna red tuvo su mejor validación en la época 1 (casi no aprendió). "
              "Prueba menos unidades, más dropout o un learning_rate menor.")
    print()


def _grafica_aprendizaje(modelo, ctx):
    g = modelo['grupos']
    n = len(next(iter(g.values())))
    print(f"GRÁFICA 6: {ctx.ticker} - Curva de aprendizaje{f' (red 1 de {n} en cada grupo)' if n > 1 else ''}")

    def serie(h, nombre, val=False):
        for k in h:
            if nombre in k and 'loss' in k and k.startswith('val_') == val:
                return h[k]
        return None

    if 'conjunta' in g:
        r = g['conjunta'][0]
        paneles = [('Pérdida total', r, 'loss'), ('Salida de precio (MSE escalado)', r, 'precio'),
                   ('Salida de señal (entropía cruzada)', r, 'senal')]
    else:
        paneles = [('Red de precio (MSE escalado)', g['precio'][0], 'loss'),
                   ('Red de señal (entropía cruzada)', g['senal'][0], 'loss')]

    fig, axes = plt.subplots(1, len(paneles), figsize=(6 * len(paneles), 5))
    for ax, (titulo, r, clave) in zip(np.atleast_1d(axes), paneles):
        h, mejor = r['historial'], r['mejor']
        tr = h.get('loss') if clave == 'loss' else serie(h, clave)
        va = h.get('val_loss') if clave == 'loss' else serie(h, clave, val=True)
        if tr is not None:
            ax.plot(range(1, len(tr) + 1), tr, color='steelblue', label='Entrenamiento')
        if va is not None:
            ax.plot(range(1, len(va) + 1), va, color='darkorange', label='Validación')
        if mejor is not None:
            ax.axvline(mejor + 1, color='red', linestyle='--', label=f'Mejor época: {mejor + 1}')
        ax.set_title(titulo); ax.set_xlabel('Época'); ax.set_ylabel('Pérdida')
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle(f'{ctx.ticker} - Curva de aprendizaje de la LSTM\nSi el entrenamiento sigue bajando y la '
                 f'validación sube, la red empieza a memorizar: ahí se detiene.', fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.show(block=False); plt.pause(0.1)

    for tipo, redes in g.items():
        r = redes[0]
        h, mejor = r['historial'], r['mejor']
        if mejor is not None and h.get('val_loss'):
            nombre = {'precio': 'Red de precio', 'senal': 'Red de señal', 'conjunta': 'Red conjunta'}[tipo]
            print(f"  {nombre}: se detuvo en la época {len(h['loss'])}; mejor validación en la época "
                  f"{mejor + 1} (pérdida {h['val_loss'][mejor]:.4f} vs entrenamiento {h['loss'][mejor]:.4f})")


def _importancia_periodo(modelo, d, f_te, n_features, rng):
    """Importancia por permutación de una red en un periodo de prueba.
    Devuelve (aumento del MAE por variable, caída del accuracy balanceado por
    variable o NaN si el periodo tiene una sola clase real)."""
    Xs, _ = armar_secuencias(f_te, modelo['scaler_x'], modelo['ventana'])
    y_rend = d.Y.loc[f_te].values[:, -1]
    y_cls = (d.y_sube.loc[f_te] - d.y_baja.loc[f_te]).values + 1
    con_clases = len(np.unique(y_cls)) > 1

    def puntuar(X):
        rend, proba = _predecir_secuencias(modelo, X)
        return (mean_absolute_error(y_rend, rend[:, -1]),
                balanced_accuracy_score(y_cls, proba.argmax(axis=1)) if con_clases else np.nan)

    mae_base, acc_base = puntuar(Xs)
    imp_reg = np.zeros((n_features, PERMUTACION_REPETICIONES))
    imp_clf = np.full_like(imp_reg, np.nan)
    for j in range(n_features):
        for r in range(PERMUTACION_REPETICIONES):
            Xp = Xs.copy()
            Xp[:, :, j] = Xs[rng.permutation(len(Xs)), :, j]      # revuelve la secuencia completa de la variable
            mae, acc = puntuar(Xp)
            imp_reg[j, r] = mae - mae_base                         # cuánto empeora el MAE
            imp_clf[j, r] = acc_base - acc                         # cuánto cae el accuracy balanceado
    return imp_reg.mean(axis=1), imp_clf.mean(axis=1)


def _grafica_importancia(modelo, ctx):
    """Importancia promediando TODOS los periodos del walk-forward, cada uno
    evaluado con la red que se entrenó para él (sin ver esos días). Un solo
    periodo suele tener una sola clase real y la parte de señal no se puede
    calcular; con todos juntos casi siempre sí."""
    print(f"\nGRÁFICA 7: {ctx.ticker} - Importancia de variables (permutación, todos los periodos)")
    d, nf = ctx.d, len(ctx.features)
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    regs, clfs, usados = [], [], 0
    # Motores anteriores no pasan ctx.splits: entonces solo el periodo más reciente
    periodos = getattr(ctx, 'splits', None) or [(ctx.f_train, ctx.f_test)]
    for f_tr, f_te in periodos:
        clave = (f_tr[0], f_tr[-1], len(f_tr), repr(modelo['p_precio']))
        red_periodo = _MODELOS_ENTRENADOS.get(clave)
        if red_periodo is None:          # en modo auto pudo elegirse otra configuración en ese periodo
            continue
        r, c = _importancia_periodo(red_periodo, d, f_te, nf, rng)
        regs.append(r); clfs.append(c); usados += 1
    if not regs:
        print("  No hay redes del walk-forward guardadas con la configuración final; se omite.")
        return pd.Series(0.0, index=ctx.features)
    regs, clfs = np.array(regs), np.array(clfs)
    con_senal = int((~np.isnan(clfs).all(axis=1)).sum())
    print(f"  Calculada en {time.time() - t0:.1f} s con {usados} periodos "
          f"(señal: {con_senal} periodos con subidas y no-subidas)")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    resultados = {}
    paneles = [('Precio: aumento del MAE', regs, 'navy'),
               ('Señal: caída del accuracy balanceado', clfs, 'saddlebrown')]
    for ax, (titulo, imp, color) in zip(axes, paneles):
        validos = imp[~np.isnan(imp).all(axis=1)]
        if len(validos) == 0:
            resultados[titulo] = pd.Series(np.nan, index=ctx.features)
            ax.text(0.5, 0.5, 'No calculable: ningún periodo tuvo\nsubidas y no-subidas', ha='center',
                    va='center', color='gray'); ax.axis('off'); continue
        media = pd.Series(np.nanmean(validos, axis=0), index=ctx.features)
        # barra de error = variación entre periodos
        std = pd.Series(np.nanstd(validos, axis=0), index=ctx.features)
        resultados[titulo] = media
        top = media.sort_values(ascending=False).head(15)[::-1]
        ax.barh(top.index, top.values, xerr=std.loc[top.index], color=color, alpha=0.8, capsize=3)
        for y, v in enumerate(top.values):
            ax.text(v, y, f' {v:.4f}', va='center', fontsize=8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(f'{titulo} ({len(validos)} periodos)'); ax.grid(True, alpha=0.3, axis='x')
    plt.suptitle(f'{ctx.ticker} - Importancia de variables en la LSTM (15 principales)\nPromedio de los periodos '
                 f'de prueba del walk-forward. Barra de error = variación entre periodos. Cerca de 0 o negativo = no ayuda.',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.show(block=False); plt.pause(0.1)

    imp_precio = resultados['Precio: aumento del MAE']
    imp_senal = resultados['Señal: caída del accuracy balanceado']
    utiles_p = imp_precio[imp_precio > 0].sort_values(ascending=False)
    print(f"  Variables que mejoran el pronóstico fuera del entrenamiento: {len(utiles_p)} de {nf}")
    print(f"    {', '.join(utiles_p.index)}")
    if not imp_senal.isna().all():
        utiles_s = imp_senal[imp_senal > 0].sort_values(ascending=False)
        print(f"  Variables que mejoran las señales fuera del entrenamiento: {len(utiles_s)} de {nf}")
        print(f"    {', '.join(utiles_s.index)}")
    return imp_precio.fillna(0)


def _grafica_ventana_hoy(modelo, imp_precio, ctx):
    print(f"\nGRÁFICA 8: {ctx.ticker} - Lo que ve la red hoy")
    hist = _historia()
    fecha = hist.index[-1]
    Xs, _ = armar_secuencias(hist.index[-1:], modelo['scaler_x'], modelo['ventana'])
    rend, proba = _predecir_secuencias(modelo, Xs)
    cambio = (np.exp(rend[0, -1]) - 1) * 100

    top = imp_precio.sort_values(ascending=False).head(N_VARIABLES_VENTANA).index.tolist()
    cols = [ctx.features.index(f) for f in top]
    fechas_v = hist.index[-modelo['ventana']:]
    fig, ax = plt.subplots(figsize=(15, 6))
    for c, nombre in zip(cols, top):
        ax.plot(fechas_v, Xs[0, -len(fechas_v):, c], linewidth=2, label=nombre)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Valor estandarizado (0 = promedio del entrenamiento)')
    ax.set_title(f'{ctx.ticker} - Últimos {modelo["ventana"]} días de las {len(top)} variables más importantes\n'
                 f'Predicción a {ctx.dias} días: {cambio:+.2f} % | Probabilidades: SUBE {proba[0, 2]:.0%}, '
                 f'MANTENER {proba[0, 1]:.0%}, BAJA {proba[0, 0]:.0%}', fontsize=12)
    ax.legend(loc='upper left'); ax.grid(True, alpha=0.3); ax.tick_params(axis='x', rotation=45)
    plt.tight_layout(); plt.show(block=False); plt.pause(0.1)
    print(f"  Datos al {fecha.date()}: cambio esperado {cambio:+.2f} %; SUBE {proba[0, 2]:.0%}, "
          f"MANTENER {proba[0, 1]:.0%}, BAJA {proba[0, 0]:.0%}")


def graficas_propias(modelo, ctx):
    imprimir_parametros(modelo)
    _grafica_aprendizaje(modelo, ctx)
    imp_precio = _grafica_importancia(modelo, ctx)
    _grafica_ventana_hoy(modelo, imp_precio, ctx)


# =====================================================================
# SECCIÓN 6. EJECUCIÓN
# =====================================================================
if __name__ == '__main__':
    motor_prediccion.N_FOLDS = PERIODOS_WALK_FORWARD
    ejecutar_modelo(
        nombre="LSTM",
        entrenar=entrenar,
        predecir=predecir,
        nombre_precio="LSTM (salida de precio)",
        nombre_senal="LSTM (salida de señal)",
        opciones_precio=_opciones(LSTM_CONFIG, OPCIONES_LSTM),
        opciones_senal=[{}],          # la red es una sola: su configuración va en opciones_precio
        graficas_propias=graficas_propias,
        features=FEATURES,
        tres_clases=True,             # señal = clase más probable de BAJA / MANTENER / SUBE
    )
