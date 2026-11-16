#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
motor_prediccion.py
===================
Motor común para todos los modelos de predicción de acciones.

Este archivo NO se ejecuta directamente. Cada modelo (modelo_regresion.py,
modelo_arbol.py, ...) define solo cómo entrena y cómo predice, y llama a
ejecutar_modelo(). El motor se encarga de lo que es igual para todos:

    1. Pedir ticker, periodo y horizonte
    2. Actualizar MySQL y cargar datos            (db_acciones.py)
    3. Calcular features estacionarias
    4. Crear targets (rendimiento por día, señal de subida y de bajada)
    5. Validación walk-forward con purga anti-fuga y, si el modelo tiene
       configuraciones (ej. profundidad de un árbol), selección en un
       conjunto de validación interno
    6. Métricas, gráficas de validación y clasificación
    7. Re-entrenamiento con todos los datos y predicción desde hoy
    8. Tabla de predicciones y resumen ejecutivo

Contrato que debe cumplir cada modelo
-------------------------------------
entrenar(X, Y_dias, y_sube, y_baja, p_precio, p_senal) -> modelo
    X       : DataFrame de features (índice = fechas)
    Y_dias  : DataFrame con el rendimiento logarítmico de cada día 1..N
    y_sube  : Serie 0/1 (sube más de 3 % al final del horizonte)
    y_baja  : Serie 0/1 (baja más de 3 % al final del horizonte)
    p_precio, p_senal : diccionarios de configuración (de las listas de opciones)

predecir(modelo, X) -> (rend_dias, p_sube, p_baja)
    rend_dias : array (n_filas, N) con el rendimiento predicho de cada día
    p_sube    : array (n_filas,) probabilidad de subir más de 3 % (o NaN)
    p_baja    : array (n_filas,) probabilidad de bajar más de 3 % (o NaN)

graficas_propias(modelo, ctx)   (opcional)
    Gráficas específicas del modelo. ctx contiene features, datos y moneda.

tres_clases                      (opcional)
    True si el modelo usa un solo clasificador BAJA / MANTENER / SUBE: la señal
    es la clase más probable. False (por defecto): umbral 0.5 en cada señal.

features                         (opcional)
    Lista de columnas a usar. Por defecto FEATURES_MODELO (22 variables);
    FEATURES_EXTENDIDAS tiene 47 (todos los indicadores como proporciones).
"""
import sys
import random
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, roc_auc_score, roc_curve,
                             precision_score, recall_score, f1_score)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from db_acciones import actualizar_ticker, cargar_dataset_mlp, mercado_de, MERCADOS

warnings.filterwarnings('ignore')

# ============================================
# PARÁMETROS GENERALES (iguales para todos los modelos)
# ============================================
SEED = 42
UMBRAL_SUBIDA = 0.03                 # COMPRAR si se espera subir más de 3 %
UMBRAL_BAJADA = 0.03                 # VENDER si se espera bajar más de 3 %
N_FOLDS = 5                          # periodos de prueba en walk-forward
MIN_FILAS_ENTRENAMIENTO = 250
FRACCION_VALIDACION = 0.20           # validación interna para elegir configuraciones
MAX_ETIQUETAS = 45                   # con más puntos, los valores se ven al pasar el cursor
GUARDAR_CSV = False

# William%R no se usa: es exactamente SO%K - 100 (colinealidad perfecta)
FEATURES_MODELO = [
    'Ret_1', 'Ret_5', 'Ret_20', 'Ret_60',
    'Dist_MA20', 'Dist_MA50', 'Dist_MA200', 'MA50_vs_MA200',
    'MACD_norm', 'MACD_hist', 'ATR_pct', 'Volatilidad20', 'BB_pos',
    'RSI', 'SO%K', 'CCI', 'ROC', 'Vol_ratio',
    'QQQ_dist_MA50', 'QQQ_ret5', 'SnP_ret5', 'DJIA_ret5',
]

# Lista extendida: todos los indicadores de dataset_<mercado> expresados como
# proporciones, más variables de calendario. Útil para modelos de árboles, que
# toleran muchas variables y colinealidad. NO se incluyen niveles de precio
# (Close, MA, EMA, QQQ_Close... en dinero), ni ADX (es constante), ni William%R
# (= SO%K - 100), ni Year / DayofYear / Week (no se repiten en el futuro o
# duplican a Month).
FEATURES_EXTENDIDAS = FEATURES_MODELO + [
    'Ret_2', 'Ret_3', 'Gap_apertura', 'Rango_dia',
    'Dist_MA5', 'Dist_MA10',
    'Dist_EMA10', 'Dist_EMA20', 'Dist_EMA50', 'Dist_EMA100', 'Dist_EMA200',
    'STD5_pct', 'Force20_norm',
    'QQQ_ret1', 'QQQ_ret2', 'QQQ_dist_MA10', 'QQQ_dist_MA20',
    'SnP_ret1', 'DJIA_ret1',
    'DayofWeek', 'Month', 'Is_month_end', 'Is_month_start',
    'Is_quarter_end', 'Is_quarter_start',
]


# Histórico completo de features (todas las fechas del periodo, con y sin
# respuesta). Lo llena ejecutar_modelo(). Lo usan los modelos de secuencias
# (LSTM) para armar la ventana de días previos a cada fecha. No es fuga de
# datos: las features de un día solo usan información de ese día o anterior.
HISTORIA_FEATURES = None


# ============================================
# UTILIDADES PARA LOS MODELOS
# ============================================
def probabilidad(modelo, X):
    """predict_proba de la clase 1, o NaN si el clasificador no existe."""
    if modelo is None:
        return np.full(len(X), np.nan)
    return modelo.predict_proba(X)[:, 1]


def decidir_senal(p_sube, p_baja, tres_clases=False):
    """Convierte las probabilidades en COMPRAR / MANTENER / VENDER.

    tres_clases=False (dos clasificadores independientes, ej. regresión logística):
        COMPRAR si P(sube) >= 0.5, VENDER si P(baja) >= 0.5.
    tres_clases=True (un clasificador BAJA / MANTENER / SUBE, ej. árbol y bosque):
        gana la clase más probable. Con tres clases la probabilidad se reparte
        (ej. 37 / 34 / 30 %) y casi nunca llega a 0.5, así que el umbral fijo
        haría que el modelo dijera MANTENER casi siempre."""
    if tres_clases:
        if np.isnan(p_sube) or np.isnan(p_baja):
            return 'MANTENER'
        p_mant = 1 - p_sube - p_baja
        mayor = max(p_sube, p_baja, p_mant)
        if p_sube == mayor:
            return 'COMPRAR'
        if p_baja == mayor:
            return 'VENDER'
        return 'MANTENER'
    if not np.isnan(p_sube) and p_sube >= 0.5 and (np.isnan(p_baja) or p_sube >= p_baja):
        return 'COMPRAR'
    if not np.isnan(p_baja) and p_baja >= 0.5:
        return 'VENDER'
    return 'MANTENER'


def _texto_params(p_precio, p_senal, claves_p=None, claves_s=None):
    """Texto de una configuración. Si se indican claves, solo muestra esas
    (las que cambian entre opciones), para que la tabla sea legible."""
    partes = [f"{k}={v}" for k, v in p_precio.items() if claves_p is None or k in claves_p]
    partes += [f"clf_{k}={v}" for k, v in p_senal.items() if claves_s is None or k in claves_s]
    return ", ".join(partes) if partes else "-"


def _claves_variables(opciones):
    """Parámetros cuyo valor cambia entre las configuraciones candidatas."""
    claves = set().union(*[o.keys() for o in opciones])
    return {k for k in claves if len({repr(o.get(k)) for o in opciones}) > 1}


def metricas_binarias(y, pred, proba):
    """Precision, Recall, F1, ROC-AUC y matriz de confusión de una señal 0/1.
    Precision: de las veces que el modelo dio la señal, % en que acertó.
    Recall   : de los casos reales, % que el modelo detectó.
    F1       : media armónica de Precision y Recall (0 a 1).
    ROC-AUC  : capacidad de ordenar casos con la probabilidad (0.5 = azar)."""
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    proba = np.asarray(proba, dtype=float)
    hay_dos = len(np.unique(y)) == 2
    return {
        'Precision': precision_score(y, pred, zero_division=0) if pred.sum() > 0 else np.nan,
        'Recall': recall_score(y, pred, zero_division=0) if y.sum() > 0 else np.nan,
        'F1': f1_score(y, pred, zero_division=0) if (y.sum() + pred.sum()) > 0 else np.nan,
        'ROC_AUC': roc_auc_score(y, proba) if hay_dos and not np.isnan(proba).all() else np.nan,
        'tp': int(((pred == 1) & (y == 1)).sum()), 'fp': int(((pred == 1) & (y == 0)).sum()),
        'tn': int(((pred == 0) & (y == 0)).sum()), 'fn': int(((pred == 0) & (y == 1)).sum()),
    }


def _fmt(v, pct=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/d"
    return f"{v:.2%}" if pct else f"{v:.3f}"


def _titulo(texto):
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


# ============================================
# 1. ENTRADA DEL USUARIO
# ============================================
def _pedir_parametros(nombre_modelo):
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print(f"SISTEMA DE PREDICCIÓN DE ACCIONES - {nombre_modelo.upper()}")
    print("=" * 70)
    print(f"Semilla fijada: {SEED} (resultados reproducibles)")
    print("=" * 70)

    print("\nINGRESA LOS DATOS DE LA ACCIÓN:")
    print("-" * 50)
    ticker = input("Símbolo de la acción (ej: AAPL, MSFT, WALMEX.MX): ").upper().strip()
    temp = input("Periodo de datos a usar (ej: 5y, 10y, max): ").strip().lower()
    while True:
        try:
            dias = int(input("Días a predecir (ej: 30, 60, 90, 180, 365): "))
            if dias > 0:
                break
            print("   AVISO: ingresa un número positivo.")
        except ValueError:
            print("   AVISO: ingresa un número válido.")
    return ticker, temp, dias


# ============================================
# 2-3. DATOS Y FEATURES
# ============================================
def calcular_features(df):
    """Agrega las features estacionarias (proporciones, rendimientos, osciladores)."""
    c = df['Close']
    df['Ret_1'] = c / df['S_Close(t-1)'] - 1
    df['Ret_5'] = c / df['S_Close(t-5)'] - 1
    df['Ret_20'] = c.pct_change(20)
    df['Ret_60'] = c.pct_change(60)
    df['Dist_MA20'] = c / df['MA20'] - 1
    df['Dist_MA50'] = c / df['MA50'] - 1
    df['Dist_MA200'] = c / df['MA200'] - 1
    df['MA50_vs_MA200'] = df['MA50'] / df['MA200'] - 1
    df['MACD_norm'] = df['MACD'] / c
    df['MACD_hist'] = (df['MACD'] - df['MACD_EMA']) / c
    df['ATR_pct'] = df['ATR'] / c
    df['Volatilidad20'] = df['SD20'] / df['MA20']
    df['BB_pos'] = (c - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])
    df['Vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['QQQ_dist_MA50'] = df['QQQ_Close'] / df['QQQ_MA50'] - 1
    df['QQQ_ret5'] = df['QQQ_Close'] / df['QQQ(t-5)'] - 1
    df['SnP_ret5'] = df['SnP_Close'] / df['SnP(t-5)'] - 1
    df['DJIA_ret5'] = df['DJIA_Close'] / df['DJIA(t-5)'] - 1
    # --- extendidas ---
    df['Ret_2'] = c / df['S_Close(t-2)'] - 1
    df['Ret_3'] = c / df['S_Close(t-3)'] - 1
    df['Gap_apertura'] = df['Open'] / df['S_Close(t-1)'] - 1
    df['Rango_dia'] = (df['High'] - df['Low']) / c
    for p in [5, 10]:
        df[f'Dist_MA{p}'] = c / df[f'MA{p}'] - 1
    for p in [10, 20, 50, 100, 200]:
        df[f'Dist_EMA{p}'] = c / df[f'EMA{p}'] - 1
    df['STD5_pct'] = df['STD5'] / c
    df['Force20_norm'] = df['ForceIndex20'] / (c * df['Volume'].rolling(20).mean())
    df['QQQ_ret1'] = df['QQQ_Close'] / df['QQQ(t-1)'] - 1
    df['QQQ_ret2'] = df['QQQ_Close'] / df['QQQ(t-2)'] - 1
    df['QQQ_dist_MA10'] = df['QQQ_Close'] / df['QQQ_MA10'] - 1
    df['QQQ_dist_MA20'] = df['QQQ_Close'] / df['QQQ_MA20'] - 1
    df['SnP_ret1'] = df['SnP_Close'] / df['SnP(t-1))'] - 1
    df['DJIA_ret1'] = df['DJIA_Close'] / df['DJIA(t-1))'] - 1
    return df.replace([np.inf, -np.inf], np.nan)


def _grafica_historico(df, ticker, moneda):
    _titulo(f"GRÁFICA 1: {ticker} - Precio Histórico")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], mode='lines', name='Precio de Cierre',
        line=dict(color='#1f77b4', width=2), fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.1)',
        hovertemplate='<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Precio:</b> %{y:.2f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA50',
                             line=dict(color='purple', width=1.5, dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], mode='lines', name='MA200',
                             line=dict(color='orange', width=1.5, dash='dash')))
    pmax, pmin = df['Close'].max(), df['Close'].min()
    fmax, fmin = df['Close'].idxmax(), df['Close'].idxmin()
    fig.add_trace(go.Scatter(x=[fmax], y=[pmax], mode='markers+text', name=f'Máximo: {pmax:.2f}',
                             marker=dict(color='green', size=10), text=[f' {pmax:.2f}'],
                             textposition='top center'))
    fig.add_trace(go.Scatter(x=[fmin], y=[pmin], mode='markers+text', name=f'Mínimo: {pmin:.2f}',
                             marker=dict(color='red', size=10), text=[f' {pmin:.2f}'],
                             textposition='bottom center'))
    fig.update_layout(
        title=f'<b>{ticker} - Precio de Cierre Histórico</b><br>'
              f'<sub>{df.index[0].date()} a {df.index[-1].date()}</sub>',
        xaxis_title='Fecha', yaxis_title=f'Precio ({moneda})', height=600, width=1300,
        template='plotly_white', hovermode='x unified', xaxis=dict(rangeslider=dict(visible=True)))
    fig.show()


# ============================================
# 4. TARGETS
# ============================================
def crear_targets(df, n_habiles, features=FEATURES_MODELO):
    c = df['Close']
    targets_dia = [f'Rend_d{j}' for j in range(1, n_habiles + 1)]
    rend = pd.DataFrame({f'Rend_d{j}': np.log(c.shift(-j) / c) for j in range(1, n_habiles + 1)},
                        index=df.index)
    df = pd.concat([df, rend], axis=1)
    df['Precio_Futuro'] = c.shift(-n_habiles)
    df['Fecha_Objetivo'] = pd.Series(df.index, index=df.index).shift(-n_habiles)
    cambio = df['Precio_Futuro'] / c - 1
    df['Señal_Subida'] = np.where(cambio.isna(), np.nan, (cambio > UMBRAL_SUBIDA).astype(float))
    df['Señal_Bajada'] = np.where(cambio.isna(), np.nan, (cambio < -UMBRAL_BAJADA).astype(float))

    limpio = df[list(features) + targets_dia +
                ['Close', 'Precio_Futuro', 'Fecha_Objetivo', 'Señal_Subida', 'Señal_Bajada']].dropna()
    d = SimpleNamespace(
        df=df, targets_dia=targets_dia, col_h=f'Rend_d{n_habiles}',
        X=limpio[list(features)], Y=limpio[targets_dia],
        y_sube=limpio['Señal_Subida'].astype(int), y_baja=limpio['Señal_Bajada'].astype(int),
        close=limpio['Close'], precio_fut=limpio['Precio_Futuro'],
        fecha_obj=pd.to_datetime(limpio['Fecha_Objetivo']), fechas=limpio.index)
    return d


# ============================================
# 5. PARTICIONES, AJUSTE Y EVALUACIÓN
# ============================================
def _purgar(d, f_train, inicio_siguiente):
    """Quita del entrenamiento las filas cuyo precio objetivo cae en o después
    del inicio del siguiente bloque (validación o prueba): evita fuga de datos."""
    return f_train[d.fecha_obj.loc[f_train].values < inicio_siguiente]


def _particion_prueba(d, fin_test, n_test):
    ini = fin_test - n_test
    f_test = d.fechas[ini:fin_test]
    f_train = _purgar(d, d.fechas[:ini], f_test[0])
    return f_train, f_test


def _entrenar_en(m, d, fechas_tr, p_precio, p_senal):
    return m.entrenar(d.X.loc[fechas_tr], d.Y.loc[fechas_tr], d.y_sube.loc[fechas_tr],
                      d.y_baja.loc[fechas_tr], p_precio, p_senal)


def _auc_prom(d, fechas_ev, p_sube, p_baja):
    vals = []
    for y, p in [(d.y_sube.loc[fechas_ev].values, p_sube), (d.y_baja.loc[fechas_ev].values, p_baja)]:
        if len(np.unique(y)) == 2 and not np.isnan(p).all():
            vals.append(roc_auc_score(y, p))
    return np.mean(vals) if vals else np.nan


def _ajustar(m, d, f_train):
    """Entrena con f_train. Si hay varias configuraciones, primero las compara en
    un bloque de validación al final de f_train (con purga) y elige la mejor:
    - precio: menor MAE en validación
    - señal : mayor AUC promedio (subida y bajada) en validación
    Devuelve (modelo, p_precio, p_senal)."""
    op_p, op_s = m.opciones_precio, m.opciones_senal
    p_precio, p_senal = op_p[0], op_s[0]

    if len(op_p) > 1 or len(op_s) > 1:
        n_val = max(20, int(len(f_train) * FRACCION_VALIDACION))
        f_val = f_train[-n_val:]
        f_in = _purgar(d, f_train[:-n_val], f_val[0])
        if len(f_in) >= MIN_FILAS_ENTRENAMIENTO and d.y_sube.loc[f_in].nunique() == 2:
            X_val = d.X.loc[f_val]
            real_val = d.precio_fut.loc[f_val].values
            hoy_val = d.close.loc[f_val].values
            if len(op_p) > 1:
                maes = []
                for op in op_p:
                    rend, _, _ = m.predecir(_entrenar_en(m, d, f_in, op, op_s[0]), X_val)
                    maes.append(mean_absolute_error(real_val, hoy_val * np.exp(rend[:, -1])))
                p_precio = op_p[int(np.argmin(maes))]
            if len(op_s) > 1:
                aucs = []
                for op in op_s:
                    _, ps, pb = m.predecir(_entrenar_en(m, d, f_in, p_precio, op), X_val)
                    a = _auc_prom(d, f_val, ps, pb)
                    aucs.append(-np.inf if np.isnan(a) else a)
                p_senal = op_s[int(np.argmax(aucs))]

    return _entrenar_en(m, d, f_train, p_precio, p_senal), p_precio, p_senal


def _evaluar(m, d, f_train, f_test):
    modelo, p_precio, p_senal = _ajustar(m, d, f_train)
    X_te = d.X.loc[f_test]
    rend, p_sube, p_baja = m.predecir(modelo, X_te)
    precio_hoy = d.close.loc[f_test].values
    precio_pred = precio_hoy * np.exp(rend[:, -1])
    precio_real = d.precio_fut.loc[f_test].values
    ys, yb = d.y_sube.loc[f_test].values, d.y_baja.loc[f_test].values
    senales_te = [decidir_senal(a, b, m.tres_clases) for a, b in zip(p_sube, p_baja)]
    pred_sube = np.array([s == 'COMPRAR' for s in senales_te], dtype=int)
    pred_baja = np.array([s == 'VENDER' for s in senales_te], dtype=int)
    mayoritaria = int(d.y_sube.loc[f_train].mode().iloc[0])
    ms, mb = metricas_binarias(ys, pred_sube, p_sube), metricas_binarias(yb, pred_baja, p_baja)
    r = {
        'inicio': f_test[0].date(), 'fin': f_test[-1].date(), 'n_train': len(f_train),
        'MAE': mean_absolute_error(precio_real, precio_pred),
        'MAE_base': mean_absolute_error(precio_real, precio_hoy),
        'Acc': accuracy_score(ys, pred_sube),
        'Acc_base': np.mean(ys == mayoritaria),
        'Prec': ms['Precision'], 'Rec': ms['Recall'], 'F1': ms['F1'], 'AUC_sube': ms['ROC_AUC'],
        'Prec_b': mb['Precision'], 'Rec_b': mb['Recall'], 'F1_b': mb['F1'], 'AUC_baja': mb['ROC_AUC'],
        'Tasa_subida': ys.mean(), 'Tasa_bajada': yb.mean(),
        'config': _texto_params(p_precio, p_senal, m.claves_p, m.claves_s),
    }
    extra = dict(precio_pred=precio_pred, pred_sube=pred_sube, pred_baja=pred_baja,
                 p_sube=p_sube, p_baja=p_baja,
                 residuos=d.Y.loc[f_test, d.col_h].values - rend[:, -1])
    return r, extra


# ============================================
# GRÁFICAS DE VALIDACIÓN
# ============================================
def _grafica_validacion(ctx):
    _titulo(f"GRÁFICA 2: {ctx.ticker} - Validación de Precio")
    HOVER = '<b>%{fullData.name}</b><br>Fecha: %{x|%Y-%m-%d}<br>Precio: %{y:.2f}<extra></extra>'
    x = ctx.d.fecha_obj.loc[ctx.f_test]
    real, pred = ctx.y_precio_test, ctx.y_precio_pred
    errors_pct = (pred - real.values) / real.values * 100
    modo = 'lines+markers+text' if len(ctx.f_test) <= MAX_ETIQUETAS else 'lines+markers'

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=x, y=real, mode=modo, text=[f'{v:.2f}' for v in real],
                             textposition='top center', name='Precio Real',
                             line=dict(color='green', width=2.5), hovertemplate=HOVER), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=pred, mode=modo, text=[f'{v:.2f}' for v in pred],
                             textposition='bottom center', name='Predicción',
                             line=dict(color='red', width=2, dash='dash'), hovertemplate=HOVER),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=errors_pct, name='Error %', opacity=0.6,
                         marker_color=['green' if e < 0 else 'red' for e in errors_pct],
                         hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Error: %{y:.2f}%<extra></extra>'),
                  row=2, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="black", row=2, col=1)
    fig.update_layout(title=f'{ctx.ticker} - Validación a {ctx.dias} días ({ctx.nombre})',
                      height=600, template='plotly_white', hovermode='x unified')
    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_yaxes(title_text=f"Precio ({ctx.moneda})", row=1, col=1)
    fig.update_yaxes(title_text="Error (%)", row=2, col=1)
    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
                       text=f'MAE: {ctx.mae:.2f} | RMSE: {ctx.rmse:.2f} | R²: {ctx.r2:.4f} | '
                            f'MAE línea base: {ctx.mae_naive:.2f}',
                       showarrow=False, bgcolor="white", bordercolor="gray", borderwidth=1)
    fig.show()


def _grafica_clasificacion(ctx):
    _titulo(f"GRÁFICA 3: {ctx.ticker} - Análisis de Clasificación")
    f_test, wf = ctx.f_test, ctx.wf
    y_true, y_pred = ctx.y_senal_test.values, ctx.y_senal_pred
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    ax1 = axes[0, 0]
    precios = ctx.d.close.loc[f_test]
    ok = (y_pred == y_true)
    ax1.plot(f_test, precios, color='blue', alpha=0.7, linewidth=1.5)
    ax1.scatter(f_test[ok], precios[ok], color='green', s=30, label=f'Correctos ({ok.sum()})', alpha=0.7)
    ax1.scatter(f_test[~ok], precios[~ok], color='red', s=40, label=f'Incorrectos ({(~ok).sum()})', alpha=0.7)
    ax1.set_title(f'{ctx.ticker} - Clasificación (Accuracy: {ctx.accuracy:.1%})')
    ax1.set_ylabel(f'Precio ({ctx.moneda})')
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.tick_params(axis='x', rotation=45)

    ax2 = axes[0, 1]
    ax2.plot(f_test, ctx.p_sube_test, 'o-', color='purple', alpha=0.7, markersize=3, label='Prob. subida')
    ax2.plot(f_test, ctx.p_baja_test, 'o-', color='orange', alpha=0.6, markersize=3, label='Prob. bajada')
    ax2.scatter(f_test, y_true, color=['red' if y == 0 else 'green' for y in y_true],
                s=30, alpha=0.6, label='Real subida (0/1)')
    ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='Umbral 0.5')
    ax2.set_title('Probabilidades de Predicción'); ax2.set_ylim(-0.1, 1.1)
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.tick_params(axis='x', rotation=45)

    ax3 = axes[1, 0]
    cm = np.array([[ctx.tn, ctx.fp], [ctx.fn, ctx.tp]])
    ax3.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14, fontweight='bold')
    ax3.set_title('Matriz de Confusión (subida)'); ax3.set_xlabel('Predicho'); ax3.set_ylabel('Real')
    ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['0 (No sube)', '1 (Sube)']); ax3.set_yticklabels(['0 (No sube)', '1 (Sube)'])

    ax4 = axes[1, 1]
    etiquetas = [str(v) for v in wf['inicio']]
    xw = np.arange(len(wf))
    ax4.bar(xw - 0.2, wf['AUC_sube'], width=0.4, label='AUC subida', color='purple', alpha=0.7)
    ax4.bar(xw + 0.2, wf['AUC_baja'], width=0.4, label='AUC bajada', color='orange', alpha=0.7)
    ax4.axhline(0.5, color='red', linestyle='--', label='Azar (0.5)')
    ax4.set_xticks(xw); ax4.set_xticklabels(etiquetas, rotation=45)
    ax4.set_ylim(0, 1); ax4.set_title('Walk-forward: AUC por periodo de prueba')
    ax4.legend(); ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'{ctx.ticker} - Resultados de Clasificación Binaria ({ctx.nombre})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.show(block=False); plt.pause(0.1)


def _grafica_estadistica(ctx):
    _titulo(f"GRÁFICA 4: {ctx.ticker} - Resumen Estadístico")
    real, pred, wf = ctx.y_precio_test, ctx.y_precio_pred, ctx.wf
    errores = pred - real.values
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(errores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=errores.mean(), color='red', linestyle='--', label=f'Media: {errores.mean():.2f}')
    axes[0, 0].axvline(x=0, color='black', alpha=0.5)
    axes[0, 0].set_title('Distribución de Errores'); axes[0, 0].set_xlabel(f'Error ({ctx.moneda})')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(real, pred, alpha=0.6, edgecolors='black', linewidth=0.5)
    lo, hi = min(real.min(), pred.min()), max(real.max(), pred.max())
    axes[0, 1].plot([lo, hi], [lo, hi], 'r--', alpha=0.7, label='Predicción Perfecta')
    axes[0, 1].set_title('Real vs Predicho'); axes[0, 1].set_xlabel(f'Real ({ctx.moneda})')
    axes[0, 1].set_ylabel(f'Predicho ({ctx.moneda})'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    etiquetas = [str(v) for v in wf['inicio']]
    xw = np.arange(len(wf))
    axes[1, 0].bar(xw - 0.2, wf['MAE'], width=0.4, label='MAE modelo', color='blue', alpha=0.7)
    axes[1, 0].bar(xw + 0.2, wf['MAE_base'], width=0.4, label='MAE línea base', color='gray', alpha=0.7)
    axes[1, 0].set_xticks(xw); axes[1, 0].set_xticklabels(etiquetas, rotation=45)
    axes[1, 0].set_title(f'Walk-forward: MAE ({ctx.moneda}) modelo vs línea base')
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3, axis='y')
    # Curva ROC con las predicciones fuera de muestra de TODOS los periodos del
    # walk-forward: un solo periodo suele tener una sola clase (ej. la acción
    # subió todos los días) y la curva no se puede trazar.
    ax = axes[1, 1]
    trazadas = 0
    for nombre_s, y, p, color in [('Subida (COMPRAR)', ctx.roc_y_sube, ctx.roc_p_sube, 'purple'),
                                  ('Bajada (VENDER)', ctx.roc_y_baja, ctx.roc_p_baja, 'orange')]:
        ok = ~np.isnan(p)
        y, p = y[ok], p[ok]
        if len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=color, linewidth=2.5,
                    label=f'{nombre_s}: AUC = {roc_auc_score(y, p):.3f}')
            trazadas += 1
        else:
            ax.plot([], [], color=color, label=f'{nombre_s}: sin casos de ambas clases')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Azar (AUC = 0.5)')
    if trazadas == 0:
        ax.text(0.5, 0.55, 'No hay casos de ambas clases en los periodos\nde prueba: la curva no se puede calcular',
                ha='center', va='center', fontsize=10, color='gray')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel('Tasa de falsos positivos'); ax.set_ylabel('Tasa de verdaderos positivos (Recall)')
    ax.set_title(f'Curva ROC (todos los periodos de prueba, {len(ctx.roc_y_sube)} días)')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)

    plt.suptitle(f'{ctx.ticker} - Análisis Estadístico ({ctx.dias} días, {ctx.nombre})',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.show(block=False); plt.pause(0.1)


# ============================================
# PROGRAMA PRINCIPAL
# ============================================
def ejecutar_modelo(nombre, entrenar, predecir, nombre_precio, nombre_senal,
                    opciones_precio=None, opciones_senal=None, graficas_propias=None,
                    features=None, tres_clases=False):
    """Corre el flujo completo para un modelo. Ver el contrato al inicio del archivo."""
    features = list(features or FEATURES_MODELO)
    m = SimpleNamespace(entrenar=entrenar, predecir=predecir,
                        opciones_precio=opciones_precio or [{}],
                        opciones_senal=opciones_senal or [{}])
    m.tres_clases = tres_clases
    m.claves_p = _claves_variables(m.opciones_precio)
    m.claves_s = _claves_variables(m.opciones_senal)
    plt.ion()

    # ---------- 1. Entrada ----------
    ticker, temp, dias = _pedir_parametros(nombre)
    moneda = MERCADOS[mercado_de(ticker)]['moneda']
    n_habiles = max(1, round(dias * 252 / 365))
    n_test = max(20, n_habiles)

    _titulo("CONFIGURACIÓN SELECCIONADA:")
    print(f"   - Modelo: {nombre}")
    print(f"   - Empresa: {ticker} ({moneda})")
    print(f"   - Temporalidad de los datos: {temp}")
    print(f"   - Horizonte de predicción: {dias} días ({n_habiles} días hábiles)")
    print(f"   - Semilla aleatoria: {SEED}")
    if len(m.opciones_precio) > 1 or len(m.opciones_senal) > 1:
        print(f"   - Configuraciones a comparar: {len(m.opciones_precio)} (precio), "
              f"{len(m.opciones_senal)} (señal)")
    else:
        if m.opciones_precio[0]:
            print(f"   - Configuración precio: {_texto_params(m.opciones_precio[0], {})}")
        if m.opciones_senal[0]:
            print(f"   - Configuración señal: {_texto_params({}, m.opciones_senal[0])}")
    print(f"   - Variables: {len(features)}")

    # ---------- 2. Datos ----------
    _titulo("1. ACTUALIZANDO BASE DE DATOS (solo días nuevos)...")
    try:
        actualizar_ticker(ticker)
    except Exception as e:
        print(f"ERROR al actualizar {ticker}: {e}")
        print("   Verifica el símbolo y la conexión a MySQL")
        sys.exit(1)
    df = cargar_dataset_mlp(ticker, periodo=temp)
    if df.empty:
        print(f"ERROR: no se encontraron datos para {ticker}")
        sys.exit(1)
    if GUARDAR_CSV:
        df.reset_index().to_csv(f"{ticker}MLP.csv", index=False)
    print(f"Datos cargados: {df.index[0].date()} a {df.index[-1].date()} ({len(df)} días)")

    # ---------- 3. Features ----------
    _titulo("2. CALCULANDO FEATURES...")
    df = calcular_features(df)
    print(f"{len(features)} features usadas por el modelo (proporciones, rendimientos y osciladores)")
    _grafica_historico(df, ticker, moneda)
    global HISTORIA_FEATURES
    HISTORIA_FEATURES = df[features].dropna()

    # ---------- 4. Targets ----------
    _titulo("3. CREANDO TARGETS...")
    d = crear_targets(df, n_habiles, features)
    print(f"Targets creados a {n_habiles} días hábiles:")
    print(f"  Precio_Futuro: {len(d.fechas)} fechas")
    print(f"  Señal_Subida (> +{UMBRAL_SUBIDA:.0%}): {int((d.y_sube == 0).sum())} no suben, "
          f"{int(d.y_sube.sum())} suben")
    print(f"  Señal_Bajada (< -{UMBRAL_BAJADA:.0%}): {int((d.y_baja == 0).sum())} no bajan, "
          f"{int(d.y_baja.sum())} bajan")
    if len(d.fechas) < n_test + n_habiles + MIN_FILAS_ENTRENAMIENTO:
        print("ERROR: datos insuficientes. Usa un periodo más largo (10y o max) o un horizonte menor.")
        sys.exit(1)

    # ---------- 5. Walk-forward ----------
    _titulo(f"4. VALIDACIÓN WALK-FORWARD (hasta {N_FOLDS} periodos de {n_test} días hábiles)")
    resultados, extras, splits = [], [], []
    for k in range(N_FOLDS):
        fin_test = len(d.fechas) - k * n_test
        if fin_test - n_test <= 0:
            break
        f_train, f_test = _particion_prueba(d, fin_test, n_test)
        if len(f_train) < MIN_FILAS_ENTRENAMIENTO or d.y_sube.loc[f_train].nunique() < 2:
            break
        r, e = _evaluar(m, d, f_train, f_test)
        resultados.append(r); extras.append(e); splits.append((f_train, f_test))
        print(f"  Periodo {f_test[0].date()} a {f_test[-1].date()} evaluado")
    if not resultados:
        print("ERROR: no hay suficientes datos para ningún periodo de prueba.")
        sys.exit(1)

    wf = pd.DataFrame(resultados).iloc[::-1].reset_index(drop=True)
    pd.set_option('display.width', 220)
    columnas = ['inicio', 'fin', 'n_train', 'MAE', 'MAE_base', 'Acc', 'Acc_base',
                'Prec', 'Rec', 'F1', 'AUC_sube', 'Prec_b', 'Rec_b', 'F1_b', 'AUC_baja']
    if len(m.opciones_precio) > 1 or len(m.opciones_senal) > 1:
        columnas.append('config')
    print()
    print(wf[columnas].to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep='n/d'))
    print("  (Prec/Rec/F1/AUC_sube: señal de subida | Prec_b/Rec_b/F1_b/AUC_baja: señal de bajada)")
    prom = wf.mean(numeric_only=True)
    print("\nPROMEDIO WALK-FORWARD:")
    print(f"  Precio : MAE {prom['MAE']:.2f} vs {prom['MAE_base']:.2f} de la línea base "
          f"-> {'el modelo la supera' if prom['MAE'] < prom['MAE_base'] else 'el modelo NO la supera'}")
    print(f"  Subida : AUC {prom['AUC_sube']:.3f} | Bajada: AUC {prom['AUC_baja']:.3f}  (0.5 = azar)")
    print(f"  Subida : Precision {_fmt(prom['Prec'])} | Recall {_fmt(prom['Rec'])} | "
          f"F1 {_fmt(prom['F1'], False)}")
    print(f"  Bajada : Precision {_fmt(prom['Prec_b'])} | Recall {_fmt(prom['Rec_b'])} | "
          f"F1 {_fmt(prom['F1_b'], False)}")
    print(f"  COMPRAR acierta {_fmt(prom['Prec'])} vs {prom['Tasa_subida']:.1%} comprando siempre")

    # ---------- 6. Métricas del periodo más reciente ----------
    _titulo("5. DIVIDIENDO DATOS (periodo de prueba más reciente)")
    ult = extras[0]
    f_train, f_test = splits[0]
    print(f"  Entrenamiento: {len(f_train)} días ({f_train[0].date()} a {f_train[-1].date()})")
    print(f"  Prueba: {len(f_test)} días ({f_test[0].date()} a {f_test[-1].date()})")

    ctx = SimpleNamespace(nombre=nombre, ticker=ticker, moneda=moneda, dias=dias, d=d, wf=wf,
                          f_train=f_train, f_test=f_test, features=features,
                          splits=splits)            # (entrenamiento, prueba) de cada periodo del walk-forward
    ctx.y_precio_test = d.precio_fut.loc[f_test]
    ctx.y_precio_pred = ult['precio_pred']
    ctx.y_senal_test = d.y_sube.loc[f_test]
    ctx.y_senal_pred = ult['pred_sube']
    ctx.p_sube_test, ctx.p_baja_test = ult['p_sube'], ult['p_baja']
    hoy_test = d.close.loc[f_test].values

    ctx.mae = mean_absolute_error(ctx.y_precio_test, ctx.y_precio_pred)
    ctx.rmse = np.sqrt(mean_squared_error(ctx.y_precio_test, ctx.y_precio_pred))
    ctx.r2 = r2_score(ctx.y_precio_test, ctx.y_precio_pred)
    ctx.mae_naive = mean_absolute_error(ctx.y_precio_test, hoy_test)
    rmse_naive = np.sqrt(mean_squared_error(ctx.y_precio_test, hoy_test))
    yt, yp = ctx.y_senal_test.values, ctx.y_senal_pred
    ctx.accuracy = accuracy_score(yt, yp)
    ctx.tp = int(((yp == 1) & (yt == 1)).sum()); ctx.fp = int(((yp == 1) & (yt == 0)).sum())
    ctx.tn = int(((yp == 0) & (yt == 0)).sum()); ctx.fn = int(((yp == 0) & (yt == 1)).sum())
    accuracy_base = resultados[0]['Acc_base']
    tp, fp, tn, fn = ctx.tp, ctx.fp, ctx.tn, ctx.fn
    ctx.ms = metricas_binarias(yt, yp, ctx.p_sube_test)
    ctx.y_baja_test = d.y_baja.loc[f_test].values
    ctx.mb = metricas_binarias(ctx.y_baja_test, ult['pred_baja'], ctx.p_baja_test)
    ms, mb = ctx.ms, ctx.mb
    # Predicciones fuera de muestra de todos los periodos (para la curva ROC)
    ctx.roc_y_sube = np.concatenate([d.y_sube.loc[ft].values for _, ft in splits])
    ctx.roc_y_baja = np.concatenate([d.y_baja.loc[ft].values for _, ft in splits])
    ctx.roc_p_sube = np.concatenate([np.asarray(e['p_sube'], dtype=float) for e in extras])
    ctx.roc_p_baja = np.concatenate([np.asarray(e['p_baja'], dtype=float) for e in extras])

    _titulo("6. MÉTRICAS")
    print(f"\nPRECIO ({nombre_precio}):")
    print(f"  MAE: {ctx.mae:.2f} | RMSE: {ctx.rmse:.2f} | R²: {ctx.r2:.4f}")
    print(f"  Línea base (precio de hoy): MAE {ctx.mae_naive:.2f} | RMSE {rmse_naive:.2f}"
          f"  -> {'el modelo la supera' if ctx.mae < ctx.mae_naive else 'el modelo NO la supera'}")
    print(f"\nSEÑAL DE SUBIDA - COMPRAR ({nombre_senal}):")
    print(f"  Accuracy: {ctx.accuracy:.2%}")
    print(f"  Línea base (siempre la clase más frecuente): {accuracy_base:.2%}")
    print(f"  Matriz: [[{tn} {fp}] [{fn} {tp}]]")
    print(f"  Precision: {_fmt(ms['Precision'])}")
    print(f"  Recall: {_fmt(ms['Recall'])}")
    print(f"  F1-score: {_fmt(ms['F1'], False)}")
    print(f"  ROC-AUC: {_fmt(ms['ROC_AUC'], False)}")
    print(f"\nSEÑAL DE BAJADA - VENDER ({nombre_senal}):")
    print(f"  Matriz: [[{mb['tn']} {mb['fp']}] [{mb['fn']} {mb['tp']}]]")
    print(f"  Precision: {_fmt(mb['Precision'])}")
    print(f"  Recall: {_fmt(mb['Recall'])}")
    print(f"  F1-score: {_fmt(mb['F1'], False)}")
    print(f"  ROC-AUC: {_fmt(mb['ROC_AUC'], False)}")
    print("  (n/d: no se puede calcular; el modelo nunca dio esa señal o no hubo casos reales en el periodo)")

    _grafica_validacion(ctx)
    _grafica_clasificacion(ctx)
    _grafica_estadistica(ctx)

    # ---------- 7. Modelo final y predicción desde hoy ----------
    _titulo(f"PREDICCIÓN FUTURA PARA {ticker}")
    modelo_f, p_precio_f, p_senal_f = _ajustar(m, d, d.fechas)
    if len(m.opciones_precio) > 1 or len(m.opciones_senal) > 1:
        print(f"Configuración elegida para el modelo final: "
              f"{_texto_params(p_precio_f, p_senal_f, m.claves_p, m.claves_s)}")

    if graficas_propias is not None:
        _titulo(f"GRÁFICAS PROPIAS DEL MODELO: {nombre}")
        ctx.X, ctx.Y = d.X, d.Y
        graficas_propias(modelo_f, ctx)

    filas_feat = df[features].dropna()
    ultima_fila = filas_feat.iloc[[-1]]
    ultima_fecha = ultima_fila.index[0]
    precio_actual = float(df.loc[ultima_fecha, 'Close'])
    sigma_h = np.std(np.concatenate([e['residuos'] for e in extras]))

    # --- (A) PRONÓSTICO RODANTE (principal, mismo método que la validación) ---
    # Cada uno de los últimos N días hábiles predice su precio a N días. El punto
    # se coloca en su fecha futura. Como cada punto parte de un día con
    # indicadores distintos, la curva muestra subidas y bajadas. El último punto
    # es la predicción hecha hoy.
    base = filas_feat.iloc[-n_habiles:]
    rend_r, ps_r, pb_r = m.predecir(modelo_f, base)
    precios = df.loc[base.index, 'Close'].values * np.exp(rend_r[:, -1])
    fechas_fut = base.index + pd.offsets.BDay(n_habiles)
    senales = [decidir_senal(a, b, m.tres_clases) for a, b in zip(ps_r, pb_r)]
    if m.tres_clases:   # confianza = probabilidad de la clase elegida
        confianzas = [a if s == 'COMPRAR' else b if s == 'VENDER' else 1 - a - b
                      for s, a, b in zip(senales, ps_r, pb_r)]
    else:
        confianzas = [a if s == 'COMPRAR' else b if s == 'VENDER' else 1 - np.nanmax([a, b])
                      for s, a, b in zip(senales, ps_r, pb_r)]
    banda_baja = precios * np.exp(-1.28 * sigma_h)
    banda_alta = precios * np.exp(1.28 * sigma_h)
    futuros = fechas_fut > ultima_fecha                # por días festivos, alguno puede ya haber pasado
    precios, fechas_fut = precios[futuros], fechas_fut[futuros]
    banda_baja, banda_alta = banda_baja[futuros], banda_alta[futuros]
    senales = [s for s, f in zip(senales, futuros) if f]
    confianzas = [c for c, f in zip(confianzas, futuros) if f]

    # --- (B) TRAYECTORIA ESPERADA DESDE HOY (secundaria, oculta en la leyenda) ---
    # Rendimiento esperado día por día usando solo la información de hoy.
    rend_tray = m.predecir(modelo_f, ultima_fila)[0][0]
    precios_tray = precio_actual * np.exp(rend_tray)
    fechas_tray = pd.bdate_range(ultima_fecha + pd.Timedelta(days=1), periods=n_habiles)

    # Señal final = la del último punto (predicción hecha hoy)
    senal_final, confianza_final = senales[-1], confianzas[-1]
    cambio_final = (precios[-1] / precio_actual - 1) * 100
    i_max = int(np.argmax(precios))
    venta_posible = precios[i_max] / precio_actual - 1 > UMBRAL_SUBIDA
    dias_vender = [f for f, s in zip(fechas_fut, senales) if s == 'VENDER']

    print(f"Último dato real: {ultima_fecha.date()} - Precio: {precio_actual:.2f} {moneda}")
    print(f"Predicción a {dias} días...")
    print("Predicción generada:")
    print(f"   - Período: {fechas_fut[0].date()} a {fechas_fut[-1].date()}")
    print(f"   - Precio final esperado: {precios[-1]:.2f} {moneda}")
    print(f"   - Rango probable (~80 %): {banda_baja[-1]:.2f} a {banda_alta[-1]:.2f} {moneda}")
    print(f"   - Cambio esperado: {cambio_final:+.2f}%")
    print(f"   - Señal: {senal_final}")
    print(f"   - Confianza: {confianza_final*100:.1f}%")
    if venta_posible:
        print(f"   - Posible venta: {fechas_fut[i_max].date()} a {precios[i_max]:.2f} {moneda} "
              f"({(precios[i_max]/precio_actual-1)*100:+.2f}%)")
    if dias_vender:
        print(f"   - Alerta de venta: se estima caída mayor a {UMBRAL_BAJADA:.0%} a partir del "
              f"{dias_vender[0].date()}")

    # ---------- Gráfica 5 ----------
    _titulo(f"GRÁFICA 5: {ticker} - Predicción Futura")
    colores = {'COMPRAR': 'green', 'MANTENER': 'gray', 'VENDER': 'red'}
    hist = df['Close'].iloc[-100:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, mode='lines', name='Datos Históricos',
                             line=dict(color='blue', width=2),
                             hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Precio: %{y:.2f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=fechas_fut, y=banda_alta, mode='lines', line=dict(width=0),
                             showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=fechas_fut, y=banda_baja, mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor='rgba(255,0,0,0.10)', name='Rango ~80 %',
                             hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=fechas_tray, y=precios_tray, mode='lines', visible='legendonly',
                             name='Trayectoria esperada desde hoy',
                             line=dict(color='black', width=1.5, dash='dot'),
                             hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Precio: %{y:.2f}<extra></extra>'))
    fig.add_trace(go.Scatter(
        x=fechas_fut, y=precios,
        mode='lines+markers+text' if n_habiles <= MAX_ETIQUETAS else 'lines+markers',
        text=[f'{v:.2f}' for v in precios], textposition='top center', textfont=dict(size=9),
        name=f'Predicción {dias} días', line=dict(color='red', width=1.5),
        marker=dict(size=6, color=[colores[s] for s in senales]),
        customdata=list(zip(((precios / precio_actual - 1) * 100).tolist(), senales)),
        hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Precio: %{y:.2f}<br>Cambio: %{customdata[0]:.2f}%'
                      '<br>Señal: %{customdata[1]}<extra></extra>'))
    fig.add_hline(y=precio_actual, line_dash="dash", line_color="gray",
                  annotation_text=f"Precio Actual: {precio_actual:.2f}")
    fig.add_trace(go.Scatter(x=[ultima_fecha], y=[precio_actual], mode='markers',
                             name='Fin Datos Reales', marker=dict(color='orange', size=10)))
    if venta_posible:
        fig.add_trace(go.Scatter(x=[fechas_fut[i_max]], y=[precios[i_max]], mode='markers+text',
                                 name='Posible venta',
                                 marker=dict(color='darkgreen', size=14, symbol='triangle-down'),
                                 text=[f'Venta {precios[i_max]:.2f}'], textposition='top center'))
    fig.update_layout(title=f'{ticker} - Predicción Futura a {dias} días ({nombre}) '
                            f'(verde: COMPRAR, gris: MANTENER, rojo: VENDER)',
                      xaxis_title='Fecha', yaxis_title=f'Precio ({moneda})', height=550,
                      template='plotly_white', hovermode='closest')
    fig.show()

    # ---------- Tabla ----------
    _titulo(f"TABLA DE PREDICCIONES - PRÓXIMOS {dias} DÍAS ({n_habiles} días hábiles)")
    print(f"{'Fecha':<15} {'Precio':<12} {'Cambio %':<12} {'Señal':<15} {'Confianza':<12}")
    print("-" * 70)
    for i in range(len(fechas_fut)):
        cambio = (precios[i] / precio_actual - 1) * 100
        print(f"{str(fechas_fut[i].date()):<15} {precios[i]:<12.2f} {cambio:+11.2f}% "
              f"{senales[i]:<15} {confianzas[i]*100:>5.1f}%")
    print("-" * 70)
    print(f"Cada fila es la predicción a {dias} días hecha el día hábil correspondiente "
          f"(pronóstico rodante); la última es la hecha hoy.")
    print("Señal y confianza: clasificador del modelo (COMPRAR / MANTENER / VENDER).")

    # ---------- Resumen ejecutivo ----------
    _titulo("RESUMEN EJECUTIVO")
    print("\nINFORMACIÓN GENERAL:")
    print(f"  - Modelo: {nombre}")
    print(f"  - Empresa: {ticker} ({moneda})")
    print(f"  - Período de datos: {df.index[0].date()} a {ultima_fecha.date()}")
    print(f"  - Precio actual: {precio_actual:.2f}")

    print(f"\nMODELO DE PRECIO ({nombre_precio}):")
    print(f"  - Error promedio (MAE): {ctx.mae:.2f}  (línea base: {ctx.mae_naive:.2f})")
    print(f"  - Raíz del error cuadrático (RMSE): {ctx.rmse:.2f}  (línea base: {rmse_naive:.2f})")
    print(f"  - Precisión del modelo (R²): {ctx.r2:.4f}")
    print(f"  - Walk-forward ({len(wf)} periodos): MAE {prom['MAE']:.2f} vs {prom['MAE_base']:.2f} línea base")

    print(f"\nMODELO DE SEÑAL ({nombre_senal}):")
    print(f"  - Aciertos totales: {ctx.accuracy:.2%}  (línea base: {accuracy_base:.2%})")
    if (tp + fp) > 0:
        print(f"  - Cuando dice COMPRAR, acierta: {tp/(tp+fp):.2%}")
    if (tp + fn) > 0:
        print(f"  - Detecta el {tp/(tp+fn):.2%} de las subidas reales")
    print(f"  - F1-score: {_fmt(ms['F1'], False)} | ROC-AUC: {_fmt(ms['ROC_AUC'], False)}")
    print(f"  - Matriz de confusión: [[{tn} {fp}] [{fn} {tp}]]")
    print(f"  - Cuando dice VENDER, acierta: {_fmt(mb['Precision'])}")
    print(f"  - Bajadas reales detectadas: {_fmt(mb['Recall'])}")
    print(f"  - Bajada: F1-score {_fmt(mb['F1'], False)} | ROC-AUC {_fmt(mb['ROC_AUC'], False)}")
    print(f"  - Walk-forward: ROC-AUC subida {_fmt(prom['AUC_sube'], False)}, "
          f"bajada {_fmt(prom['AUC_baja'], False)} (0.5 = azar) | "
          f"F1 subida {_fmt(prom['F1'], False)}, bajada {_fmt(prom['F1_b'], False)}")

    print(f"\nPREDICCIÓN FUTURA ({dias} días):")
    print(f"  - Precio esperado: {precios[-1]:.2f}")
    print(f"  - Rango probable: {banda_baja[-1]:.2f} a {banda_alta[-1]:.2f}")
    print(f"  - Cambio esperado: {cambio_final:+.2f}%")
    print(f"  - Señal: {senal_final}")
    print(f"  - Confianza: {confianza_final*100:.1f}%")
    if venta_posible:
        print(f"  - Posible venta: {fechas_fut[i_max].date()} a {precios[i_max]:.2f} "
              f"({(precios[i_max]/precio_actual-1)*100:+.2f}%)")
    else:
        print(f"  - Posible venta: no se estima una ganancia mayor a {UMBRAL_SUBIDA:.0%} en el horizonte")
    if dias_vender:
        print(f"  - Alerta de venta: caída estimada mayor a {UMBRAL_BAJADA:.0%} desde {dias_vender[0].date()}")

    print("\n" + "=" * 70)
    print("DISCLAIMER: Esta herramienta es solo para fines educativos.")
    print("   No constituye asesoramiento financiero profesional.")
    print("   Las decisiones de inversión deben tomarse con criterio propio.")
    print("=" * 70)

    plt.ioff()
    plt.show(block=True)
    return ctx
