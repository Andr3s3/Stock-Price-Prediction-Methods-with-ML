#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 00:18:39 2026

@author: andr3s
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import random
import sklearn
plt.ion()

warnings.filterwarnings('ignore')

# ============================================
# FIJAR SEMILLA PARA REPRODUCIBILIDAD
# ============================================
SEED = 42

# Fijar semilla para Python
random.seed(SEED)

# Fijar semilla para NumPy
np.random.seed(SEED)

# Fijar semilla para scikit-learn
sklearn.utils.check_random_state(SEED)

print("=" * 70)
print("SISTEMA DE PREDICCIÓN DE ACCIONES")
print("=" * 70)
print(f"🔒 Semilla fijada: {SEED} (resultados reproducibles)")
print("=" * 70)

# ============================================
# INPUT DEL USUARIO
# ============================================
print("\n📌 INGRESA LOS DATOS DE LA ACCIÓN:")
print("-" * 50)

# Solicitar ticker
TICKER = input("🔹 Símbolo de la acción (ej: AAPL, MSFT, GOOGL, AMZN): ").upper().strip()
TEMP = input("Ingresa el tiempo desde el que quieres bajar los datos(ej: 5d, 3wk, 3mo, 5y, max): ").strip().lower()
# Solicitar días a predecir
while True:
    try:
        DIAS_HORIZONTE = int(input("🔹 Días a predecir (ej: 30, 60, 90, 180, 365): "))
        if DIAS_HORIZONTE > 0:
            break
        else:
            print("   ⚠️ Por favor, ingresa un número positivo.")
    except ValueError:
        print("   ⚠️ Por favor, ingresa un número válido.")

print("\n" + "=" * 70)
print("🔍 CONFIGURACIÓN SELECCIONADA:")
print(f"   • Empresa: {TICKER}")
print(f"   • Temporalidad de los datos: {TEMP}")
print(f"   • Horizonte de predicción: {DIAS_HORIZONTE} días")
print(f"   • Semilla aleatoria: {SEED}")
print("=" * 70)

# Parámetros derivados
UMBRAL_SUBIDA = 0.03
DIAS_DE_PRUEBA = DIAS_HORIZONTE
DIAS_A_MOSTRAR = DIAS_HORIZONTE
PORCENTAJE_TEST = 0.2

# ============================================
# 1. DESCARGA Y PREPARACIÓN DE DATOS
# ============================================
print("\n" + "=" * 70)
print("1. DESCARGANDO DATOS DE YAHOO FINANCE...")
print("=" * 70)

print(f"📊 Descargando datos para {TICKER}...")


# Descargar datos
df = yf.download(TICKER, period=TEMP, progress=False)

if df.empty:
    print(f"❌ Error: No se encontraron datos para {TICKER}")
    print("   Verifica que el símbolo sea correcto (ej: AAPL, MSFT, GOOGL)")
    exit()

print(f"✓ Datos descargados: {len(df)} registros")
print(f"✓ Período: {df.index[0].date()} a {df.index[-1].date()}")

# ============================================
# 2. CREAR TODAS LAS COLUMNAS DEL DATASET
# ============================================
print("\n" + "=" * 70)
print("2. CALCULANDO INDICADORES TÉCNICOS...")
print("=" * 70)

COLUMNAS_MLP = [
    'Open', 'High', 'Low', 'Close', 'Volume', 
    'SD20', 'Upper_Band', 'Lower_Band', 
    'S_Close(t-1)', 'S_Close(t-2)', 'S_Close(t-3)', 'S_Close(t-5)', 'S_Open(t-1)',
    'MA5', 'MA10', 'MA20', 'MA50', 'MA200',
    'EMA10', 'EMA20', 'EMA50', 'EMA100', 'EMA200',
    'MACD', 'MACD_EMA', 'ATR', 'ADX', 'CCI', 'ROC', 'RSI',
    'William%R', 'SO%K', 'STD5',
    'ForceIndex1', 'ForceIndex20',
    'Date_col', 'Day', 'DayofWeek', 'DayofYear', 'Week',
    'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start',
    'Is_year_end', 'Is_year_start', 'Is_leap_year',
    'Year', 'Month',
    'QQQ_Close', 'QQQ(t-1)', 'QQQ(t-2)', 'QQQ(t-5)',
    'QQQ_MA10', 'QQQ_MA20', 'QQQ_MA50',
    'SnP_Close', 'SnP(t-1))', 'SnP(t-5)',
    'DJIA_Close', 'DJIA(t-1))', 'DJIA(t-5)',
    'Close_forcast'
]

# Inicializar DataFrame
df_mlp = pd.DataFrame(index=df.index, columns=COLUMNAS_MLP)

# Columnas básicas
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if col in df.columns:
        df_mlp[col] = df[col]

# Bandas de Bollinger
window = 20
df_mlp['MA20_temp'] = df_mlp['Close'].rolling(window=window).mean()
df_mlp['SD20'] = df_mlp['Close'].rolling(window=window).std()
df_mlp['Upper_Band'] = df_mlp['MA20_temp'] + (df_mlp['SD20'] * 2)
df_mlp['Lower_Band'] = df_mlp['MA20_temp'] - (df_mlp['SD20'] * 2)

# Precios retrasados
df_mlp['S_Close(t-1)'] = df_mlp['Close'].shift(1)
df_mlp['S_Close(t-2)'] = df_mlp['Close'].shift(2)
df_mlp['S_Close(t-3)'] = df_mlp['Close'].shift(3)
df_mlp['S_Close(t-5)'] = df_mlp['Close'].shift(5)
df_mlp['S_Open(t-1)'] = df_mlp['Open'].shift(1)

# Medias móviles simples
for period in [5, 10, 20, 50, 200]:
    df_mlp[f'MA{period}'] = df_mlp['Close'].rolling(window=period).mean()

# Medias móviles exponenciales
for period in [10, 20, 50, 100, 200]:
    df_mlp[f'EMA{period}'] = df_mlp['Close'].ewm(span=period, adjust=False).mean()

# MACD
exp1 = df_mlp['Close'].ewm(span=12, adjust=False).mean()
exp2 = df_mlp['Close'].ewm(span=26, adjust=False).mean()
df_mlp['MACD'] = exp1 - exp2
df_mlp['MACD_EMA'] = df_mlp['MACD'].ewm(span=9, adjust=False).mean()

# ATR
high_low = df_mlp['High'] - df_mlp['Low']
high_close = np.abs(df_mlp['High'] - df_mlp['Close'].shift())
low_close = np.abs(df_mlp['Low'] - df_mlp['Close'].shift())
ranges = pd.concat([high_low, high_close, low_close], axis=1)
true_range = np.max(ranges, axis=1)
df_mlp['ATR'] = true_range.rolling(window=14).mean()

# ADX
df_mlp['ADX'] = 50

# CCI
TP = (df_mlp['High'] + df_mlp['Low'] + df_mlp['Close']) / 3
SMA_TP = TP.rolling(window=20).mean()
MAD = TP.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
df_mlp['CCI'] = (TP - SMA_TP) / (0.015 * MAD)

# ROC
df_mlp['ROC'] = df_mlp['Close'].pct_change(periods=12) * 100

# RSI
delta = df_mlp['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df_mlp['RSI'] = 100 - (100 / (1 + rs))

# Williams %R
period_w = 14
highest_high = df_mlp['High'].rolling(window=period_w).max()
lowest_low = df_mlp['Low'].rolling(window=period_w).min()
df_mlp['William%R'] = -100 * ((highest_high - df_mlp['Close']) / (highest_high - lowest_low))

# Stochastic %K
lowest_low_14 = df_mlp['Low'].rolling(window=14).min()
highest_high_14 = df_mlp['High'].rolling(window=14).max()
df_mlp['SO%K'] = 100 * ((df_mlp['Close'] - lowest_low_14) / (highest_high_14 - lowest_low_14))

# STD5
df_mlp['STD5'] = df_mlp['Close'].rolling(window=5).std()

# Force Index
df_mlp['ForceIndex1'] = (df_mlp['Close'] - df_mlp['Close'].shift(1)) * df_mlp['Volume']
df_mlp['ForceIndex20'] = df_mlp['ForceIndex1'].rolling(window=20).mean()

# Features de fecha
df_mlp['Date_col'] = df_mlp.index
df_mlp['Day'] = df_mlp.index.day
df_mlp['DayofWeek'] = df_mlp.index.dayofweek
df_mlp['DayofYear'] = df_mlp.index.dayofyear
df_mlp['Week'] = df_mlp.index.isocalendar().week
df_mlp['Is_month_end'] = df_mlp.index.is_month_end.astype(int)
df_mlp['Is_month_start'] = df_mlp.index.is_month_start.astype(int)
df_mlp['Is_quarter_end'] = df_mlp.index.is_quarter_end.astype(int)
df_mlp['Is_quarter_start'] = df_mlp.index.is_quarter_start.astype(int)
df_mlp['Is_year_end'] = df_mlp.index.is_year_end.astype(int)
df_mlp['Is_year_start'] = df_mlp.index.is_year_start.astype(int)
df_mlp['Is_leap_year'] = (df_mlp.index.year % 4 == 0).astype(int)
df_mlp['Year'] = df_mlp.index.year
df_mlp['Month'] = df_mlp.index.month

# Índices de mercado
start_date = df_mlp.index[0].strftime('%Y-%m-%d')
end_date = df_mlp.index[-1].strftime('%Y-%m-%d')

print("✓ Descargando índices de mercado...")

# QQQ
try:
    qqq = yf.download('QQQ', start=start_date, end=end_date, progress=False)
    if not qqq.empty:
        df_mlp['QQQ_Close'] = qqq['Close']
        df_mlp['QQQ(t-1)'] = qqq['Close'].shift(1)
        df_mlp['QQQ(t-2)'] = qqq['Close'].shift(2)
        df_mlp['QQQ(t-5)'] = qqq['Close'].shift(5)
        df_mlp['QQQ_MA10'] = qqq['Close'].rolling(window=10).mean()
        df_mlp['QQQ_MA20'] = qqq['Close'].rolling(window=20).mean()
        df_mlp['QQQ_MA50'] = qqq['Close'].rolling(window=50).mean()
except:
    print("   ⚠️ No se pudo descargar QQQ, continuando...")

# S&P 500
try:
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    if not spy.empty:
        df_mlp['SnP_Close'] = spy['Close']
        df_mlp['SnP(t-1))'] = spy['Close'].shift(1)
        df_mlp['SnP(t-5)'] = spy['Close'].shift(5)
except:
    df_mlp['SnP_Close'] = 0
    df_mlp['SnP(t-1))'] = 0
    df_mlp['SnP(t-5)'] = 0

# DJIA
try:
    dia = yf.download('DIA', start=start_date, end=end_date, progress=False)
    if not dia.empty:
        df_mlp['DJIA_Close'] = dia['Close']
        df_mlp['DJIA(t-1))'] = dia['Close'].shift(1)
        df_mlp['DJIA(t-5)'] = dia['Close'].shift(5)
except:
    df_mlp['DJIA_Close'] = 0
    df_mlp['DJIA(t-1))'] = 0
    df_mlp['DJIA(t-5)'] = 0

# Target
df_mlp['Close_forcast'] = df_mlp['Close'].shift(-1)

# Limpiar
if 'MA20_temp' in df_mlp.columns:
    df_mlp = df_mlp.drop(columns=['MA20_temp'])

df_mlp = df_mlp.dropna()
df_mlp = df_mlp[COLUMNAS_MLP]
df_mlp = df_mlp.reset_index()
df_mlp = df_mlp.rename(columns={'index': 'Date'})
df_mlp['Date'] = pd.to_datetime(df_mlp['Date']).dt.date
df_mlp['Date_col'] = df_mlp['Date'].astype(str)

columnas_ordenadas = ['Date'] + [col for col in df_mlp.columns if col != 'Date']
df_mlp = df_mlp[columnas_ordenadas]

# Guardar CSV
nombre_archivo = f"{TICKER}MLP.csv"
df_mlp.to_csv(nombre_archivo, index=False)

print(f"\n✓ Datos guardados en: {nombre_archivo}")
print(f"   • Total registros: {len(df_mlp)}")
print(f"   • Fechas: {df_mlp['Date'].iloc[0]} a {df_mlp['Date'].iloc[-1]}")

# ============================================
# 3. CARGAR DATOS Y CONFIGURAR MODELO
# ============================================
print("\n" + "=" * 70)
print("3. CARGANDO DATOS Y ENTRENANDO MODELOS...")
print("=" * 70)

df_accion = pd.read_csv(nombre_archivo)
df_accion['Date'] = pd.to_datetime(df_accion['Date'])
df_accion.set_index('Date', inplace=True)

print(f"✓ Datos cargados: {df_accion.index[0].date()} a {df_accion.index[-1].date()} ({len(df_accion)} días)")

# Features
FEATURES_MODELO = ['Close', 'Volume', 'RSI', 'MA20', 'MA200', 
    'MACD', 'Upper_Band', 'Lower_Band', 'ATR',
    'S_Close(t-1)', 'EMA20', 'William%R', 'ROC',
    'QQQ_Close', 'SnP_Close']

# ============================================
# 4. GRÁFICA 1: PRECIO HISTÓRICO
# ============================================
print("\n" + "=" * 70)
print(f"GRÁFICA 1: {TICKER} - Precio Histórico")
print("=" * 70)

fig_historico = go.Figure()
fig_historico.add_trace(go.Scatter(
    x=df_accion.index, y=df_accion['Close'],
    mode='lines', name='Precio de Cierre',
    line=dict(color='#1f77b4', width=2),
    fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.1)',
    hovertemplate='<b>📅 Fecha:</b> %{x}<br><b>💰 Precio:</b> $%{y:.2f}<extra></extra>'
))

ma50 = df_accion['Close'].rolling(50).mean()
ma200 = df_accion['Close'].rolling(200).mean()
fig_historico.add_trace(go.Scatter(x=df_accion.index, y=ma50, mode='lines', name='MA50',
                                   line=dict(color='purple', width=1.5, dash='dot')))
fig_historico.add_trace(go.Scatter(x=df_accion.index, y=ma200, mode='lines', name='MA200',
                                   line=dict(color='orange', width=1.5, dash='dash')))

precio_max, precio_min = df_accion['Close'].max(), df_accion['Close'].min()
fecha_max, fecha_min = df_accion['Close'].idxmax(), df_accion['Close'].idxmin()

fig_historico.add_trace(go.Scatter(x=[fecha_max], y=[precio_max], mode='markers+text',
                                   name=f'Máximo: ${precio_max:.2f}',
                                   marker=dict(color='green', size=12, symbol='star'),
                                   text=[f' ${precio_max:.2f}'], textposition='top center'))
fig_historico.add_trace(go.Scatter(x=[fecha_min], y=[precio_min], mode='markers+text',
                                   name=f'Mínimo: ${precio_min:.2f}',
                                   marker=dict(color='red', size=10),
                                   text=[f' ${precio_min:.2f}'], textposition='bottom center'))

fig_historico.update_layout(
    title=f'<b>{TICKER} - Precio de Cierre Histórico</b><br><sub>{df_accion.index[0].date()} a {df_accion.index[-1].date()}</sub>',
    xaxis_title='Fecha', yaxis_title='Precio (USD)', height=600, width=1300,
    template='plotly_white', hovermode='x unified',
    xaxis=dict(rangeslider=dict(visible=True), rangeselector=dict(
        buttons=[dict(count=1, label="1y", step="year", stepmode="backward"),
                 dict(step="all", label="Todo")]))
)
fig_historico.show()

# ============================================
# 5. CREAR TARGETS
# ============================================
print("\n" + "=" * 70)
print("4. CREANDO TARGETS...")
print("=" * 70)

def obtener_precio_futuro(df, fecha, dias, max_offset=10):
    fecha_obj = fecha + pd.Timedelta(days=dias)
    if fecha_obj > df.index[-1]:
        return np.nan
    if fecha_obj in df.index:
        return df.loc[fecha_obj, 'Close']
    for offset in range(1, max_offset + 1):
        for delta in [offset, -offset]:
            fecha_cercana = fecha_obj + pd.Timedelta(days=delta)
            if fecha_cercana in df.index and fecha_cercana <= df.index[-1]:
                return df.loc[fecha_cercana, 'Close']
    return np.nan

print(f"Creando targets a {DIAS_HORIZONTE} días...")
df_accion['Precio_Futuro'] = None
df_accion['Señal_Subida'] = None

for i, fecha in enumerate(df_accion.index):
    precio_futuro = obtener_precio_futuro(df_accion, fecha, DIAS_HORIZONTE)
    df_accion.loc[fecha, 'Precio_Futuro'] = precio_futuro
    if pd.notna(precio_futuro):
        cambio = (precio_futuro - df_accion.loc[fecha, 'Close']) / df_accion.loc[fecha, 'Close']
        df_accion.loc[fecha, 'Señal_Subida'] = 1 if cambio > UMBRAL_SUBIDA else 0
    else:
        df_accion.loc[fecha, 'Señal_Subida'] = np.nan

print("✓ Targets creados:")
print(f"  Precio_Futuro: {df_accion['Precio_Futuro'].notna().sum()} fechas")
distribucion = df_accion['Señal_Subida'].value_counts()
print(f"  Señal_Subida: {distribucion.get(0,0)} no suben, {distribucion.get(1,0)} suben")

# ============================================
# 6. PREPARAR DATOS
# ============================================
print("\n" + "=" * 70)
print("5. PREPARANDO DATOS...")
print("=" * 70)

features = [f for f in FEATURES_MODELO if f in df_accion.columns]
X = df_accion[features].copy()
y_precio = df_accion['Precio_Futuro'].copy()
y_senal = df_accion['Señal_Subida'].copy()

datos_limpios = pd.concat([X, y_precio, y_senal], axis=1).dropna()
X_clean = datos_limpios[features]
y_precio_clean = datos_limpios['Precio_Futuro']
y_senal_clean = datos_limpios['Señal_Subida'].astype(int)

print(f"✓ Datos limpios: {len(datos_limpios)} filas")
print(f"  Rango: {datos_limpios.index[0].date()} a {datos_limpios.index[-1].date()}")

# ============================================
# 7. DIVIDIR DATOS
# ============================================
print("\n" + "=" * 70)
print("6. DIVIDIENDO DATOS")
print("=" * 70)

fechas = datos_limpios.index
dias_prueba = min(DIAS_DE_PRUEBA, len(fechas))
fechas_prueba = fechas[-dias_prueba:]
fechas_entrenamiento = fechas[:-dias_prueba]

X_train = X_clean.loc[fechas_entrenamiento]
X_test = X_clean.loc[fechas_prueba]
y_precio_train = y_precio_clean.loc[fechas_entrenamiento]
y_precio_test = y_precio_clean.loc[fechas_prueba]
y_senal_train = y_senal_clean.loc[fechas_entrenamiento]
y_senal_test = y_senal_clean.loc[fechas_prueba]

print("✓ División:")
print(f"  Entrenamiento: {len(X_train)} días ({fechas_entrenamiento[0].date()} a {fechas_entrenamiento[-1].date()})")
print(f"  Prueba: {len(X_test)} días ({fechas_prueba[0].date()} a {fechas_prueba[-1].date()})")

# ============================================
# 8. ESCALAR Y ENTRENAR MODELOS
# ============================================
print("\n" + "=" * 70)
print("7. ENTRENANDO MODELOS")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo de precio
modelo_precio = LinearRegression()
modelo_precio.fit(X_train_scaled, y_precio_train)

# Modelo de señal (con random_state fijo)
modelo_senal = LogisticRegression(max_iter=1000, random_state=SEED)
modelo_senal.fit(X_train_scaled, y_senal_train)

print("✓ Modelos entrenados correctamente")

# ============================================
# 9. PREDICCIONES Y MÉTRICAS
# ============================================
print("\n" + "=" * 70)
print("8. MÉTRICAS")
print("=" * 70)

y_precio_pred = modelo_precio.predict(X_test_scaled)
y_senal_pred = modelo_senal.predict(X_test_scaled)
y_senal_proba = modelo_senal.predict_proba(X_test_scaled)[:, 1]

# Métricas de precio
mae = mean_absolute_error(y_precio_test, y_precio_pred)
rmse = np.sqrt(mean_squared_error(y_precio_test, y_precio_pred))
r2 = r2_score(y_precio_test, y_precio_pred)

# Métricas de señal
accuracy_senal = accuracy_score(y_senal_test, y_senal_pred)
tp = ((y_senal_pred == 1) & (y_senal_test == 1)).sum()
fp = ((y_senal_pred == 1) & (y_senal_test == 0)).sum()
tn = ((y_senal_pred == 0) & (y_senal_test == 0)).sum()
fn = ((y_senal_pred == 0) & (y_senal_test == 1)).sum()

print("\n📊 PRECIO (Regresión Lineal):")
print(f"  MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R²: {r2:.4f}")
print("\n📊 SEÑAL (Regresión Logística):")
print(f"  Accuracy: {accuracy_senal:.2%}")
print(f"  Matriz: [[{tn} {fp}] [{fn} {tp}]]")
if (tp+fp) > 0:
    print(f"  Precision: {tp/(tp+fp):.2%}")
if (tp+fn) > 0:
    print(f"  Recall: {tp/(tp+fn):.2%}")

# ============================================
# GRÁFICA 2: VALIDACIÓN DE PRECIO
# ============================================
print("\n" + "=" * 70)
print(f"GRÁFICA 2: {TICKER} - Validación de Precio")
print("=" * 70)

fechas_predichas = X_test.index + pd.Timedelta(days=DIAS_HORIZONTE)

fig_precio = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
fig_precio.add_trace(go.Scatter(x=fechas_predichas, y=y_precio_test, mode='lines+markers',
                                name='Precio Real', line=dict(color='green', width=2.5)), row=1, col=1)
fig_precio.add_trace(go.Scatter(x=fechas_predichas, y=y_precio_pred, mode='lines+markers',
                                name='Predicción', line=dict(color='red', width=2, dash='dash')), row=1, col=1)

errors_pct = ((y_precio_pred - y_precio_test) / y_precio_test) * 100
colors = ['green' if e < 0 else 'red' for e in errors_pct]
fig_precio.add_trace(go.Bar(x=fechas_predichas, y=errors_pct, name='Error %',
                            marker_color=colors, opacity=0.6), row=2, col=1)
fig_precio.add_hline(y=0, line_dash="solid", line_color="black", row=2, col=1)

fig_precio.update_layout(title=f'{TICKER} - Validación a {DIAS_HORIZONTE} días', height=600,
                         template='plotly_white', hovermode='x unified')
fig_precio.update_xaxes(title_text="Fecha", row=2, col=1)
fig_precio.update_yaxes(title_text="Precio ($)", row=1, col=1)
fig_precio.update_yaxes(title_text="Error (%)", row=2, col=1)
fig_precio.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
                          text=f'MAE: ${mae:.2f} | RMSE: ${rmse:.2f} | R²: {r2:.4f}',
                          showarrow=False, bgcolor="white", bordercolor="gray", borderwidth=1)
fig_precio.show()

# ============================================
# GRÁFICA 3: CLASIFICACIÓN
# ============================================
print("\n" + "=" * 70)
print(f"GRÁFICA 3: {TICKER} - Análisis de Clasificación")
print("=" * 70)

fig1, axes1 = plt.subplots(2, 2, figsize=(15, 10))

ax1 = axes1[0, 0]
ax1.plot(X_test.index, X_test['Close'], color='blue', alpha=0.7, linewidth=1.5)

correct_mask = (y_senal_pred == y_senal_test.values)
incorrect_mask = (y_senal_pred != y_senal_test.values)

correct_dates = X_test.index[correct_mask]
correct_prices = X_test.loc[correct_dates, 'Close']
ax1.scatter(correct_dates, correct_prices, color='green', s=40,
           label=f'Correctos ({sum(correct_mask)})', alpha=0.7)

incorrect_dates = X_test.index[incorrect_mask]
incorrect_prices = X_test.loc[incorrect_dates, 'Close']
ax1.scatter(incorrect_dates, incorrect_prices, color='red', s=60,
           label=f'Incorrectos ({sum(incorrect_mask)})', alpha=0.7)

ax1.set_title(f'{TICKER} - Clasificación (Accuracy: {accuracy_senal:.1%})')
ax1.set_ylabel('Precio ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

ax2 = axes1[0, 1]
ax2.plot(X_test.index, y_senal_proba, 'o-', color='purple', alpha=0.7, markersize=3, label='Probabilidad')
ax2.scatter(X_test.index, y_senal_test.values, 
           color=['red' if y==0 else 'green' for y in y_senal_test.values],
           s=40, alpha=0.6, label='Real (0/1)')
ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, label='Umbral 0.5')
ax2.set_title('Probabilidades de Predicción')
ax2.set_ylabel('Probabilidad')
ax2.set_ylim(-0.1, 1.1)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

ax3 = axes1[1, 0]
conf_matrix = np.array([[tn, fp], [fn, tp]])
ax3.imshow(conf_matrix, cmap='Blues', aspect='auto')
for i in range(2):
    for j in range(2):
        ax3.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', fontsize=14, fontweight='bold')
ax3.set_title('Matriz de Confusión')
ax3.set_xlabel('Predicho')
ax3.set_ylabel('Real')
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['0 (No sube)', '1 (Sube)'])
ax3.set_yticklabels(['0 (No sube)', '1 (Sube)'])

ax4 = axes1[1, 1]
ax4.hist(y_senal_proba, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
ax4.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Umbral 0.5')
ax4.set_title('Distribución de Probabilidades')
ax4.set_xlabel('Probabilidad')
ax4.set_ylabel('Frecuencia')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle(f'{TICKER} - Resultados de Clasificación Binaria', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)  

# ============================================
# GRÁFICA 4: RESUMEN ESTADÍSTICO
# ============================================
print("\n" + "=" * 70)
print(f"GRÁFICA 4: {TICKER} - Resumen Estadístico")
print("=" * 70)

errors_abs = y_precio_pred - y_precio_test

fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))

axes4[0, 0].hist(errors_abs, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
axes4[0, 0].axvline(x=errors_abs.mean(), color='red', linestyle='--', label=f'Media: ${errors_abs.mean():.2f}')
axes4[0, 0].axvline(x=0, color='black', linestyle='-', alpha=0.5)
axes4[0, 0].set_title('Distribución de Errores')
axes4[0, 0].set_xlabel('Error ($)')
axes4[0, 0].set_ylabel('Frecuencia')
axes4[0, 0].legend()
axes4[0, 0].grid(True, alpha=0.3)

axes4[0, 1].scatter(y_precio_test, y_precio_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
max_val = max(y_precio_test.max(), y_precio_pred.max())
min_val = min(y_precio_test.min(), y_precio_pred.min())
axes4[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Predicción Perfecta')
axes4[0, 1].set_title('Real vs Predicho')
axes4[0, 1].set_xlabel('Real ($)')
axes4[0, 1].set_ylabel('Predicho ($)')
axes4[0, 1].legend()
axes4[0, 1].grid(True, alpha=0.3)

metrics_data = {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MAPE': np.mean(np.abs(errors_pct))}
bars_metrics = axes4[1, 0].bar(range(len(metrics_data)), list(metrics_data.values()), 
                              color=['blue', 'green', 'orange', 'red'], alpha=0.7)
axes4[1, 0].set_xticks(range(len(metrics_data)))
axes4[1, 0].set_xticklabels(list(metrics_data.keys()), rotation=45)
axes4[1, 0].set_title('Métricas')
axes4[1, 0].set_ylabel('Valor')
for bar, value in zip(bars_metrics, metrics_data.values()):
    axes4[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{value:.3f}', ha='center', va='bottom', fontsize=9)
axes4[1, 0].grid(True, alpha=0.3, axis='y')

axes4[1, 1].axis('off')

plt.suptitle(f'{TICKER} - Análisis Estadístico ({DIAS_HORIZONTE} días)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)  

# ============================================
# 10. PREDICCIÓN FUTURA
# ============================================
print("\n" + "=" * 70)
print(f"PREDICCIÓN FUTURA PARA {TICKER}")
print("=" * 70)

ultima_fecha = df_accion.index[-1]
precio_actual = df_accion['Close'].iloc[-1]

print(f"📊 Último dato real: {ultima_fecha.date()} - Precio: ${precio_actual:.2f}")
print(f"🔮 Predicción a {DIAS_HORIZONTE} días...")

datos_base = df_accion[features].iloc[-DIAS_HORIZONTE:].copy()
X_base_scaled = scaler.transform(datos_base)

precios_futuros = modelo_precio.predict(X_base_scaled)
senal_futura = modelo_senal.predict(X_base_scaled)
senal_futura_proba = modelo_senal.predict_proba(X_base_scaled)[:, 1]

df_futuro = pd.DataFrame({
    'Fecha': [datos_base.index[i] + timedelta(days=DIAS_HORIZONTE) for i in range(len(datos_base))],
    'Precio': precios_futuros,
    'Señal': senal_futura,
    'Probabilidad': senal_futura_proba
})
df_futuro = df_futuro[df_futuro['Fecha'] > ultima_fecha].head(DIAS_A_MOSTRAR)

fechas_mostrar = pd.to_datetime(df_futuro['Fecha'].values)
precios_mostrar = df_futuro['Precio'].values
senal_mostrar = df_futuro['Señal'].values
prob_mostrar = df_futuro['Probabilidad'].values

cambio_final = ((precios_mostrar[-1] - precio_actual) / precio_actual) * 100

print("✅ Predicción generada:")
print(f"   • Período: {fechas_mostrar[0].date()} a {fechas_mostrar[-1].date()}")
print(f"   • Precio final esperado: ${precios_mostrar[-1]:.2f}")
print(f"   • Cambio esperado: {cambio_final:+.2f}%")
print(f"   • Señal: {'🟢 COMPRAR' if senal_mostrar[-1] == 1 else '⚪ MANTENER'}")
print(f"   • Confianza: {prob_mostrar[-1]*100:.1f}%")

# ============================================
# GRÁFICA 5: PREDICCIÓN FUTURA
# ============================================
print("\n" + "=" * 70)
print(f"GRÁFICA 5: {TICKER} - Predicción Futura")
print("=" * 70)

fig_fut = go.Figure()
fig_fut.add_trace(go.Scatter(x=df_accion.index[-100:], y=df_accion['Close'].iloc[-100:],
                             mode='lines', name='📊 Datos Históricos', line=dict(color='blue', width=2)))
fig_fut.add_trace(go.Scatter(x=fechas_mostrar, y=precios_mostrar, mode='lines+markers',
                             name=f'🔮 Predicción {DIAS_HORIZONTE} días',
                             line=dict(color='red', width=2), marker=dict(size=6)))
fig_fut.add_hline(y=precio_actual, line_dash="dash", line_color="gray",
                  annotation_text=f"💰 Precio Actual: ${precio_actual:.2f}")
fig_fut.add_trace(go.Scatter(x=[ultima_fecha], y=[precio_actual], mode='markers',
                             name='📌 Fin Datos Reales', marker=dict(color='orange', size=10)))

fig_fut.update_layout(title=f'{TICKER} - Predicción Futura a {DIAS_HORIZONTE} días',
                      xaxis_title='Fecha', yaxis_title='Precio (USD)', height=500,
                      template='plotly_white', hovermode='x unified')
fig_fut.show()

# ============================================
# TABLA DE PREDICCIONES
# ============================================
print("\n" + "=" * 70)
print(f"📋 TABLA DE PREDICCIONES - PRÓXIMOS {DIAS_HORIZONTE} DÍAS")
print("=" * 70)
print(f"{'Fecha':<15} {'Precio':<12} {'Cambio %':<12} {'Señal':<15} {'Confianza':<12}")
print("-" * 70)

for i in range(len(fechas_mostrar)):
    cambio = ((precios_mostrar[i] - precio_actual) / precio_actual) * 100
    senal_texto = "🟢 COMPRAR" if senal_mostrar[i] == 1 else "⚪ MANTENER"
    print(f"{fechas_mostrar[i].date():<15} ${precios_mostrar[i]:<11.2f} {cambio:+11.2f}% {senal_texto:<15} {prob_mostrar[i]*100:>5.1f}%")

# ============================================
# RESUMEN EJECUTIVO
# ============================================
print("\n" + "=" * 70)
print("RESUMEN EJECUTIVO")
print("=" * 70)

print("\n📅 INFORMACIÓN GENERAL:")
print(f"  • Empresa: {TICKER}")
print(f"  • Período de datos: {df_accion.index[0].date()} a {df_accion.index[-1].date()}")
print(f"  • Precio actual: ${precio_actual:.2f}")

print("\n📊 MODELO DE PRECIO (Regresión Lineal):")
print(f"  • Error promedio (MAE): ${mae:.2f}")
print(f"  • Raíz del error cuadrático (RMSE): ${rmse:.2f}")
print(f"  • Precisión del modelo (R²): {r2:.4f}")

print("\n📊 MODELO DE SEÑAL (Regresión Logística):")
print(f"  • Aciertos totales: {accuracy_senal:.2%}")
if (tp+fp) > 0:
    print(f"  • Cuando dice COMPRAR, acierta: {tp/(tp+fp):.2%}")
if (tp+fn) > 0:
    print(f"  • Detecta el {tp/(tp+fn):.2%} de las subidas reales")
print(f"  • Matriz de confusión: [[{tn} {fp}] [{fn} {tp}]]")

print(f"\n🔮 PREDICCIÓN FUTURA ({DIAS_HORIZONTE} días):")
print(f"  • Precio esperado: ${precios_mostrar[-1]:.2f}")
print(f"  • Cambio esperado: {cambio_final:+.2f}%")
print(f"  • Señal: {'🟢 COMPRAR' if senal_mostrar[-1] == 1 else '⚪ MANTENER'}")
print(f"  • Confianza: {prob_mostrar[-1]*100:.1f}%")

print("\n" + "=" * 70)
print("⚠️ DISCLAIMER: Esta herramienta es solo para fines educativos.")
print("   No constituye asesoramiento financiero profesional.")
print("   Las decisiones de inversión deben tomarse con criterio propio.")
print("=" * 70)

plt.ioff()
plt.show(block=True)  