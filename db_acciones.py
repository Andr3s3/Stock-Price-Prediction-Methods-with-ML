#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
db_acciones.py
==============
Almacenamiento de precios e indicadores técnicos en MySQL/MariaDB (servidor LAMP).

- Crea la base de datos `prediccion_acciones` y sus tablas si no existen.
- Descarga de Yahoo Finance SOLO los días que faltan (actualización incremental).
- Detecta cuando Yahoo re-ajusta el histórico (dividendos / splits) y en ese caso
  vuelve a descargar todo el histórico del ticker para no mezclar precios.
- Calcula el mismo dataset de 64 columnas de DData.ipynb para que TODOS los
  modelos (lineal, logística, RF, XGBoost, LSTM, MLP) lean los mismos datos.
- Tablas separadas por bolsa (se detecta por el sufijo del ticker):
    EE. UU. -> precios_usa, dataset_usa   (AAPL, MSFT, QQQ, SPY, DIA)
    México  -> precios_mx,  dataset_mx    (WALMEX.MX, GMEXICOB.MX, ^MXX)
  y una tabla `empresas` con el mercado y la moneda (USD / MXN) de cada ticker.

Uso desde terminal (por ejemplo, una vez al día con cron):
    python3 db_acciones.py AAPL WALMEX.MX       # actualiza esos tickers
    python3 db_acciones.py --todas            # actualiza todos los que ya están en la BD
    python3 db_acciones.py --sql > crear_bd.sql   # imprime el SQL de creación

Uso desde otro script:
    from db_acciones import actualizar_ticker, cargar_dataset_mlp
    actualizar_ticker('AAPL')
    df = cargar_dataset_mlp('AAPL', periodo='5y')

Contraseña: se lee de la variable de entorno MYSQL_PASSWORD; si no existe,
se pide por teclado (no se guarda en el código).
"""
import os
import re
import sys
import argparse
import getpass
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN DE CONEXIÓN
# ============================================
DB_NAME = os.environ.get('MYSQL_DB', 'prediccion_acciones')

DB_CONFIG = dict(
    host=os.environ.get('MYSQL_HOST', 'localhost'),
    user=os.environ.get('MYSQL_USER', 'admin_web'),
    unix_socket=os.environ.get('MYSQL_SOCKET', '/var/run/mysqld/mysqld.sock'),
    autocommit=True,
    charset='utf8mb4',
)

# ============================================
# MERCADOS: cada bolsa tiene sus propias tablas
#   USA -> precios_usa, dataset_usa   (AAPL, MSFT, QQQ, SPY, DIA...)
#   MX  -> precios_mx,  dataset_mx    (WALMEX.MX, GMEXICOB.MX, ^MXX...)
# El mercado se detecta por el sufijo del ticker en Yahoo Finance (.MX).
# ============================================
MERCADOS = {
    'USA': {'moneda': 'USD', 'indices': ['QQQ', 'SPY', 'DIA']},
    'MX':  {'moneda': 'MXN', 'indices': ['^MXX']},        # ^MXX = S&P/BMV IPC
}

# Índices de EE. UU. que usa el dataset como features (para ambos mercados,
# así todas las acciones tienen las mismas 63 columnas)
INDICES = ['QQQ', 'SPY', 'DIA']


def mercado_de(ticker):
    t = ticker.upper().strip()
    return 'MX' if t.endswith('.MX') or t == '^MXX' else 'USA'


def tabla_precios(ticker):
    return f"precios_{mercado_de(ticker).lower()}"


def tabla_dataset(ticker):
    return f"dataset_{mercado_de(ticker).lower()}"

# Tolerancia para detectar que Yahoo re-ajustó precios pasados (0.01 %)
TOLERANCIA_AJUSTE = 1e-4
# Días naturales que se vuelven a descargar para comparar con lo guardado
DIAS_SOLAPE = 14

# ============================================
# COLUMNAS DEL DATASET (idénticas a DData.ipynb)
# ============================================
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

COLUMNAS_ENTERAS = [
    'Day', 'DayofWeek', 'DayofYear', 'Week',
    'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start',
    'Is_year_end', 'Is_year_start', 'Is_leap_year', 'Year', 'Month', 'ADX'
]


def _nombre_sql(col):
    """'S_Close(t-1)' -> 'S_Close_t_1', 'William%R' -> 'William_R'.
    Se evitan '%' y paréntesis en MySQL ('%' rompe los parámetros de pymysql)."""
    return re.sub(r'[^0-9A-Za-z_]+', '_', col).strip('_')


# Mapeo nombre original <-> nombre en MySQL
COL_A_SQL = {c: _nombre_sql(c) for c in COLUMNAS_MLP}
SQL_A_COL = {v: k for k, v in COL_A_SQL.items()}
assert len(SQL_A_COL) == len(COL_A_SQL), "Nombres SQL duplicados"


# ============================================
# CONEXIÓN
# ============================================
_PASSWORD_CACHE = None


def _password():
    global _PASSWORD_CACHE
    if _PASSWORD_CACHE is None:
        _PASSWORD_CACHE = os.environ.get('MYSQL_PASSWORD')
        if _PASSWORD_CACHE is None:
            _PASSWORD_CACHE = getpass.getpass(
                f"Contraseña MySQL para '{DB_CONFIG['user']}': ")
    return _PASSWORD_CACHE


def conectar():
    """Conecta al servidor, crea la BD si no existe y la selecciona."""
    import pymysql
    conn = pymysql.connect(password=_password(), **DB_CONFIG)
    crear_base_datos(conn)
    return conn


# ============================================
# ESQUEMA
# ============================================
def _ddl_dataset(tabla):
    defs = []
    for c in COLUMNAS_MLP:
        if c == 'Date_col':
            tipo = 'CHAR(10)'
        elif c == 'Volume':
            tipo = 'BIGINT'
        elif c in COLUMNAS_ENTERAS:
            tipo = 'SMALLINT'
        else:
            tipo = 'DOUBLE'
        defs.append(f"  `{COL_A_SQL[c]}` {tipo} NULL")
    columnas = ",\n".join(defs)
    return f"""CREATE TABLE IF NOT EXISTS {tabla} (
  ticker VARCHAR(16) NOT NULL,
  fecha DATE NOT NULL,
{columnas},
  PRIMARY KEY (ticker, fecha),
  CONSTRAINT fk_{tabla}_empresa FOREIGN KEY (ticker)
      REFERENCES empresas(ticker) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""


def _ddl_precios(tabla):
    return f"""CREATE TABLE IF NOT EXISTS {tabla} (
  ticker VARCHAR(16) NOT NULL,
  fecha DATE NOT NULL,
  `Open` DOUBLE NULL,
  `High` DOUBLE NULL,
  `Low` DOUBLE NULL,
  `Close` DOUBLE NOT NULL,
  `Volume` BIGINT NULL,
  PRIMARY KEY (ticker, fecha),
  CONSTRAINT fk_{tabla}_empresa FOREIGN KEY (ticker)
      REFERENCES empresas(ticker) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""


def sentencias_ddl():
    sentencias = [
        f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci",
        f"USE `{DB_NAME}`",
        """CREATE TABLE IF NOT EXISTS empresas (
  ticker VARCHAR(16) NOT NULL PRIMARY KEY,
  mercado ENUM('USA','MX') NOT NULL,
  moneda CHAR(3) NOT NULL,
  tipo ENUM('accion','indice') NOT NULL DEFAULT 'accion',
  primera_fecha DATE NULL,
  ultima_fecha DATE NULL,
  ultima_actualizacion DATETIME(6) NULL,
  ultima_recarga DATETIME(6) NULL,
  dataset_construido DATETIME(6) NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    ]
    for m in MERCADOS:
        sentencias.append(_ddl_precios(f"precios_{m.lower()}"))
        sentencias.append(_ddl_dataset(f"dataset_{m.lower()}"))
    return sentencias


def crear_base_datos(conn):
    with conn.cursor() as cur:
        for sql in sentencias_ddl():
            cur.execute(sql)


# ============================================
# UTILIDADES
# ============================================
def _a_python(v):
    """Convierte valores numpy/pandas a tipos que pymysql entiende (NaN -> NULL)."""
    if v is None:
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.date()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else float(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def filtrar_periodo(df, periodo):
    """Filtra un DataFrame (índice de fechas) con la misma notación de yfinance:
    5d, 3wk, 3mo, 5y, ytd, max."""
    periodo = (periodo or 'max').strip().lower()
    if periodo == 'max' or df.empty:
        return df
    fin = df.index[-1]
    if periodo == 'ytd':
        return df[df.index >= pd.Timestamp(year=fin.year, month=1, day=1)]
    m = re.fullmatch(r'(\d+)\s*(d|wk|mo|y)', periodo)
    if not m:
        raise ValueError(f"Periodo no válido: '{periodo}' (usa 5d, 3wk, 3mo, 5y, ytd o max)")
    n, u = int(m.group(1)), m.group(2)
    offset = {'d': pd.DateOffset(days=n), 'wk': pd.DateOffset(weeks=n),
              'mo': pd.DateOffset(months=n), 'y': pd.DateOffset(years=n)}[u]
    return df[df.index > fin - offset]


# ============================================
# DESCARGA DE YAHOO FINANCE
# ============================================
def _descargar(ticker, **kwargs):
    """Descarga OHLCV ajustado (auto_adjust=True, igual que el script original)."""
    import yfinance as yf
    df = yf.download(ticker, auto_adjust=True, progress=False, **kwargs)
    if df is None or df.empty:
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    if isinstance(df.columns, pd.MultiIndex):          # yfinance >= 0.2.48
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Close'])
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df


# ============================================
# TABLAS precios_usa / precios_mx
# ============================================
def _registrar_empresa(conn, ticker, tipo):
    m = mercado_de(ticker)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO empresas (ticker, mercado, moneda, tipo) VALUES (%s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE ticker = ticker",
            (ticker, m, MERCADOS[m]['moneda'], tipo))


def _insertar_precios(cur, ticker, df):
    filas = [
        (ticker, f.date(), _a_python(r.Open), _a_python(r.High), _a_python(r.Low),
         float(r.Close), None if pd.isna(r.Volume) else int(round(float(r.Volume))))
        for f, r in zip(df.index, df.itertuples(index=False))
    ]
    if not filas:
        return 0
    cur.executemany(
        f"INSERT INTO {tabla_precios(ticker)} (ticker, fecha, `Open`, `High`, `Low`, `Close`, `Volume`) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s) "
        "ON DUPLICATE KEY UPDATE `Open`=VALUES(`Open`), `High`=VALUES(`High`), "
        "`Low`=VALUES(`Low`), `Close`=VALUES(`Close`), `Volume`=VALUES(`Volume`)",
        filas)
    return len(filas)


def _recarga_completa(conn, ticker):
    df = _descargar(ticker, period='max')
    if df.empty:
        raise RuntimeError(f"Yahoo Finance no devolvió datos para {ticker}")
    conn.begin()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {tabla_dataset(ticker)} WHERE ticker=%s", (ticker,))
            cur.execute(f"DELETE FROM {tabla_precios(ticker)} WHERE ticker=%s", (ticker,))
            n = _insertar_precios(cur, ticker, df)
            cur.execute("UPDATE empresas SET ultima_recarga=%s, dataset_construido=NULL "
                        "WHERE ticker=%s", (datetime.now(), ticker))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return n


def actualizar_precios(conn, ticker, tipo='accion'):
    """Añade a precios_<mercado> solo los días nuevos de `ticker`.
    Devuelve (filas_insertadas_o_actualizadas, hubo_recarga_completa)."""
    ticker = ticker.upper().strip()
    _registrar_empresa(conn, ticker, tipo)

    with conn.cursor() as cur:
        cur.execute(f"SELECT MAX(fecha) FROM {tabla_precios(ticker)} WHERE ticker=%s", (ticker,))
        ultima = cur.fetchone()[0]

    if ultima is None:
        print(f"   {ticker}: primera descarga (histórico completo)...")
        n = _recarga_completa(conn, ticker)
        recarga = True
    else:
        inicio = ultima - timedelta(days=DIAS_SOLAPE)
        nuevo = _descargar(ticker, start=inicio.strftime('%Y-%m-%d'))

        # ¿Cambió el histórico ya guardado? (dividendo/split re-ajusta precios pasados)
        # Se excluye el último día guardado: pudo guardarse con el mercado abierto.
        with conn.cursor() as cur:
            cur.execute(f"SELECT fecha, `Close` FROM {tabla_precios(ticker)} "
                        "WHERE ticker=%s AND fecha>=%s AND fecha<%s",
                        (ticker, inicio, ultima))
            viejo = pd.Series({pd.Timestamp(f): c for f, c in cur.fetchall()}, dtype=float)

        comun = viejo.index.intersection(nuevo.index)
        recarga = False
        if len(comun) > 0:
            dif = (nuevo.loc[comun, 'Close'] / viejo.loc[comun] - 1).abs()
            if (dif > TOLERANCIA_AJUSTE).any():
                print(f"   {ticker}: Yahoo re-ajustó el histórico "
                      f"(dif. máx {dif.max():.4%}); recargando completo...")
                n = _recarga_completa(conn, ticker)
                recarga = True

        if not recarga:
            nuevo = nuevo[nuevo.index >= pd.Timestamp(ultima)]
            with conn.cursor() as cur:
                n = _insertar_precios(cur, ticker, nuevo)
            nuevos_dias = int((nuevo.index > pd.Timestamp(ultima)).sum())
            print(f"   {ticker}: {nuevos_dias} día(s) nuevo(s) desde {ultima}")

    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE empresas e JOIN (SELECT MIN(fecha) a, MAX(fecha) b FROM {tabla_precios(ticker)} "
            "WHERE ticker=%s) p SET e.primera_fecha=p.a, e.ultima_fecha=p.b, "
            "e.ultima_actualizacion=%s WHERE e.ticker=%s",
            (ticker, datetime.now(), ticker))
    return n, recarga


def cargar_precios(conn, ticker, periodo='max'):
    """Devuelve OHLCV desde MySQL con el mismo formato que yf.download."""
    with conn.cursor() as cur:
        cur.execute("SELECT fecha, `Open`, `High`, `Low`, `Close`, `Volume` "
                    f"FROM {tabla_precios(ticker)} WHERE ticker=%s ORDER BY fecha", (ticker.upper(),))
        filas = cur.fetchall()
    df = pd.DataFrame(filas, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
    return filtrar_periodo(df, periodo)


# ============================================
# INDICADORES (mismo cálculo que DData.ipynb / Linear-Logistic-Regresion.py)
# ============================================
def construir_dataset_mlp(df, qqq, spy, dia):
    """df, qqq, spy, dia: DataFrames OHLCV con índice de fechas.
    Devuelve el dataset con índice 'Date' y las columnas COLUMNAS_MLP.
    A diferencia del original, la última fila se conserva aunque
    Close_forcast sea NaN (aún no se conoce el cierre de mañana)."""
    for nombre, d in [('QQQ', qqq), ('SPY', spy), ('DIA', dia)]:
        if d is None or d.empty:
            raise RuntimeError(f"No hay datos de {nombre} en la base de datos")

    df_mlp = pd.DataFrame(index=df.index, columns=COLUMNAS_MLP)

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df_mlp[col] = df[col].astype(float)

    # Bandas de Bollinger
    window = 20
    ma20_temp = df_mlp['Close'].rolling(window=window).mean()
    df_mlp['SD20'] = df_mlp['Close'].rolling(window=window).std()
    df_mlp['Upper_Band'] = ma20_temp + (df_mlp['SD20'] * 2)
    df_mlp['Lower_Band'] = ma20_temp - (df_mlp['SD20'] * 2)

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

    # ADX (placeholder del original: constante, no aporta información)
    df_mlp['ADX'] = 50

    # CCI
    TP = (df_mlp['High'] + df_mlp['Low'] + df_mlp['Close']) / 3
    SMA_TP = TP.rolling(window=20).mean()
    MAD = TP.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
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
    idx = df_mlp.index
    df_mlp['Date_col'] = idx.strftime('%Y-%m-%d')
    df_mlp['Day'] = idx.day
    df_mlp['DayofWeek'] = idx.dayofweek
    df_mlp['DayofYear'] = idx.dayofyear
    df_mlp['Week'] = idx.isocalendar().week.astype(int).values
    df_mlp['Is_month_end'] = idx.is_month_end.astype(int)
    df_mlp['Is_month_start'] = idx.is_month_start.astype(int)
    df_mlp['Is_quarter_end'] = idx.is_quarter_end.astype(int)
    df_mlp['Is_quarter_start'] = idx.is_quarter_start.astype(int)
    df_mlp['Is_year_end'] = idx.is_year_end.astype(int)
    df_mlp['Is_year_start'] = idx.is_year_start.astype(int)
    df_mlp['Is_leap_year'] = idx.is_leap_year.astype(int)
    df_mlp['Year'] = idx.year
    df_mlp['Month'] = idx.month

    # Índices de mercado. Se calculan sobre el calendario PROPIO del índice y
    # luego se alinean a las fechas de la acción usando el último valor conocido
    # (máx. 5 días). Para acciones de EE. UU. el resultado es idéntico al
    # original; para la BMV evita perder los días en que Wall Street cerró
    # (4 de julio, Thanksgiving...) y México no.
    def _alinear(serie):
        union = df_mlp.index.union(serie.index)
        return serie.reindex(union).ffill(limit=5).reindex(df_mlp.index)

    q = qqq['Close']
    df_mlp['QQQ_Close'] = _alinear(q)
    df_mlp['QQQ(t-1)'] = _alinear(q.shift(1))
    df_mlp['QQQ(t-2)'] = _alinear(q.shift(2))
    df_mlp['QQQ(t-5)'] = _alinear(q.shift(5))
    df_mlp['QQQ_MA10'] = _alinear(q.rolling(window=10).mean())
    df_mlp['QQQ_MA20'] = _alinear(q.rolling(window=20).mean())
    df_mlp['QQQ_MA50'] = _alinear(q.rolling(window=50).mean())

    s = spy['Close']
    df_mlp['SnP_Close'] = _alinear(s)
    df_mlp['SnP(t-1))'] = _alinear(s.shift(1))
    df_mlp['SnP(t-5)'] = _alinear(s.shift(5))

    d = dia['Close']
    df_mlp['DJIA_Close'] = _alinear(d)
    df_mlp['DJIA(t-1))'] = _alinear(d.shift(1))
    df_mlp['DJIA(t-5)'] = _alinear(d.shift(5))

    # Target del MLP (cierre del día siguiente)
    df_mlp['Close_forcast'] = df_mlp['Close'].shift(-1)

    # Limpiar: NaN de las ventanas (MA200, etc.) pero SIN perder el último día
    sin_target = [c for c in COLUMNAS_MLP if c != 'Close_forcast']
    df_mlp = df_mlp.dropna(subset=sin_target)
    df_mlp = df_mlp[COLUMNAS_MLP]

    for c in COLUMNAS_MLP:
        if c != 'Date_col':
            df_mlp[c] = pd.to_numeric(df_mlp[c])
    df_mlp.index.name = 'Date'
    return df_mlp


# ============================================
# TABLAS dataset_usa / dataset_mx
# ============================================
def _necesita_reconstruir(conn, ticker):
    with conn.cursor() as cur:
        cur.execute("SELECT dataset_construido FROM empresas WHERE ticker=%s", (ticker,))
        fila = cur.fetchone()
        construido = fila[0] if fila else None
        if construido is None:
            return True
        deps = [ticker] + INDICES
        marcadores = ", ".join(["%s"] * len(deps))
        cur.execute(f"SELECT MAX(ultima_recarga) FROM empresas WHERE ticker IN ({marcadores})",
                    deps)
        ultima_recarga = cur.fetchone()[0]
    return ultima_recarga is not None and ultima_recarga > construido


def actualizar_dataset_mlp(conn, ticker, reconstruir=False):
    """Recalcula indicadores sobre TODO el histórico guardado y escribe solo las
    filas nuevas (o todas si alguna serie fue recargada)."""
    ticker = ticker.upper().strip()
    reconstruir = reconstruir or _necesita_reconstruir(conn, ticker)

    df = cargar_precios(conn, ticker)
    qqq, spy, dia = (cargar_precios(conn, t) for t in INDICES)
    dataset = construir_dataset_mlp(df, qqq, spy, dia)

    with conn.cursor() as cur:
        if reconstruir:
            cur.execute(f"DELETE FROM {tabla_dataset(ticker)} WHERE ticker=%s", (ticker,))
            desde = None
        else:
            # La última fila guardada se reescribe: ahora ya se conoce su Close_forcast
            cur.execute(f"SELECT MAX(fecha) FROM {tabla_dataset(ticker)} WHERE ticker=%s", (ticker,))
            desde = cur.fetchone()[0]

    if desde is not None:
        dataset = dataset[dataset.index >= pd.Timestamp(desde)]

    cols_sql = [COL_A_SQL[c] for c in COLUMNAS_MLP]
    lista = ", ".join(f"`{c}`" for c in cols_sql)
    marcadores = ", ".join(["%s"] * (len(cols_sql) + 2))
    update = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in cols_sql)
    sql = (f"INSERT INTO {tabla_dataset(ticker)} (ticker, fecha, {lista}) VALUES ({marcadores}) "
           f"ON DUPLICATE KEY UPDATE {update}")

    filas = [(ticker, f.date(), *[_a_python(v) for v in fila])
             for f, fila in zip(dataset.index, dataset.itertuples(index=False, name=None))]
    with conn.cursor() as cur:
        for i in range(0, len(filas), 1000):
            cur.executemany(sql, filas[i:i + 1000])
        if reconstruir:
            cur.execute("UPDATE empresas SET dataset_construido=%s WHERE ticker=%s",
                        (datetime.now(), ticker))

    modo = "reconstruido" if reconstruir else "actualizado"
    print(f"   {tabla_dataset(ticker)} {modo} para {ticker}: {len(filas)} fila(s) escritas")
    return len(filas)


def cargar_dataset_mlp(ticker, periodo='max', conn=None):
    """Lee dataset_<mercado> como DataFrame con índice 'Date' y los nombres de
    columna originales (los mismos que tenía el CSV <TICKER>MLP.csv)."""
    cerrar = conn is None
    conn = conn or conectar()
    try:
        cols_sql = [COL_A_SQL[c] for c in COLUMNAS_MLP]
        lista = ", ".join(f"`{c}`" for c in cols_sql)
        with conn.cursor() as cur:
            cur.execute(f"SELECT fecha, {lista} FROM {tabla_dataset(ticker)} WHERE ticker=%s ORDER BY fecha",
                        (ticker.upper(),))
            filas = cur.fetchall()
    finally:
        if cerrar:
            conn.close()
    df = pd.DataFrame(filas, columns=['Date'] + COLUMNAS_MLP)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    for c in COLUMNAS_MLP:
        if c != 'Date_col':
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
    return filtrar_periodo(df, periodo)


# ============================================
# FUNCIÓN PRINCIPAL DE SINCRONIZACIÓN
# ============================================
def actualizar_ticker(ticker, conn=None):
    """Sincroniza índices + ticker + dataset de su mercado. Úsalo antes de cada modelo."""
    cerrar = conn is None
    conn = conn or conectar()
    try:
        ticker = ticker.upper().strip()
        m = mercado_de(ticker)
        print(f"Sincronizando {ticker} (mercado {m}) con MySQL ({DB_NAME})...")
        # Índices de EE. UU. (features del dataset) + índices del mercado propio
        indices = list(dict.fromkeys(INDICES + MERCADOS[m]['indices']))
        for idx in indices:
            actualizar_precios(conn, idx, tipo='indice')
        if ticker not in indices:
            actualizar_precios(conn, ticker, tipo='accion')
        actualizar_dataset_mlp(conn, ticker)
    finally:
        if cerrar:
            conn.close()


def tickers_en_bd(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT ticker FROM empresas WHERE tipo='accion' ORDER BY ticker")
        return [r[0] for r in cur.fetchall()]


def main():
    p = argparse.ArgumentParser(description="Actualiza precios e indicadores en MySQL")
    p.add_argument('tickers', nargs='*', help='Ej: AAPL MSFT NVDA')
    p.add_argument('--todas', action='store_true', help='Actualiza todas las empresas guardadas')
    p.add_argument('--sql', action='store_true', help='Solo imprime el SQL de creación')
    args = p.parse_args()

    if args.sql:
        print(";\n\n".join(sentencias_ddl()) + ";")
        return

    conn = conectar()
    try:
        tickers = [t.upper() for t in args.tickers]
        if args.todas:
            tickers = sorted(set(tickers) | set(tickers_en_bd(conn)))
        if not tickers:
            p.error("indica al menos un ticker o usa --todas")
        for t in tickers:
            try:
                actualizar_ticker(t, conn)
            except Exception as e:
                print(f"   ERROR {t}: {e}")
    finally:
        conn.close()


if __name__ == '__main__':
    main()
