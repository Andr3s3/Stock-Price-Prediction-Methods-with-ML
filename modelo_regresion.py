#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modelo_regresion.py
===================
Regresión Lineal (precio) + Regresión Logística (señales de subida y bajada).

Todo el flujo (datos, features, validación walk-forward, métricas, gráficas,
tabla y resumen) lo hace motor_prediccion.py. Aquí solo se define cómo se
entrena y cómo se predice, más una gráfica propia con los coeficientes.

Uso:
    python3 modelo_regresion.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from motor_prediccion import ejecutar_modelo, probabilidad, SEED


def entrenar(X, Y_dias, y_sube, y_baja, p_precio, p_senal):
    # Los modelos lineales necesitan las features en la misma escala
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # Regresión multi-salida: un rendimiento predicho por cada día del horizonte
    precio = LinearRegression().fit(Xs, Y_dias)

    def clasificador(y):
        if y.nunique() < 2:
            return None
        return LogisticRegression(max_iter=2000, class_weight='balanced',
                                  random_state=SEED).fit(Xs, y)

    return {'scaler': scaler, 'precio': precio,
            'sube': clasificador(y_sube), 'baja': clasificador(y_baja)}


def predecir(modelo, X):
    Xs = modelo['scaler'].transform(X)
    rend_dias = modelo['precio'].predict(Xs)
    return rend_dias, probabilidad(modelo['sube'], Xs), probabilidad(modelo['baja'], Xs)


def graficas_propias(modelo, ctx):
    """Coeficientes estandarizados: qué variables pesan más y en qué dirección."""
    print(f"GRÁFICA 6: {ctx.ticker} - Coeficientes de los modelos")
    coef = pd.DataFrame({
        'Precio (rendimiento final)': modelo['precio'].coef_[-1],
        'Prob. subida': modelo['sube'].coef_[0] if modelo['sube'] is not None else np.nan,
        'Prob. bajada': modelo['baja'].coef_[0] if modelo['baja'] is not None else np.nan,
    }, index=ctx.features)

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    for ax, col in zip(axes, coef.columns):
        valores = coef[col]
        ax.barh(coef.index, valores, color=['green' if v > 0 else 'red' for v in valores], alpha=0.7)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(col)
        ax.grid(True, alpha=0.3, axis='x')
    axes[0].invert_yaxis()
    plt.suptitle(f'{ctx.ticker} - Coeficientes estandarizados (verde: empuja hacia arriba, '
                 f'rojo: hacia abajo)', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.show(block=False); plt.pause(0.1)


if __name__ == '__main__':
    ejecutar_modelo(
        nombre="Regresión Lineal + Logística",
        entrenar=entrenar,
        predecir=predecir,
        nombre_precio="Regresión Lineal",
        nombre_senal="Regresión Logística",
        graficas_propias=graficas_propias,
    )
