# -*- coding: utf-8 -*-
"""
Projeto Mecânico Simplificado de uma Linha de Transmissão usando Python

Arquivos necessários:
- route.csv deve estar na mesma pasta deste script.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve 

# Fecha figuras antigas ao reexecutar o script
plt.close("all")

# -----------------------------------------------------------------------------
# 1. Parâmetros físicos do cabo
# -----------------------------------------------------------------------------
Tnom = 140.6e3        # tração nominal [N]
massa_cabo = 1473    # massa do cabo [kg/km]
g = 9.81             # aceleração da gravidade [m/s²]

Tmax = 0.25 * Tnom              # tração máxima admissível [N]
mu_s = massa_cabo * g / 1000    # peso por unidade de comprimento [N/m]

# -----------------------------------------------------------------------------
# 2. Leitura do perfil topográfico
# -----------------------------------------------------------------------------
# O arquivo route.csv deve estar na mesma pasta deste script.
# usecols=(3, 2): coluna 3 -> distância; coluna 2 -> elevação.
M = np.loadtxt(
    "route.csv",
    delimiter=",",
    skiprows=1,
    usecols=(3, 2)
)

# -----------------------------------------------------------------------------
# 3. Conversão de unidades e separação das colunas
# -----------------------------------------------------------------------------
# A distância é lida em km e convertida para m.
M[:, 0] *= 1000

# Variáveis auxiliares para tornar o código mais legível.
distancia = M[:, 0]   # distância ao longo da rota [m]
elevacao = M[:, 1]    # elevação do terreno [m]


# -----------------------------------------------------------------------------
# 4. Curva de altura mínima
# -----------------------------------------------------------------------------
rmin = 6.0                   # altura mínima em relação ao terreno [m]
y_min = elevacao + rmin      # elevação mínima admissível para o cabo [m]

# -----------------------------------------------------------------------------
# Correção da altura mínima para regiões especiais
# -----------------------------------------------------------------------------
# Cada linha: [posição inicial [m], posição final [m], altura mínima [m]]
# Se posição inicial = posição final, a restrição é pontual.

restricoes_altura = np.array([
    [200.0, 250.0, 8.0],   # estrada rural [m]
    [400.0, 440.0, 8.0]    # estrada rural [m]
])

for x_ini, x_fim, h_min in restricoes_altura:

    if x_ini == x_fim:
        indice = np.argmin(np.abs(distancia - x_ini))
        y_min[indice] = elevacao[indice] + h_min

    else:
        indices = np.where((distancia >= x_ini) &
                           (distancia <= x_fim))[0]
        y_min[indices] = elevacao[indices] + h_min
        
# -----------------------------------------------------------------------------
# 5. Posicionamento das torres
# -----------------------------------------------------------------------------

# Cada linha representa:
# [posição horizontal desejada, altura da torre]

torres_desejadas = np.array([
    [0.0,   20.0],
    [290.0, 20.0]
])

torres = []

for x_desejado, H in torres_desejadas:

    indice = np.argmin(np.abs(distancia - x_desejado))

    x_torre = distancia[indice]
    y_base = elevacao[indice]
    y_topo = y_base + H

    torres.append([x_torre, y_base, y_topo])

torres = np.array(torres)

# -----------------------------------------------------------------------------
# 6. Gráfico final da Aula 1 (Comentar Seção para continuação do script)
# -----------------------------------------------------------------------------
# plt.figure()
# plt.plot(distancia, elevacao, "b", label="Terreno")
# plt.plot(distancia, y_min, "r", label="Altura mínima")

# for i in range(len(torres)):
#     x_torre = torres[i, 0]
#     y_base = torres[i, 1]
#     y_topo = torres[i, 2]

#     label = "Torres" if i == 0 else None
#     plt.plot([x_torre, x_torre], [y_base, y_topo], "k", label=label)

# plt.xlabel("Distância [m]")
# plt.ylabel("Elevação [m]")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# -----------------------------------------------------------------------------
# 7. Tração horizontal por vão
# ----------------------------------------------------------------
# Cada elemento de T0_vaos corresponde a um vão entre torres consecutivas.
# Para n torres, devem existir n - 1 valores de T0.
T0_vaos = np.array(
    [0.96 * Tmax]
                   )

if len(T0_vaos) != len(torres) - 1:
    raise ValueError("O número de valores em T0_vaos deve ser igual a len(torres) - 1.")

# ----------------------------------
# 8. Equação não linear para encontrar x0
# ----------------------------------
def equacao_x0(x0_array, xt1, xt2, yt1, yt2, T0, mu_s):
    x0 = x0_array[0]
    termo_1 = np.cosh((mu_s / T0) * (xt1 - x0))
    termo_2 = np.cosh((mu_s / T0) * (xt2 - x0))
    return yt1 - yt2 - (T0 / mu_s) * (termo_1 - termo_2)

# -----------------------------------------------------------------------------
# 9. Função para calcular uma catenária
# -----------------------------------------------------------------------------
def calcular_catenaria(xt1, yt1, xt2, yt2, T0, mu_s, dx=1.0):
    parametros = (xt1, xt2, yt1, yt2, T0, mu_s)
    x0 = fsolve(equacao_x0, [xt1], args=parametros)[0]
    y0 = yt1 - (T0 / mu_s) * (
    np.cosh((mu_s / T0) * (xt1 - x0)) - 1
    )
    x_real = np.arange(xt1, xt2 + dx, dx)
    y_real = (T0 / mu_s) * (
    np.cosh((mu_s / T0) * (x_real - x0)) - 1
    ) + y0
    y_local = y_real - y0
    T = T0 + mu_s * y_local
    return x_real, y_real, T, x0, y

