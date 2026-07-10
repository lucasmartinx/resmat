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
M = np.loadtxt(
    "route.csv",
    delimiter=",",
    skiprows=1,
    usecols=(3, 2)
)

# -----------------------------------------------------------------------------
# 3. Conversão de unidades e separação das colunas
# -----------------------------------------------------------------------------
M[:, 0] *= 1000

distancia = M[:, 0]   # distância ao longo da rota [m]
elevacao = M[:, 1]    # elevação do terreno [m]

# -----------------------------------------------------------------------------
# 4. Curva de altura mínima
# -----------------------------------------------------------------------------
rmin = 6.0                   # altura mínima em relação ao terreno [m]
y_min = elevacao + rmin      # elevação mínima admissível para o cabo [m]

# Correção da altura mínima para regiões especiais
restricoes_altura = np.array([
    [200.0, 250.0, 8.0],   # estrada rural [m]
    [400.0, 440.0, 8.0]    # estrada rural [m]
])

for x_ini, x_fim, h_min in restricoes_altura:
    if x_ini == x_fim:
        indice = np.argmin(np.abs(distancia - x_ini))
        y_min[indice] = elevacao[indice] + h_min
    else:
        indices = np.where((distancia >= x_ini) & (distancia <= x_fim))[0]
        y_min[indices] = elevacao[indices] + h_min
        
# -----------------------------------------------------------------------------
# 5. Posicionamento das torres (ATUALIZADO)
# -----------------------------------------------------------------------------
# Adicionadas as torres 4 e 5 para cobrir o perfil do route.csv
torres_desejadas = np.array([
    [210.0,  20.0],  # Torre 1
    [530.0,  20.0],  # Torre 2
    [1000.0,  20.0],  # Torre 3
  #  [1200.0, 25.0],  # Torre 4 
    [1500.0, 20.0]   # Torre 5
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
# 7. Tração horizontal por vão (ATUALIZADO)
# -----------------------------------------------------------------------------
# Como temos 5 torres, precisamos de exatamente 4 valores de tração (n - 1).
T0_vaos = np.array([
    0.96 * Tmax,  # Vão 1
    0.96 * Tmax,  # Vão 2
    0.96 * Tmax,  # Vão 3
   # 0.96 * Tmax   # Vão 4
])

if len(T0_vaos) != len(torres) - 1:
    raise ValueError(
        "O número de valores em T0_vaos deve ser igual a len(torres) - 1."
    )

# --------------------------------------------------------------------
# 8. Equação não linear para encontrar x0
# --------------------------------------------------------
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

    return x_real, y_real, T, x0, y0

# -----------------------------------------------------------------------------
# 10. Cálculo das catenárias entre torres consecutivas (CORRIGIDO)
# -----------------------------------------------------------------------------
catenarias = []

for i in range(len(torres) - 1):
    # Coordenadas da torre inicial do vão
    xt1 = torres[i, 0]
    yt1 = torres[i, 2]

    # Coordenadas da torre final do vão
    xt2 = torres[i + 1, 0]
    yt2 = torres[i + 1, 2]

    T0 = T0_vaos[i]

    # Chamada correta utilizando apenas os dados do par de torres do vão
    x_cat, y_cat, T_cat, x0, y0 = calcular_catenaria(
        xt1, yt1, xt2, yt2, T0, mu_s
    )

    catenarias.append({
        "vao": i + 1,
        "xt1": xt1,
        "xt2": xt2,
        "T0": T0,
        "x": x_cat,
        "y": y_cat,
        "T": T_cat,
        "x0": x0,
        "y0": y0
    })
    
# -----------------------------------------------------------------------------
# 11. Gráfico com todas as catenárias
# -----------------------------------------------------------------------------
plt.figure(1)
plt.plot(distancia, elevacao, "b", label="Terreno")
plt.plot(distancia, y_min, "r", label="Altura mínima")

for i in range(len(torres)):
    x_torre = torres[i, 0]
    y_base = torres[i, 1]
    y_topo = torres[i, 2]
    label = "Torres" if i == 0 else None
    plt.plot([x_torre, x_torre], [y_base, y_topo], "k", label=label)

for i, cat in enumerate(catenarias):
    label = "Catenária" if i == 0 else None
    plt.plot(cat["x"], cat["y"], "m", label=label)

plt.xlabel("Distância [m]")
plt.ylabel("Elevação [m]")
plt.grid(True)
plt.legend()
plt.tight_layout()

# -----------------------------------------------------------------------------
# 12. Gráfico da tração em todos os vãos
# -----------------------------------------------------------------------------
plt.figure(2)

for i, cat in enumerate(catenarias):
    label = "Tração no cabo" if i == 0 else None
    plt.plot(cat["x"], cat["T"], "b", label=label)

T_limite_global = Tmax * np.ones(len(distancia))
plt.plot(distancia, T_limite_global, "r", label="Tração máxima admissível")

plt.xlabel("Distância [m]")
plt.ylabel("Tração [N]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
