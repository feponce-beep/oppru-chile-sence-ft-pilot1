# =============================================================================
# fase1_analisis_descriptivo.py
# Tesis Doctoral - Policy Paper | Franquicia Tributaria SENCE
# Universidad Complutense de Madrid - Doctorado en Sociologia
# =============================================================================
# Descripcion:
#   Fase 1: Analisis descriptivo de la Encuesta Longitudinal de Empresas (ELE-7)
#   Periodo: 2020-2022
#   Objetivo: Caracterizar la distribucion de empresas segun tamano, sector,
#   participacion en la Franquicia Tributaria SENCE, y variables de resultado
#   (salarios medios, horas de formacion, inversion en capacitacion).
#
# Datos: ELE-7 (INE Chile) - NO se incluyen microdatos confidenciales.
#        Este script opera sobre datos anonimizados o diccionarios de variables.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Configuracion general
# ---------------------------------------------------------------------------
RAND_SEED = 42
np.random.seed(RAND_SEED)

OUTPUT_DIR = Path("outputs/fase1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Paleta institucional UCM
COLOR_PRIMARY = "#003A70"   # azul UCM
COLOR_ACCENT  = "#C8102E"   # rojo UCM
COLOR_NEUTRAL = "#6C757D"

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

# ---------------------------------------------------------------------------
# 1. Diccionario de variables ELE-7 (referencia - sin microdatos)
# ---------------------------------------------------------------------------
VAR_DICT = {
    "id_empresa":        "Identificador anonimizado de empresa",
    "tamano":            "Tamano empresa (micro/pequena/mediana/grande)",
    "sector_ciiu":       "Sector economico (CIIU Rev.4)",
    "region":            "Region administrativa (Chile)",
    "usa_franquicia":    "Uso Franquicia Tributaria SENCE (1=Si, 0=No)",
    "monto_ft":          "Monto utilizado en Franquicia Tributaria (CLP)",
    "hrs_capacitacion":  "Horas totales de capacitacion por trabajador",
    "salario_medio":     "Salario promedio mensual (CLP)",
    "n_trabajadores":    "Numero de trabajadores equivalentes",
    "anio":              "Anio de referencia (2020, 2021, 2022)",
}

print("Diccionario de variables ELE-7:")
for var, desc in VAR_DICT.items():
    print(f"  {var:25s}: {desc}")

# ---------------------------------------------------------------------------
# 2. Simulacion de datos representativos (en ausencia de microdatos)
#    Distribucion basada en estadisticas publicadas ELE-7 (INE, 2023)
# ---------------------------------------------------------------------------

n_empresas = 5_000

np.random.seed(RAND_SEED)

tamano_dist = np.random.choice(
    ["Micro", "Pequena", "Mediana", "Grande"],
    size=n_empresas,
    p=[0.45, 0.30, 0.17, 0.08]
)

sector_dist = np.random.choice(
    ["Comercio", "Industria", "Servicios", "Construccion", "Agricultura", "Otros"],
    size=n_empresas,
    p=[0.25, 0.15, 0.30, 0.12, 0.08, 0.10]
)

# Probabilidad de uso FT segun tamano
ft_prob = {"Micro": 0.05, "Pequena": 0.18, "Mediana": 0.42, "Grande": 0.71}
usa_ft = np.array([np.random.binomial(1, ft_prob[t]) for t in tamano_dist])

# Salario medio (CLP, aproximacion ELE-7)
base_sal = {"Micro": 450_000, "Pequena": 620_000, "Mediana": 820_000, "Grande": 1_150_000}
salario_medio = np.array([
    max(350_000, np.random.normal(base_sal[t], base_sal[t] * 0.25))
    for t in tamano_dist
])

# Horas de capacitacion
hrs_cap = np.where(
    usa_ft == 1,
    np.random.gamma(shape=3, scale=8, size=n_empresas),
    np.random.gamma(shape=1.2, scale=4, size=n_empresas)
)

df = pd.DataFrame({
    "id_empresa":       range(1, n_empresas + 1),
    "tamano":           tamano_dist,
    "sector_ciiu":      sector_dist,
    "usa_franquicia":   usa_ft,
    "salario_medio":    salario_medio.round(0),
    "hrs_capacitacion": hrs_cap.round(1),
    "anio":             np.random.choice([2020, 2021, 2022], size=n_empresas),
})

print(f"\nDataset simulado: {len(df):,} empresas")
print(df.dtypes)

# ---------------------------------------------------------------------------
# 3. Estadisticas descriptivas
# ---------------------------------------------------------------------------
print("\n--- Uso Franquicia Tributaria por tamano ---")
ft_tamano = df.groupby("tamano")["usa_franquicia"].agg(["sum", "mean", "count"])
ft_tamano.columns = ["n_usa_ft", "tasa_uso", "n_total"]
ft_tamano["tasa_uso_pct"] = (ft_tamano["tasa_uso"] * 100).round(1)
print(ft_tamano.to_string())

print("\n--- Salario medio por tamano y uso FT ---")
sal_desc = df.groupby(["tamano", "usa_franquicia"])["salario_medio"].describe()
print(sal_desc.round(0).to_string())

# ---------------------------------------------------------------------------
# 4. Visualizaciones
# ---------------------------------------------------------------------------

# Fig 1: Tasa de uso FT por tamano
fig, ax = plt.subplots(figsize=(8, 5))
order = ["Micro", "Pequena", "Mediana", "Grande"]
colors = [COLOR_PRIMARY if t in ["Mediana", "Grande"] else COLOR_NEUTRAL for t in order]
bars = ax.bar(
    ft_tamano.loc[order, "tasa_uso_pct"].index,
    ft_tamano.loc[order, "tasa_uso_pct"].values,
    color=colors, edgecolor="white", linewidth=0.8
)
ax.set_xlabel("Tamano de empresa", fontsize=11)
ax.set_ylabel("Tasa de uso FT SENCE (%)", fontsize=11)
ax.set_title("Tasa de uso Franquicia Tributaria por tamano\n(ELE-7, 2020-2022 - datos simulados)",
             fontsize=12, pad=12)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
for bar, val in zip(bars, ft_tamano.loc[order, "tasa_uso_pct"].values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig1_tasa_uso_ft_tamano.png")
plt.close()
print("Figura guardada: fig1_tasa_uso_ft_tamano.png")

# Fig 2: Distribucion salarios por uso FT
fig, ax = plt.subplots(figsize=(8, 5))
df_ft = df[df["usa_franquicia"] == 1]["salario_medio"] / 1_000
df_no = df[df["usa_franquicia"] == 0]["salario_medio"] / 1_000
ax.hist(df_no, bins=40, alpha=0.6, color=COLOR_NEUTRAL, label="No usa FT")
ax.hist(df_ft, bins=40, alpha=0.7, color=COLOR_PRIMARY, label="Usa FT")
ax.axvline(df_ft.mean(), color=COLOR_ACCENT, linestyle="--", lw=1.5,
           label=f"Media FT: ${df_ft.mean():.0f}K")
ax.axvline(df_no.mean(), color=COLOR_NEUTRAL, linestyle="--", lw=1.5,
           label=f"Media no-FT: ${df_no.mean():.0f}K")
ax.set_xlabel("Salario medio (miles CLP)", fontsize=11)
ax.set_ylabel("Frecuencia", fontsize=11)
ax.set_title("Distribucion salario medio segun uso Franquicia Tributaria\n(ELE-7, 2020-2022 - datos simulados)",
             fontsize=12, pad=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig2_dist_salarios_ft.png")
plt.close()
print("Figura guardada: fig2_dist_salarios_ft.png")

# Fig 3: Boxplot horas capacitacion por tamano
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df, x="tamano", y="hrs_capacitacion", order=order,
            hue="usa_franquicia", palette=[COLOR_NEUTRAL, COLOR_PRIMARY],
            flierprops=dict(marker="o", markersize=2, alpha=0.4), ax=ax)
ax.set_xlabel("Tamano de empresa", fontsize=11)
ax.set_ylabel("Horas de capacitacion por trabajador", fontsize=11)
ax.set_title("Horas de capacitacion por tamano y uso FT\n(ELE-7, 2020-2022 - datos simulados)",
             fontsize=12, pad=12)
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, ["No usa FT", "Usa FT"], title="Franquicia Tributaria", fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig3_horas_cap_tamano.png")
plt.close()
print("Figura guardada: fig3_horas_cap_tamano.png")

print("\n[Fase 1 completada] Outputs en:", OUTPUT_DIR)
