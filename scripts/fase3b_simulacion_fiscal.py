# =============================================================================
# fase3b_simulacion_fiscal.py
# Tesis Doctoral - Policy Paper | Franquicia Tributaria SENCE
# Universidad Complutense de Madrid - Doctorado en Sociologia
# =============================================================================
# Descripcion:
#   Fase 3b: Simulacion del impacto fiscal de la Franquicia Tributaria SENCE
#   Modelado del costo fiscal agregado, distribucion por tamano de empresa,
#   y escenarios contrafactuales de reforma.
#
# Escenarios analizados:
#   1. Escenario base: uso real FT segun ELE-7 (2020-2022)
#   2. Escenario reforma: eliminacion gradual para empresas grandes
#   3. Escenario focalizado: redireccion hacia PyMEs
#
# Datos: parametros publicos SII/SENCE + simulacion (sin microdatos)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

# ---------------------------------------------------------------------------
# 0. Configuracion
# ---------------------------------------------------------------------------
RAND_SEED = 42
np.random.seed(RAND_SEED)

OUTPUT_DIR = Path("outputs/fase3b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_BASE     = "#003A70"
COLOR_REFORMA  = "#C8102E"
COLOR_FOCALIZ  = "#28A745"
COLOR_NEUTRAL  = "#6C757D"

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# 1. Parametros fiscales del sistema FT SENCE
#    Fuente: SII, SENCE, DIPRES Chile (valores publicos 2022)
# ---------------------------------------------------------------------------
@dataclass
class ParamsFiscales:
    """Parametros publicos del sistema Franquicia Tributaria SENCE."""
    # Limite FT como fraccion de planilla (segun Ley 19.518)
    limite_ft_pequena:   float = 0.015   # 1.5% planilla anual
    limite_ft_mediana:   float = 0.010   # 1.0% planilla anual
    limite_ft_grande:    float = 0.008   # 0.8% planilla anual

    # Tasa impuesto de primera categoria (2022)
    tasa_impuesto:       float = 0.27

    # Numero estimado empresas por segmento (ELE-7 / SII)
    n_pequena:           int   = 85_000
    n_mediana:           int   = 15_000
    n_grande:            int   = 4_500

    # Planilla anual media por segmento (CLP)
    planilla_pequena:    float = 120_000_000    # ~120M CLP
    planilla_mediana:    float = 850_000_000    # ~850M CLP
    planilla_grande:     float = 8_500_000_000  # ~8.500M CLP

    # Tasa de utilizacion FT (fraccion de empresas que la usan)
    uso_pequena:         float = 0.18
    uso_mediana:         float = 0.42
    uso_grande:          float = 0.71

    # Fraccion del limite FT efectivamente utilizada
    eficiencia_uso:      float = 0.65

p = ParamsFiscales()

# ---------------------------------------------------------------------------
# 2. Calculo costo fiscal base
# ---------------------------------------------------------------------------
def calcular_costo_fiscal(params, escenario_uso=None, escenario_limite=None):
    """
    Calcula el costo fiscal agregado del sistema FT SENCE.

    Args:
        params: ParamsFiscales con parametros base
        escenario_uso: dict opcional con tasas de uso alternativas
        escenario_limite: dict opcional con limites FT alternativos

    Returns:
        DataFrame con resultados por segmento y agregados
    """
    uso    = escenario_uso    or {"pequena": params.uso_pequena,
                                  "mediana": params.uso_mediana,
                                  "grande":  params.uso_grande}
    limite = escenario_limite or {"pequena": params.limite_ft_pequena,
                                   "mediana": params.limite_ft_mediana,
                                   "grande":  params.limite_ft_grande}

    segmentos = [
        ("Pequena", params.n_pequena, params.planilla_pequena,
         uso["pequena"], limite["pequena"]),
        ("Mediana", params.n_mediana, params.planilla_mediana,
         uso["mediana"], limite["mediana"]),
        ("Grande",  params.n_grande,  params.planilla_grande,
         uso["grande"],  limite["grande"]),
    ]

    rows = []
    for seg, n_emp, planilla, tasa_uso, lim_ft in segmentos:
        n_usa_ft       = n_emp * tasa_uso
        ft_por_empresa = planilla * lim_ft * params.eficiencia_uso
        ft_total       = n_usa_ft * ft_por_empresa
        # Costo fiscal = perdida recaudacion (credito contra impuesto)
        costo_fiscal   = ft_total * params.tasa_impuesto
        rows.append({
            "Segmento":         seg,
            "n_empresas":       n_emp,
            "n_usa_ft":         round(n_usa_ft, 0),
            "ft_total_MCLP":    round(ft_total / 1e6, 1),
            "costo_fiscal_MCLP":round(costo_fiscal / 1e6, 1),
        })

    df = pd.DataFrame(rows)
    df.loc["TOTAL"] = ["-", df["n_empresas"].sum(), df["n_usa_ft"].sum(),
                       df["ft_total_MCLP"].sum(), df["costo_fiscal_MCLP"].sum()]
    return df

df_base = calcular_costo_fiscal(p)
print("=== Escenario Base ===")
print(df_base.to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Escenarios de reforma
# ---------------------------------------------------------------------------

# Escenario 1: Reforma - eliminar FT para grandes empresas
df_reforma = calcular_costo_fiscal(
    p,
    escenario_uso={"pequena": p.uso_pequena, "mediana": p.uso_mediana, "grande": 0.0},
    escenario_limite={"pequena": p.limite_ft_pequena, "mediana": p.limite_ft_mediana, "grande": 0.0}
)
print("\n=== Escenario Reforma (sin FT empresas grandes) ===")
print(df_reforma.to_string(index=False))

# Escenario 2: Focalizacion - ampliar uso PyME, reducir grandes
df_focaliz = calcular_costo_fiscal(
    p,
    escenario_uso={"pequena": 0.35, "mediana": 0.55, "grande": 0.30},
    escenario_limite={"pequena": 0.020, "mediana": 0.012, "grande": 0.004}
)
print("\n=== Escenario Focalizacion PyME ===")
print(df_focaliz.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Comparacion escenarios
# ---------------------------------------------------------------------------

# Costo fiscal total por escenario
costo_base    = df_base.loc["TOTAL",    "costo_fiscal_MCLP"]
costo_reforma = df_reforma.loc["TOTAL", "costo_fiscal_MCLP"]
costo_focal   = df_focaliz.loc["TOTAL", "costo_fiscal_MCLP"]

print(f"\n--- Comparacion costo fiscal total (millones CLP) ---")
print(f"Base:         {costo_base:>12,.1f} M CLP")
print(f"Reforma:      {costo_reforma:>12,.1f} M CLP  ({(costo_reforma/costo_base-1)*100:+.1f}%)")
print(f"Focalizacion: {costo_focal:>12,.1f} M CLP  ({(costo_focal/costo_base-1)*100:+.1f}%)")

# ---------------------------------------------------------------------------
# 5. Visualizaciones
# ---------------------------------------------------------------------------

# Fig 1: Costo fiscal por segmento (escenario base)
df_segs = df_base[df_base["Segmento"] != "-"].copy()
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(df_segs["Segmento"], df_segs["costo_fiscal_MCLP"],
              color=[COLOR_NEUTRAL, COLOR_BASE, COLOR_REFORMA], edgecolor="white")
ax.set_xlabel("Segmento de empresa", fontsize=11)
ax.set_ylabel("Costo fiscal (millones CLP)", fontsize=11)
ax.set_title("Costo fiscal estimado Franquicia Tributaria SENCE\npor segmento de empresa (2022, escenario base)",
             fontsize=12, pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f} M"))
for bar, val in zip(bars, df_segs["costo_fiscal_MCLP"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f"{val:,.0f} M", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig1_costo_fiscal_base.png")
plt.close()
print("Figura guardada: fig1_costo_fiscal_base.png")

# Fig 2: Comparacion escenarios
escenarios = ["Base", "Reforma\n(sin grandes)", "Focalizacion\nPyME"]
costos = [costo_base, costo_reforma, costo_focal]
colores = [COLOR_BASE, COLOR_REFORMA, COLOR_FOCALIZ]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(escenarios, costos, color=colores, edgecolor="white", width=0.55)
ax.set_ylabel("Costo fiscal total (millones CLP)", fontsize=11)
ax.set_title("Comparacion costo fiscal FT SENCE por escenario de reforma\n(simulacion parametrica, 2022)",
             fontsize=12, pad=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f} M"))
for bar, val, base in zip(bars, costos, [costo_base]*3):
    pct = (val/costo_base - 1) * 100
    label = f"{val:,.0f} M\n({pct:+.1f}%)"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            label, ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig2_comparacion_escenarios.png")
plt.close()
print("Figura guardada: fig2_comparacion_escenarios.png")

# Fig 3: Distribucion costo fiscal por segmento (stacked, 3 escenarios)
fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
for ax, (df_esc, titulo, color) in zip(
    axes,
    [(df_base,    "Base",          COLOR_BASE),
     (df_reforma, "Reforma",       COLOR_REFORMA),
     (df_focaliz, "Focalizacion",  COLOR_FOCALIZ)]
):
    segs = df_esc[df_esc["Segmento"] != "-"]
    ax.bar(segs["Segmento"], segs["costo_fiscal_MCLP"], color=color, alpha=0.85)
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("Segmento", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
axes[0].set_ylabel("Costo fiscal (M CLP)", fontsize=11)
fig.suptitle("Distribucion costo fiscal FT SENCE por escenario y segmento",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig3_distribucion_escenarios.png", bbox_inches="tight")
plt.close()
print("Figura guardada: fig3_distribucion_escenarios.png")

print("\n[Fase 3b completada] Outputs en:", OUTPUT_DIR)
