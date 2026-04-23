"""
Genera figuras longitudinales definitivas v5.1
  - figura5_tasas_longitudinal.png: tasas por tamaño a lo largo de 4 olas
  - figura6_prima_salarial_longitudinal.png: β_SAL_PYME con IC95% por ola
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "resultados"
FIG = ROOT / "figuras"

df = pd.read_csv(RES / "tabla_longitudinal_v5.csv")

AZUL_OSCURO = "#1A3D5C"
AZUL_MEDIO = "#2E75B6"
AZUL_CLARO = "#A8C4D4"
ROJO = "#C0392B"
VERDE = "#1A6B2A"
GRIS = "#7F8C8D"

# ═══════════════════════════════════════════════════════════════════════════
# FIGURA 5 — Tasas por tamaño, serie longitudinal
# ═══════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel izquierdo: tasas absolutas por tamaño
years = df["anio"].tolist()
ax1.plot(years, df["tasa_grande_pct"], marker="o", markersize=11, linewidth=2.5,
         color=AZUL_OSCURO, label="Grande")
ax1.plot(years, df["tasa_pyme_pct"], marker="s", markersize=11, linewidth=2.5,
         color=AZUL_MEDIO, label="PYME")
ax1.plot(years, df["tasa_micro_pct"], marker="^", markersize=11, linewidth=2.5,
         color=AZUL_CLARO, label="Micro")

for idx, row in df.iterrows():
    ax1.annotate(f"{row['tasa_grande_pct']:.1f}%",
                 (row["anio"], row["tasa_grande_pct"]),
                 textcoords="offset points", xytext=(0, 12),
                 ha="center", fontsize=9, color=AZUL_OSCURO, fontweight="bold")
    ax1.annotate(f"{row['tasa_pyme_pct']:.1f}%",
                 (row["anio"], row["tasa_pyme_pct"]),
                 textcoords="offset points", xytext=(0, -16),
                 ha="center", fontsize=8, color=AZUL_MEDIO)

ax1.set_xlabel("Año de referencia", fontsize=11)
ax1.set_ylabel("Tasa de actividad formativa (%)", fontsize=11)
ax1.set_title("Panel A. Tasa de actividad formativa por tamaño, 2013-2022",
              fontsize=12, fontweight="bold", pad=10)
ax1.set_xticks(years)
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_ylim(0, max(df["tasa_grande_pct"]) * 1.18)

# Panel derecho: ratio Grande/PYME
ratios = df["ratio_grande_pyme"].tolist()
bars = ax2.bar(range(len(df)), ratios, color=[AZUL_CLARO, AZUL_MEDIO, AZUL_MEDIO, AZUL_OSCURO],
               edgecolor="white", linewidth=1.5, width=0.65)
for bar, r, y in zip(bars, ratios, years):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
             f"{r:.2f}×", ha="center", fontsize=11, fontweight="bold", color="#222")

ax2.set_xticks(range(len(df)))
ax2.set_xticklabels([f"{row['ola']}\n({row['anio']})" for _, row in df.iterrows()], fontsize=10)
ax2.set_ylabel("Ratio Grande / PYME", fontsize=11)
ax2.set_title("Panel B. Concentración del beneficio formativo, 2013-2022",
              fontsize=12, fontweight="bold", pad=10)
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(True, alpha=0.3, axis="y")
ax2.set_ylim(0, max(ratios) * 1.2)
ax2.axhline(y=1, color=GRIS, linestyle="--", linewidth=0.8, alpha=0.5)

fig.suptitle("Figura 5. Actividad formativa en empresas chilenas: evolución longitudinal\n"
             "Fuente: ELE-4, ELE-5, ELE-6, ELE-7 · Ponderado por FE_TRANSVERSAL · Dotación corregida (persona-mes/12)",
             fontsize=13, fontweight="bold", y=1.02)

fig.text(0.5, -0.03,
         "Nota: CAP_ALLY=1 si la empresa declaró capacitación como motivo de asociación con otras empresas (D097, D176, D106). "
         "Proxy de actividad formativa; no equivale al uso de la Franquicia Tributaria SENCE. "
         "En ELE-4 solo D097 y D106 están disponibles.",
         ha="center", fontsize=9, style="italic", color=GRIS, wrap=True)

plt.tight_layout()
fig.savefig(FIG / "figura5_tasas_longitudinal.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figura 5 guardada")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURA 6 — Prima salarial PYME por ola (forest plot)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 5.5))

primas = df["prima_sal_pyme_pct"].tolist()
ic_lo = df["ic95_pyme_inf"].tolist()
ic_hi = df["ic95_pyme_sup"].tolist()
labels = [f"{row['ola']}\n({row['anio']})" for _, row in df.iterrows()]

y_pos = range(len(df))
ax.axvline(x=0, color=GRIS, linestyle="--", linewidth=1, alpha=0.5)

for i, (p, lo, hi) in enumerate(zip(primas, ic_lo, ic_hi)):
    ax.errorbar(p, i,
                xerr=[[p - lo], [hi - p]],
                fmt="o", color=AZUL_OSCURO, markersize=12,
                capsize=7, capthick=1.8, elinewidth=2.2)
    sig_marker = "**"  # todos son significativos al 5%
    ax.text(hi + 1.5, i, f"+{p:.1f}% {sig_marker}  [IC95%: {lo:+.1f}%; {hi:+.1f}%]",
            va="center", fontsize=10, color="#222")

# Promedio
promedio = np.mean(primas)
ax.axvline(x=promedio, color=ROJO, linestyle=":", linewidth=1.3, alpha=0.7,
           label=f"Promedio: +{promedio:.1f}%")

ax.set_yticks(list(y_pos))
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Prima salarial PYME asociada a CAP_ALLY=1 (%, IC 95%)", fontsize=11)
ax.set_title("Figura 6. Prima salarial PYME asociada a actividad formativa, 2013-2022\n"
             "Coeficientes OLS con FE sector CIIU · Errores robustos HC3",
             fontsize=12, fontweight="bold", pad=12)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(-5, 42)
ax.set_ylim(-0.5, len(df) - 0.3)
ax.grid(True, alpha=0.3, axis="x")
ax.legend(loc="lower right", fontsize=10)

fig.text(0.5, -0.05,
         "Nota: todas las estimaciones son significativas al 5%. Los coeficientes no son comparables entre olas como efectos causales; "
         "son asociaciones condicionales de formación con masa salarial por trabajador, controlando por sector, antigüedad y estructura "
         "empresarial. La estabilidad del patrón (cuatro olas, diez años, primas entre +11,3% y +19,4%) es el hallazgo central.",
         ha="center", fontsize=9, style="italic", color=GRIS, wrap=True)

plt.tight_layout()
fig.savefig(FIG / "figura6_prima_salarial_longitudinal.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figura 6 guardada")
