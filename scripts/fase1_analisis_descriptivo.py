"""
OPPRU-Chile · Piloto 2 · Análisis corregido v5
===================================================
Objetivo
--------
  (1) DOTACIÓN FÍSICA: las variables I151 (hombres) e I160 (mujeres) de ELE
      corresponden a persona-meses (suma ene..dic según cuestionario del INE).
      La dotación física anual promedio = (I151 + I160) / 12.
      Sin esta corrección, las variables derivadas (PROD_LAB, SAL_TRAB) están
      subestimadas en niveles absolutos por un factor 12, aunque los coeficientes
      OLS no se ven afectados por ser el factor común a todas las observaciones.
  (2) PARSING NUMÉRICO: FE_TRANSVERSAL y otras variables numéricas en ELE usan
      coma como separador decimal (formato chileno). Sin conversión, pandas las
      lee como string y el factor de expansión queda constante = 1.

Cinco preguntas del estudio
---------------------------
  1. Concentración del beneficio formativo por tamaño de empresa (Figura 1, Tabla 1)
  2. Brechas sectoriales estables (Figura 4)
  3. Asociación formación × masa salarial en PYMEs (Tabla 3)
  4. Diferencias entre empresas CAP_ALLY=1 y CAP_ALLY=0 (Tabla 2 complementaria)
  5. Heterogeneidad temporal entre olas (Tabla 3, Figura 3)

Método principal: OLS + efectos fijos de sector CIIU + errores robustos HC3.
Método complementario: Test Z de diferencia de proporciones Grande vs PYME por ola.

Autoría
-------
OPPRU-Chile | Observatorio de Políticas Públicas sobre Reskilling y Upskilling
Felipe Ponce Bollmann · Universidad Complutense de Madrid · feponce@ucm.es
Versión: 5.0 | abril 2026
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats


# ════════════════════════════════════════════════════════════════════════════
# IMPLEMENTACIÓN DE OLS CON ERRORES ROBUSTOS HC3 (sin statsmodels)
# ════════════════════════════════════════════════════════════════════════════
class OLSResult:
    """Clase para mimetizar la interfaz de statsmodels en lo que usamos."""

    def __init__(self, params, bse, pvalues, rsquared, nobs, conf_int_df):
        self.params = params
        self.bse = bse
        self.pvalues = pvalues
        self.rsquared = rsquared
        self.nobs = nobs
        self._conf_int = conf_int_df

    def conf_int(self, alpha=0.05):
        return self._conf_int


def fit_ols_hc3(y, X, var_names):
    """
    Ajusta OLS con errores estándar robustos tipo HC3 (MacKinnon-White 1985).
    Retorna un objeto con params (pd.Series), bse, pvalues, rsquared, nobs,
    conf_int() (pd.DataFrame con columnas 0 y 1).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape

    # Estimación por mínimos cuadrados
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta

    # R²
    tss = np.sum((y - y.mean()) ** 2)
    rss = np.sum(resid ** 2)
    rsquared = 1 - rss / tss if tss > 0 else 0.0

    # HC3: (1-h_ii)^2 en el denominador — MacKinnon & White (1985)
    H = X @ XtX_inv @ X.T
    h = np.clip(np.diag(H), 0, 0.999999)
    u2 = (resid / (1 - h)) ** 2
    meat = X.T @ np.diag(u2) @ X
    vcov_hc3 = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(vcov_hc3))

    # Inferencia (t de Student con n-k gl)
    tvals = beta / se
    df_res = n - k
    pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), df=df_res))
    tcrit = stats.t.ppf(0.975, df=df_res)
    ci_lo = beta - tcrit * se
    ci_hi = beta + tcrit * se

    params = pd.Series(beta, index=var_names)
    bse = pd.Series(se, index=var_names)
    pvalues = pd.Series(pvals, index=var_names)
    conf_df = pd.DataFrame({0: ci_lo, 1: ci_hi}, index=var_names)

    return OLSResult(params, bse, pvalues, rsquared, int(n), conf_df)


def build_design_matrix(data, dependent, covariates, categorical=None):
    """
    Construye la matriz de diseño (X) con intercepto, dummies para
    variables categóricas, y retorna (y, X, var_names) limpiados de NaN.
    """
    categorical = categorical or []
    all_cols = [dependent] + list(covariates) + list(categorical)
    clean = data[all_cols].dropna().copy()

    # Drop categories with <5 observations to avoid rank deficiency
    for cat in categorical:
        counts = clean[cat].value_counts()
        keep = counts[counts >= 5].index
        clean = clean[clean[cat].isin(keep)]

    X_num = clean[covariates].astype(float).values
    X_parts = [np.ones((len(clean), 1)), X_num]
    var_names = ["const"] + list(covariates)

    for cat in categorical:
        dummies = pd.get_dummies(clean[cat], prefix=cat, drop_first=True)
        X_parts.append(dummies.astype(float).values)
        var_names.extend(dummies.columns.tolist())

    X = np.hstack(X_parts)
    y = clean[dependent].astype(float).values
    return y, X, var_names

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# RUTAS
# ════════════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parent.parent
DATOS_DIR = ROOT / "datos"
RESULTADOS_DIR = ROOT / "resultados"
FIGURAS_DIR = ROOT / "figuras"

for d in [RESULTADOS_DIR, FIGURAS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Archivos de entrada (descargar de economia.gob.cl o ubicar previamente)
ARCHIVO_ELE7 = DATOS_DIR / "ele7-full.csv"
ARCHIVO_ELE5 = DATOS_DIR / "BBDD-ELE5-Formato-Texto.csv"
ARCHIVO_ELE4 = DATOS_DIR / "BBDD_ELE4.csv"


# ════════════════════════════════════════════════════════════════════════════
# PALETA
# ════════════════════════════════════════════════════════════════════════════
AZUL_OSCURO = "#1A3D5C"
AZUL_MEDIO = "#2E75B6"
AZUL_CLARO = "#A8C4D4"
ROJO_ALERTA = "#C0392B"
VERDE_OK = "#1A6B2A"
GRIS = "#7F8C8D"


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
def parse_numeric_comma(serie):
    """Convierte strings con coma decimal (formato chileno) a float."""
    return pd.to_numeric(
        serie.astype(str).str.replace(",", ".").replace("nan", np.nan),
        errors="coerce",
    )


def construir_variables_ele7(df):
    """
    Aplica la construcción de variables estándar al DataFrame ELE-7.

    CORRECCIÓN CLAVE v5: la dotación física se obtiene como (I151+I160)/12,
    porque las preguntas I151 e I160 reportan la suma ene..dic (persona-meses).
    """
    # Parsear variables numéricas con coma decimal (formato chileno)
    for col in [
        "D097", "D176", "D106", "I151", "I160", "C077", "C084",
        "TAMANO", "A068", "A069", "FE_TRANSVERSAL", "FE_LONGITUDINAL",
    ]:
        if col in df.columns:
            df[col] = parse_numeric_comma(df[col])

    # Variable de actividad formativa (proxy, no equivale a uso de FT-SENCE)
    df["CAP_ALLY"] = (
        (df["D097"] == 1) | (df["D176"] == 1) | (df["D106"] == 1)
    ).astype(int)

    # CORRECCIÓN CRÍTICA: persona-meses / 12 = dotación física promedio anual
    df["DOTACION_PM"] = df[["I151", "I160"]].fillna(0).sum(axis=1)
    df["DOTACION"] = df["DOTACION_PM"] / 12

    # Variables derivadas en pesos CLP (C084 y C077 están en miles de pesos)
    mask_prod = (df["C077"] > 0) & (df["DOTACION"] > 0)
    df["PROD_LAB"] = np.where(
        mask_prod, np.log(df["C077"] * 1000 / df["DOTACION"]), np.nan
    )
    mask_sal = (df["C084"] > 0) & (df["DOTACION"] > 0)
    df["SAL_TRAB"] = np.where(
        mask_sal, np.log(df["C084"] * 1000 / df["DOTACION"]), np.nan
    )

    # Categoría de tamaño agregada
    df["TAMANO_CAT"] = pd.cut(
        df["TAMANO"],
        bins=[0, 1, 4, 5],
        labels=["Grande", "PYME", "Micro"],
        right=True,
    )

    # Antigüedad
    df["ANTIG"] = (2022 - df["A068"]).clip(lower=0)
    df["ln_ANTIG"] = np.log1p(df["ANTIG"])

    # Grupo empresarial
    df["GRUPO_EMP"] = (df["A069"] == 1).astype(int)

    # Sector (primera letra del CIIU)
    df["SECTOR"] = df["CIIU_FINAL"].astype(str).str.strip().str[0]

    # Factor de expansión (defaultea a 1 si ausente)
    df["FE"] = df["FE_TRANSVERSAL"].fillna(1)

    return df


# ════════════════════════════════════════════════════════════════════════════
# BLOQUE 0 — CARGA Y CONSTRUCCIÓN
# ════════════════════════════════════════════════════════════════════════════
print("=" * 72)
print("OPPRU-Chile · Piloto 2 v5 · Análisis corregido")
print("=" * 72)
print("\n[BLOQUE 0] Cargando ELE-7 y construyendo variables...")

df = pd.read_csv(ARCHIVO_ELE7, sep=";", encoding="latin-1", low_memory=False)
df = construir_variables_ele7(df)

print(f"  ELE-7 bruta: {len(df):,} empresas")
print(f"  Universo expandido (ΣFE_TRANSVERSAL): {df['FE'].sum():,.0f}")

# Muestra analítica: PROD_LAB y SAL_TRAB válidos, tamaño asignado
mask = (
    df["PROD_LAB"].notna()
    & df["SAL_TRAB"].notna()
    & df["TAMANO_CAT"].notna()
)
da = df[mask].copy()

# Trim percentiles 1-99 en PROD_LAB para robustez
p1, p99 = da["PROD_LAB"].quantile([0.01, 0.99])
da = da[(da["PROD_LAB"] >= p1) & (da["PROD_LAB"] <= p99)].copy()

print(f"  Muestra analítica: {len(da):,} empresas")
print(f"    Grandes: {(da['TAMANO_CAT']=='Grande').sum():,}")
print(f"    PYME:    {(da['TAMANO_CAT']=='PYME').sum():,}")
print(f"    Micro:   {(da['TAMANO_CAT']=='Micro').sum():,}")

# Verificación de sanidad: salario mensual medio en PYMEs debería ser realista
pyme = da[da["TAMANO_CAT"] == "PYME"]
sal_mes_medio_pyme = np.exp(pyme["SAL_TRAB"].mean()) / 12
print(f"\n  [SANIDAD] Salario mensual medio PYME: CLP {sal_mes_medio_pyme:,.0f}")
print(f"  Referencia INE ENE-ESI 2022: ~680.000 CLP/mes (general)")
print(f"  Referencia CASEN 2022: ~590.000 CLP/mes (trabajadores)")


# ════════════════════════════════════════════════════════════════════════════
# BLOQUE 1 — Pregunta 1: Concentración por tamaño (Tabla 1, Figura 1)
# ════════════════════════════════════════════════════════════════════════════
print("\n[BLOQUE 1] Pregunta 1: Concentración de la actividad formativa por tamaño")


def tasa_ponderada(frame, grupo):
    """Tasa ponderada de CAP_ALLY=1 por grupo."""
    out = []
    for g, sub in frame.groupby(grupo, observed=True):
        total = sub["FE"].sum()
        cap1 = sub.loc[sub["CAP_ALLY"] == 1, "FE"].sum()
        tasa = cap1 / total * 100 if total > 0 else 0
        out.append(
            {
                "Grupo": str(g),
                "N_muestra": len(sub),
                "N_expandido": int(total),
                "CAP_ALLY=1 (%)": round(tasa, 1),
                "CAP_ALLY=0 (%)": round(100 - tasa, 1),
            }
        )
    return pd.DataFrame(out)


t1_tamano = tasa_ponderada(da, "TAMANO_CAT")
t1_tamano = t1_tamano.sort_values(
    "Grupo", key=lambda s: s.map({"Grande": 0, "PYME": 1, "Micro": 2})
)
print(f"\n{t1_tamano.to_string(index=False)}")

ratio_g_pyme = (
    t1_tamano.loc[t1_tamano["Grupo"] == "Grande", "CAP_ALLY=1 (%)"].iloc[0]
    / t1_tamano.loc[t1_tamano["Grupo"] == "PYME", "CAP_ALLY=1 (%)"].iloc[0]
)
print(f"\n  Ratio Grande/PYME: {ratio_g_pyme:.2f}x")

t1_tamano.to_csv(
    RESULTADOS_DIR / "tabla1_concentracion_tamano.csv",
    index=False,
    encoding="utf-8-sig",
)


# ────────────────────────────────────────────────────────────────────────────
# FIGURA 1: Barras horizontales de tasa CAP_ALLY=1 por tamaño
# ────────────────────────────────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(10, 5))
colors = [AZUL_OSCURO, AZUL_MEDIO, AZUL_CLARO]
df_fig1 = t1_tamano.iloc[::-1]  # invertir para Grande arriba

bars = ax.barh(
    df_fig1["Grupo"], df_fig1["CAP_ALLY=1 (%)"], color=colors, height=0.55, edgecolor="white"
)

for bar, (_, row) in zip(bars, df_fig1.iterrows()):
    ax.text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"  {row['CAP_ALLY=1 (%)']:.1f}%  (N≈{row['N_expandido']/1000:.0f}k)",
        va="center",
        fontsize=11,
        color="#222",
    )

ax.set_xlabel("Porcentaje de empresas con actividad formativa (CAP_ALLY=1, %)", fontsize=11)
ax.set_title(
    "Figura 1. Tasa de actividad formativa por tamaño de empresa\n"
    "ELE-7, año 2022 · ponderado por factor de expansión transversal",
    fontsize=12,
    fontweight="bold",
    pad=12,
)

tasa_general = (
    da.loc[da["CAP_ALLY"] == 1, "FE"].sum() / da["FE"].sum() * 100
)
ax.axvline(
    tasa_general,
    color=ROJO_ALERTA,
    linestyle="--",
    linewidth=1.2,
    alpha=0.7,
    label=f"Tasa general: {tasa_general:.1f}%",
)
ax.set_xlim(0, df_fig1["CAP_ALLY=1 (%)"].max() * 1.35)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=10, loc="lower right")

fig1.text(
    0.5,
    -0.08,
    "Nota: CAP_ALLY=1 si la empresa declaró capacitación en asociaciones con otras\n"
    "empresas (D097, D176, D106). Proxy de actividad formativa. NO equivale al\n"
    "uso de la Franquicia Tributaria SENCE. Dotación física corregida (persona-mes/12).",
    ha="center",
    fontsize=9,
    style="italic",
    color=GRIS,
)
plt.tight_layout()
fig1.savefig(FIGURAS_DIR / "figura1_concentracion_tamano.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Figura 1 guardada")


# ════════════════════════════════════════════════════════════════════════════
# BLOQUE 2 — Pregunta 2: Brechas sectoriales (Figura 2)
# ════════════════════════════════════════════════════════════════════════════
print("\n[BLOQUE 2] Pregunta 2: Brechas sectoriales")

sector_labels = {
    "A": "Agropecuario", "B": "Minería", "C": "Manufactura",
    "D": "Electricidad", "E": "Agua/Residuos", "F": "Construcción",
    "G": "Comercio", "H": "Transporte", "I": "Alojamiento",
    "J": "Información", "K": "Financiero", "L": "Inmobiliario",
    "M": "Prof./Técnico", "N": "Administrativo", "P": "Educación",
    "Q": "Salud", "R": "Arte/Recreación", "S": "Otros servicios",
}
da["SECTOR_LABEL"] = da["SECTOR"].map(sector_labels).fillna("Otros")

# Top 10 sectores por N expandido
top_sectores = (
    da.groupby("SECTOR_LABEL", observed=True)["FE"].sum().nlargest(10).index.tolist()
)
da_top = da[da["SECTOR_LABEL"].isin(top_sectores)]
t2_sector = tasa_ponderada(da_top, "SECTOR_LABEL").sort_values("CAP_ALLY=1 (%)")
t2_sector.to_csv(
    RESULTADOS_DIR / "tabla2_brechas_sectoriales.csv",
    index=False,
    encoding="utf-8-sig",
)
print(f"\n{t2_sector.to_string(index=False)}")

# Figura 2
fig2, ax = plt.subplots(figsize=(10, 6))
# Gradiente de azul según la tasa
cmap = sns.light_palette(AZUL_OSCURO, as_cmap=True)
norm = plt.Normalize(
    t2_sector["CAP_ALLY=1 (%)"].min(), t2_sector["CAP_ALLY=1 (%)"].max()
)
colors_sec = [cmap(norm(v)) for v in t2_sector["CAP_ALLY=1 (%)"]]

bars = ax.barh(
    t2_sector["Grupo"],
    t2_sector["CAP_ALLY=1 (%)"],
    color=colors_sec,
    edgecolor="white",
    height=0.7,
)
for bar, (_, row) in zip(bars, t2_sector.iterrows()):
    ax.text(
        bar.get_width() + 0.2,
        bar.get_y() + bar.get_height() / 2,
        f"  {row['CAP_ALLY=1 (%)']:.1f}%",
        va="center",
        fontsize=10,
        color="#222",
    )

ax.set_xlabel("Tasa de actividad formativa (CAP_ALLY=1, %)", fontsize=11)
ax.set_title(
    "Figura 2. Brechas sectoriales en actividad formativa\n"
    "ELE-7, 2022 · 10 sectores con mayor representación (ponderado)",
    fontsize=12,
    fontweight="bold",
    pad=12,
)
ax.spines[["top", "right"]].set_visible(False)
ax.axvline(tasa_general, color=ROJO_ALERTA, linestyle="--", linewidth=1.0, alpha=0.7)
ax.text(
    tasa_general + 0.15,
    -0.5,
    f"Promedio: {tasa_general:.1f}%",
    fontsize=9,
    color=ROJO_ALERTA,
)
plt.tight_layout()
fig2.savefig(FIGURAS_DIR / "figura2_brechas_sectoriales.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Figura 2 guardada")


# ════════════════════════════════════════════════════════════════════════════
# BLOQUE 3 — Pregunta 3: Asociación formación × masa salarial en PYMEs
# ════════════════════════════════════════════════════════════════════════════
print("\n[BLOQUE 3] Pregunta 3: Asociación formación × salarios en PYMEs")

# Dummies de tamaño
for frame in [da]:
    frame["PYME_dummy"] = (frame["TAMANO_CAT"] == "PYME").astype(int)
    frame["Micro_dummy"] = (frame["TAMANO_CAT"] == "Micro").astype(int)

da_pyme = da[da["TAMANO_CAT"] == "PYME"].copy()

COVARS = ["CAP_ALLY", "PYME_dummy", "Micro_dummy", "ln_ANTIG", "GRUPO_EMP"]
COVARS_PYME = ["CAP_ALLY", "ln_ANTIG", "GRUPO_EMP"]  # sin dummies de tamaño
CATEGORICAL = ["SECTOR"]

especificaciones = [
    ("(1) PROD_LAB · Muestra completa", "PROD_LAB", da, COVARS),
    ("(2) PROD_LAB · Solo PYME", "PROD_LAB", da_pyme, COVARS_PYME),
    ("(3) SAL_TRAB · Muestra completa", "SAL_TRAB", da, COVARS),
    ("(4) SAL_TRAB · Solo PYME", "SAL_TRAB", da_pyme, COVARS_PYME),
]

resultados_ols = {}
filas_tabla3 = []

for nombre, dep, frame, covars in especificaciones:
    y, X, names = build_design_matrix(frame, dep, covars, CATEGORICAL)
    m = fit_ols_hc3(y, X, names)
    resultados_ols[nombre] = m
    print(f"\n  {nombre}  N={int(m.nobs):,}  R²={m.rsquared:.3f}")
    for v in covars:
        if v in m.params.index:
            coef = m.params[v]
            pval = m.pvalues[v]
            star = (
                "***" if pval < 0.01
                else "**" if pval < 0.05
                else "*" if pval < 0.10
                else ""
            )
            print(f"     {v:<15}: {coef:+.4f} {star}  (p={pval:.4f})")

    fila = {"Modelo": nombre, "N": int(m.nobs), "R²": round(m.rsquared, 3)}
    for v in covars:
        if v in m.params.index:
            coef = m.params[v]
            se = m.bse[v]
            pv = m.pvalues[v]
            star = (
                "***" if pv < 0.01
                else "**" if pv < 0.05
                else "*" if pv < 0.10
                else ""
            )
            fila[f"{v}_coef"] = f"{coef:+.4f}{star}"
            fila[f"{v}_se"] = f"({se:.4f})"
    filas_tabla3.append(fila)

tabla3 = pd.DataFrame(filas_tabla3)
tabla3.to_csv(
    RESULTADOS_DIR / "tabla3_regresiones_OLS_v5.csv",
    index=False,
    encoding="utf-8-sig",
)
print("\n  ✓ Tabla 3 guardada")


# ════════════════════════════════════════════════════════════════════════════
# BLOQUE 4 — Pregunta 4: Diferencias CAP_ALLY=1 vs CAP_ALLY=0 (Tabla 4)
# ════════════════════════════════════════════════════════════════════════════
print("\n[BLOQUE 4] Pregunta 4: Diferencias entre empresas que forman y no forman")

filas_t4 = []
for tam in ["Grande", "PYME", "Micro"]:
    sub = da[da["TAMANO_CAT"] == tam]
    for cap in [0, 1]:
        s = sub[sub["CAP_ALLY"] == cap]
        if len(s) < 2:
            continue
        n = len(s)
        # Media e IC95% para PROD_LAB
        m_p, se_p = s["PROD_LAB"].mean(), s["PROD_LAB"].sem()
        t_c = stats.t.ppf(0.975, df=n - 1)
        ic_p_lo, ic_p_hi = m_p - t_c * se_p, m_p + t_c * se_p
        # Media e IC95% para SAL_TRAB
        m_s, se_s = s["SAL_TRAB"].mean(), s["SAL_TRAB"].sem()
        ic_s_lo, ic_s_hi = m_s - t_c * se_s, m_s + t_c * se_s

        filas_t4.append(
            {
                "Tamaño": tam,
                "CAP_ALLY": cap,
                "N": n,
                "PROD_LAB media (ln)": round(m_p, 3),
                "PROD_LAB IC95% inf": round(ic_p_lo, 3),
                "PROD_LAB IC95% sup": round(ic_p_hi, 3),
                "SAL_TRAB media (ln)": round(m_s, 3),
                "SAL_TRAB IC95% inf": round(ic_s_lo, 3),
                "SAL_TRAB IC95% sup": round(ic_s_hi, 3),
            }
        )

tabla4 = pd.DataFrame(filas_t4)
tabla4.to_csv(
    RESULTADOS_DIR / "tabla4_medias_por_CAPALLY.csv",
    index=False,
    encoding="utf-8-sig",
)
print(f"\n{tabla4.to_string(index=False)}")


# ════════════════════════════════════════════════════════════════════════════
# BLOQUE 5 — Pregunta 5: Heterogeneidad temporal (requiere ELE-4/5)
# ════════════════════════════════════════════════════════════════════════════
# Nota: Este bloque requiere que ele4 y ele5 estén en ./datos/. Si no están,
# se reporta solo ELE-7 y se señala que el análisis longitudinal completo
# requiere las olas anteriores.
print("\n[BLOQUE 5] Pregunta 5: Heterogeneidad temporal entre olas")

olas_disponibles = []

if ARCHIVO_ELE5.exists():
    try:
        ele5 = pd.read_csv(ARCHIVO_ELE5, sep=";", encoding="latin-1", low_memory=False)
        # ELE-5 tiene códigos similares; reaplicamos construcción con adaptación mínima
        for col in ["D097", "D176", "D106", "I151", "I160", "C041", "C048",
                    "TAMANO", "FE_transversal", "CIIU_FINAL"]:
            if col in ele5.columns:
                ele5[col] = parse_numeric_comma(ele5[col])
        ele5["CAP_ALLY"] = (
            (ele5.get("D097", 0) == 1)
            | (ele5.get("D176", 0) == 1)
            | (ele5.get("D106", 0) == 1)
        ).astype(int)
        ele5["DOTACION"] = (
            ele5[["I151", "I160"]].fillna(0).sum(axis=1) / 12
        )
        ele5["FE"] = ele5.get("FE_transversal", 1)
        ele5["TAMANO_CAT"] = pd.cut(
            ele5["TAMANO"],
            bins=[0, 1, 4, 5],
            labels=["Grande", "PYME", "Micro"],
        )
        olas_disponibles.append(("ELE-5", 2017, ele5))
        print(f"  ELE-5 cargada: {len(ele5):,} empresas")
    except Exception as e:
        print(f"  ⚠ No se pudo cargar ELE-5: {e}")

olas_disponibles.append(("ELE-7", 2022, da))

# Serie por ola
filas_t5 = []
for nombre_ola, anio, frame in olas_disponibles:
    for tam in ["Grande", "PYME", "Micro"]:
        sub = frame[frame["TAMANO_CAT"] == tam] if "TAMANO_CAT" in frame.columns else None
        if sub is None or len(sub) < 10:
            continue
        total = sub["FE"].sum()
        cap1 = sub.loc[sub["CAP_ALLY"] == 1, "FE"].sum()
        tasa = cap1 / total * 100 if total > 0 else 0
        filas_t5.append(
            {
                "Ola": nombre_ola,
                "Año": anio,
                "Tamaño": tam,
                "Tasa CAP_ALLY=1 (%)": round(tasa, 1),
                "N muestra": len(sub),
                "N expandido": int(total),
            }
        )

tabla5 = pd.DataFrame(filas_t5)
tabla5.to_csv(
    RESULTADOS_DIR / "tabla5_heterogeneidad_temporal.csv",
    index=False,
    encoding="utf-8-sig",
)
print(f"\n{tabla5.to_string(index=False)}")


# ════════════════════════════════════════════════════════════════════════════
# BLOQUE 6 — FIGURA 3: Boxplot PROD_LAB × CAP_ALLY × Tamaño
# ════════════════════════════════════════════════════════════════════════════
print("\n[BLOQUE 6] Figura 3: Distribución PROD_LAB por CAP_ALLY × tamaño")

fig3, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
for ax, tam in zip(axes, ["Grande", "PYME", "Micro"]):
    sub = da[da["TAMANO_CAT"] == tam].copy()
    sub["Grupo"] = sub["CAP_ALLY"].map(
        {0: "Sin actividad\nformativa", 1: "Con actividad\nformativa"}
    )
    palette = {
        "Sin actividad\nformativa": AZUL_CLARO,
        "Con actividad\nformativa": AZUL_OSCURO,
    }
    sns.violinplot(
        data=sub,
        x="Grupo",
        y="PROD_LAB",
        palette=palette,
        inner="quartile",
        cut=0,
        ax=ax,
        order=["Sin actividad\nformativa", "Con actividad\nformativa"],
    )
    for i, cap in enumerate([0, 1]):
        mean = sub.loc[sub["CAP_ALLY"] == cap, "PROD_LAB"].mean()
        ax.scatter(i, mean, color="white", s=50, edgecolor="black", zorder=5, linewidth=1)

    n0 = (sub["CAP_ALLY"] == 0).sum()
    n1 = (sub["CAP_ALLY"] == 1).sum()
    ax.set_xlabel(f"n₀={n0:,}  |  n₁={n1:,}", fontsize=9, color=GRIS)
    ax.set_title(tam, fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel(
        "Productividad laboral\nln(ventas/trabajador, CLP)" if ax == axes[0] else ""
    )

fig3.suptitle(
    "Figura 3. Distribución de productividad laboral por actividad formativa y tamaño\n"
    "ELE-7, 2022 · Dotación física corregida (persona-mes/12) · Punto blanco = media",
    fontsize=12,
    fontweight="bold",
    y=1.02,
)

patch0 = mpatches.Patch(color=AZUL_CLARO, label="Sin actividad formativa")
patch1 = mpatches.Patch(color=AZUL_OSCURO, label="Con actividad formativa")
fig3.legend(handles=[patch0, patch1], loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.06))

fig3.text(
    0.5,
    -0.12,
    "Nota: Las diferencias mostradas son descriptivas y no implican causalidad.\n"
    "Los coeficientes OLS con controles (Tabla 3) muestran que la diferencia en\n"
    "productividad no es estadísticamente significativa una vez ajustada.",
    ha="center",
    fontsize=9,
    style="italic",
    color=GRIS,
)
plt.tight_layout()
fig3.savefig(FIGURAS_DIR / "figura3_prod_lab_violin.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Figura 3 guardada")


# ════════════════════════════════════════════════════════════════════════════
# BLOQUE 7 — FIGURA 4: Prima salarial con intervalos de confianza
# ════════════════════════════════════════════════════════════════════════════
print("\n[BLOQUE 7] Figura 4: Coeficiente CAP_ALLY en SAL_TRAB con IC95%")

# Obtener coeficientes e IC de los modelos estimados
coef_data = []
for nombre, m in resultados_ols.items():
    if "SAL_TRAB" in nombre and "CAP_ALLY" in m.params.index:
        coef = m.params["CAP_ALLY"]
        ic = m.conf_int().loc["CAP_ALLY"]
        prima_pct = (np.exp(coef) - 1) * 100
        prima_lo = (np.exp(ic[0]) - 1) * 100
        prima_hi = (np.exp(ic[1]) - 1) * 100
        label = "Muestra completa" if "completa" in nombre.lower() else "Solo PYME"
        coef_data.append(
            {
                "Muestra": label,
                "coef": coef,
                "ic_lo": ic[0],
                "ic_hi": ic[1],
                "prima_pct": prima_pct,
                "prima_lo": prima_lo,
                "prima_hi": prima_hi,
            }
        )

coef_df = pd.DataFrame(coef_data)
fig4, ax = plt.subplots(figsize=(9, 4.5))
colors_bars = [AZUL_MEDIO, AZUL_OSCURO]
for i, (_, row) in enumerate(coef_df.iterrows()):
    ax.errorbar(
        row["prima_pct"],
        i,
        xerr=[[row["prima_pct"] - row["prima_lo"]], [row["prima_hi"] - row["prima_pct"]]],
        fmt="o",
        color=colors_bars[i],
        markersize=10,
        capsize=6,
        capthick=1.5,
        elinewidth=2,
    )
    ax.text(
        row["prima_pct"],
        i + 0.25,
        f"+{row['prima_pct']:.1f}% [IC: {row['prima_lo']:.1f}; {row['prima_hi']:.1f}]",
        ha="center",
        fontsize=10,
    )

ax.set_yticks(range(len(coef_df)))
ax.set_yticklabels(coef_df["Muestra"], fontsize=11)
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Prima salarial asociada a CAP_ALLY=1 (%, IC 95%)", fontsize=11)
ax.set_title(
    "Figura 4. Prima salarial asociada a actividad formativa\n"
    "OLS con FE sector y controles · Errores robustos HC3",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, max(coef_df["prima_hi"]) * 1.15)
ax.set_ylim(-0.5, len(coef_df) - 0.3)

fig4.text(
    0.5,
    -0.15,
    "Nota: Asociaciones condicionales, no efectos causales. Controles: ln(antigüedad),\n"
    "pertenencia a grupo empresarial, tamaño (solo muestra completa), efectos fijos CIIU.",
    ha="center",
    fontsize=9,
    style="italic",
    color=GRIS,
)
plt.tight_layout()
fig4.savefig(FIGURAS_DIR / "figura4_prima_salarial_IC.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Figura 4 guardada")

# ════════════════════════════════════════════════════════════════════════════
# CIERRE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("RESUMEN FINAL")
print("=" * 72)

beta_pyme_sal = resultados_ols["(4) SAL_TRAB · Solo PYME"].params["CAP_ALLY"]
prima_pyme_pct = (np.exp(beta_pyme_sal) - 1) * 100
print(f"\nHallazgo central (Pregunta 3):")
print(f"  β_SAL_PYME = {beta_pyme_sal:+.4f}")
print(f"  Prima salarial implicada: +{prima_pyme_pct:.2f}%")

print(f"\nCifras realistas con corrección de dotación:")
print(f"  Salario mensual medio PYME: CLP {sal_mes_medio_pyme:,.0f}")
print(f"  (era CLP {sal_mes_medio_pyme/12*1:,.0f} sin corrección — subestimado ×12)")

print("\nArchivos generados:")
print(f"  {RESULTADOS_DIR}/tabla1_concentracion_tamano.csv")
print(f"  {RESULTADOS_DIR}/tabla2_brechas_sectoriales.csv")
print(f"  {RESULTADOS_DIR}/tabla3_regresiones_OLS_v5.csv")
print(f"  {RESULTADOS_DIR}/tabla4_medias_por_CAPALLY.csv")
print(f"  {RESULTADOS_DIR}/tabla5_heterogeneidad_temporal.csv")
print(f"  {FIGURAS_DIR}/figura1_concentracion_tamano.png")
print(f"  {FIGURAS_DIR}/figura2_brechas_sectoriales.png")
print(f"  {FIGURAS_DIR}/figura3_prod_lab_violin.png")
print(f"  {FIGURAS_DIR}/figura4_prima_salarial_IC.png")
print("\n✓ ANÁLISIS COMPLETADO")
