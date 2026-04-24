"""
Verificación independiente de la regresión de feminización
===========================================================
Reconstruye el panel ELE-6 × ELE-7 (3,150 empresas) y replica la regresión
OLS con HC3 reportada en el manuscrito académico (coeficientes esperados:
β_FEMINIZACION = -0.301 en muestra completa, -0.259 en PYMEs).

Incluye tres chequeos de robustez:
  1. Momento de medición de FEMINIZACION (2019 vs 2022 vs promedio)
  2. VIF entre FEMINIZACION y efectos fijos de sector
  3. Especificación alternativa con terciles de feminización
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

UPLOADS = Path("/mnt/user-data/uploads")
OUTPUTS = Path("/mnt/user-data/outputs")
OUTPUTS.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ════════════════════════════════════════════════════════════════════════════
def parse_comma(s):
    """Convierte strings con coma decimal chilena a float."""
    return pd.to_numeric(
        pd.Series(s).astype(str).str.replace(",", ".").replace("nan", np.nan),
        errors="coerce",
    )


class OLSHC3:
    """OLS con errores robustos HC3 (MacKinnon-White 1985), en NumPy puro."""

    def __init__(self, y, X, names):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, k = X.shape
        XtX = X.T @ X
        XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ X.T @ y
        resid = y - X @ beta
        tss = np.sum((y - y.mean()) ** 2)
        rss = np.sum(resid ** 2)
        self.rsquared = 1 - rss / tss if tss > 0 else 0.0
        H = X @ XtX_inv @ X.T
        h = np.clip(np.diag(H), 0, 0.999999)
        u2 = (resid / (1 - h)) ** 2
        meat = X.T @ np.diag(u2) @ X
        vcov = XtX_inv @ meat @ XtX_inv
        se = np.sqrt(np.diag(vcov))
        tvals = beta / se
        df_res = n - k
        pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), df=df_res))
        tcrit = stats.t.ppf(0.975, df=df_res)
        self.params = pd.Series(beta, index=names)
        self.bse = pd.Series(se, index=names)
        self.pvalues = pd.Series(pvals, index=names)
        self.ci_lo = pd.Series(beta - tcrit * se, index=names)
        self.ci_hi = pd.Series(beta + tcrit * se, index=names)
        self.nobs = int(n)
        self.k = int(k)


def design_matrix(df, dep, covars, cats=None):
    """Construye X con intercepto, covariables numéricas y dummies categóricas."""
    cats = cats or []
    cols = [dep] + list(covars) + list(cats)
    clean = df[cols].dropna().copy()
    # Eliminar categorías con <5 observaciones
    for c in cats:
        vc = clean[c].value_counts()
        clean = clean[clean[c].isin(vc[vc >= 5].index)]
    if len(clean) == 0:
        return None, None, None
    X_parts = [np.ones((len(clean), 1)), clean[covars].astype(float).values]
    names = ["const"] + list(covars)
    for c in cats:
        d = pd.get_dummies(clean[c], prefix=c, drop_first=True).astype(float)
        X_parts.append(d.values)
        names.extend(d.columns.tolist())
    return clean[dep].astype(float).values, np.hstack(X_parts), names


def clasificar_tamano(x):
    if pd.isna(x): return np.nan
    if x == 1: return "Grande"
    if x in (2, 3, 4): return "PYME"
    if x == 5: return "Micro"
    return np.nan


def print_coef_table(m, title, vars_highlight=None):
    """Imprime tabla de coeficientes estilo paper."""
    vars_highlight = vars_highlight or []
    print(f"\n{title}")
    print(f"  N = {m.nobs:,}  |  R² = {m.rsquared:.3f}  |  k = {m.k}")
    print(f"  {'Variable':<25} {'Coef.':>10} {'SE':>10} {'p-value':>10} {'Prima %':>10}")
    print("  " + "-" * 70)
    for v in m.params.index:
        if v == "const" or v.startswith("SECTOR_"):
            continue
        coef = m.params[v]
        se = m.bse[v]
        pv = m.pvalues[v]
        star = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""
        prima = (np.exp(coef) - 1) * 100
        highlight = "◀" if any(h in v for h in vars_highlight) else ""
        print(f"  {v:<25} {coef:>+10.4f} {se:>10.4f} {pv:>10.4f} {prima:>+9.2f}% {star:<3} {highlight}")


# ════════════════════════════════════════════════════════════════════════════
# CARGA: ELE-6 (año 2019)
# ════════════════════════════════════════════════════════════════════════════
print("=" * 78)
print("CARGANDO ELE-6 (olas 2019)")
print("=" * 78)

c1 = pd.read_csv(UPLOADS / "c1-caracterizacio_n-de-la-empresa84293ab7883447c0831c03d7f61fac81.csv",
                  sep=";", encoding="latin-1", low_memory=False)
c2 = pd.read_csv(UPLOADS / "c2-contabilidad-y-finanzascbdd693f84224c27ac7825aeee8367ac.csv",
                  sep=";", encoding="latin-1", low_memory=False)
c3 = pd.read_csv(UPLOADS / "c3-recursos-humanosbe8bad3ce0ae4f7788d9c6ef392c42cf.csv",
                  sep=";", encoding="latin-1", low_memory=False)
c4 = pd.read_csv(UPLOADS / "c4-mercados-clientes-y-proveedores.csv",
                  sep=";", encoding="latin-1", low_memory=False)

print(f"  c1 (caracterización):      {len(c1):,} × {c1.shape[1]}")
print(f"  c2 (contabilidad):         {len(c2):,} × {c2.shape[1]}")
print(f"  c3 (recursos humanos):     {len(c3):,} × {c3.shape[1]}")
print(f"  c4 (mercados/asociación):  {len(c4):,} × {c4.shape[1]}")

# Merge en ROL_ficticio (estable en ELE-6)
e6 = c1[["ROL_ficticio", "Tamano", "CIIU_FINAL", "FE_TRANSVERSAL", "A068", "A069"]].copy()
e6 = e6.merge(c2[["ROL_ficticio", "C041", "C048"]], on="ROL_ficticio", how="left")
e6 = e6.merge(c3[["ROL_ficticio", "I151", "I160"]], on="ROL_ficticio", how="left")
e6 = e6.merge(c4[["ROL_ficticio", "D097", "D176", "D106"]], on="ROL_ficticio", how="left")

# Parseo numérico
for col in ["Tamano", "FE_TRANSVERSAL", "A068", "A069", "C041", "C048",
            "I151", "I160", "D097", "D176", "D106"]:
    e6[col] = parse_comma(e6[col])

print(f"\n  ELE-6 completo:  {len(e6):,} empresas")
print(f"  N declarado en manuscrito: 4,478")


# ════════════════════════════════════════════════════════════════════════════
# CARGA: ELE-7 (año 2022)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("CARGANDO ELE-7 (ola 2022)")
print("=" * 78)

e7 = pd.read_csv(UPLOADS / "ele7-full.csv", sep=";", encoding="latin-1", low_memory=False)
# Nota: en ELE-7 la variable es ROL_FICTICIO (mayúsculas)
cols_e7 = ["ROL_FICTICIO", "TAMANO", "CIIU_FINAL", "FE_TRANSVERSAL",
           "A068", "A069", "C077", "C084", "I151", "I160",
           "D097", "D176", "D106"]
cols_presentes = [c for c in cols_e7 if c in e7.columns]
e7 = e7[cols_presentes].copy()

for col in ["TAMANO", "FE_TRANSVERSAL", "A068", "A069", "C077", "C084",
            "I151", "I160", "D097", "D176", "D106"]:
    if col in e7.columns:
        e7[col] = parse_comma(e7[col])

e7 = e7.rename(columns={"ROL_FICTICIO": "ROL_ficticio"})
print(f"  ELE-7 cargado:  {len(e7):,} empresas")
print(f"  N declarado en manuscrito: 6,592")


# ════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DEL PANEL BALANCEADO 2019 × 2022
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("CONSTRUCCIÓN DEL PANEL BALANCEADO (match por ROL_ficticio)")
print("=" * 78)

# Asegurar tipo consistente del identificador
e6["ROL_ficticio"] = e6["ROL_ficticio"].astype(str).str.strip()
e7["ROL_ficticio"] = e7["ROL_ficticio"].astype(str).str.strip()

# Ids comunes
ids_e6 = set(e6["ROL_ficticio"])
ids_e7 = set(e7["ROL_ficticio"])
ids_comunes = ids_e6 & ids_e7
print(f"  IDs únicos ELE-6:     {len(ids_e6):,}")
print(f"  IDs únicos ELE-7:     {len(ids_e7):,}")
print(f"  Intersección (panel): {len(ids_comunes):,}")
print(f"  N declarado en manuscrito: 3,150")

# Construir panel ancho: una fila por empresa con variables _19 y _22
e6_panel = e6[e6["ROL_ficticio"].isin(ids_comunes)].copy()
e7_panel = e7[e7["ROL_ficticio"].isin(ids_comunes)].copy()

# Renombrar para sufijar
rename_e6 = {c: f"{c.upper()}_19" for c in e6_panel.columns if c != "ROL_ficticio"}
rename_e7 = {c: f"{c.upper()}_22" for c in e7_panel.columns if c != "ROL_ficticio"}
e6_panel = e6_panel.rename(columns=rename_e6)
e7_panel = e7_panel.rename(columns=rename_e7)

panel = e6_panel.merge(e7_panel, on="ROL_ficticio", how="inner")
panel = panel.drop_duplicates(subset="ROL_ficticio")
print(f"  Panel final (deduplicado): {len(panel):,} empresas")


# ════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DE VARIABLES
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("CONSTRUCCIÓN DE VARIABLES")
print("=" * 78)

# --- CAP_ALLY en cada ola (NaN → 0) ---
for suf in ["19", "22"]:
    d097 = panel[f"D097_{suf}"].fillna(0)
    d176 = panel[f"D176_{suf}"].fillna(0)
    d106 = panel[f"D106_{suf}"].fillna(0)
    panel[f"CAP_ALLY_{suf}"] = ((d097 == 1) | (d176 == 1) | (d106 == 1)).astype(int)

# --- Trayectoria ---
def trayectoria(row):
    a, b = row["CAP_ALLY_19"], row["CAP_ALLY_22"]
    if a == 0 and b == 0: return "Nunca"
    if a == 1 and b == 1: return "Siempre"
    if a == 0 and b == 1: return "Adoptadora"
    if a == 1 and b == 0: return "Desertora"
    return np.nan

panel["TRAYECTORIA"] = panel.apply(trayectoria, axis=1)

print("\n  Distribución de trayectorias:")
print(panel["TRAYECTORIA"].value_counts(dropna=False).to_string())
print(f"\n  Ratio deserción/adopción = {(panel['TRAYECTORIA']=='Desertora').sum() / max(1,(panel['TRAYECTORIA']=='Adoptadora').sum()):.2f}")

# --- Dotación física y feminización en cada ola ---
for suf in ["19", "22"]:
    i151 = panel[f"I151_{suf}"].fillna(0)  # hombres (persona-meses)
    i160 = panel[f"I160_{suf}"].fillna(0)  # mujeres (persona-meses)
    total = i151 + i160
    panel[f"DOTACION_{suf}"] = total / 12  # promedio anual
    panel[f"FEMINIZ_{suf}"] = np.where(total > 0, i160 / total, np.nan)

# Feminización en tres operacionalizaciones
panel["FEMINIZ_promedio"] = (panel["FEMINIZ_19"] + panel["FEMINIZ_22"]) / 2

# --- Salario 2022 ---
# En ELE-7: C084 = remuneraciones totales (miles CLP); C077 = ventas
mask_sal_22 = (panel["C084_22"] > 0) & (panel["DOTACION_22"] > 0)
panel["SAL_TRAB_22"] = np.where(
    mask_sal_22,
    np.log(panel["C084_22"] * 1000 / panel["DOTACION_22"]),
    np.nan
)

# --- Categoría de tamaño (usando el tamaño en 2022, típico del manuscrito) ---
panel["TAMANO_CAT"] = panel["TAMANO_22"].apply(clasificar_tamano)

# --- Controles ---
panel["ANTIG"] = (2022 - panel["A068_22"]).clip(lower=0)
panel["ln_ANTIG"] = np.log1p(panel["ANTIG"])
panel["GRUPO_EMP"] = (panel["A069_22"] == 1).astype(int)
panel["SECTOR"] = panel["CIIU_FINAL_22"].astype(str).str.strip().str[0]

# Dummies de tamaño y trayectoria
panel["PYME_d"] = (panel["TAMANO_CAT"] == "PYME").astype(int)
panel["Micro_d"] = (panel["TAMANO_CAT"] == "Micro").astype(int)

for t in ["Siempre", "Adoptadora", "Desertora"]:  # Nunca = categoría de referencia
    panel[f"TRAY_{t}"] = (panel["TRAYECTORIA"] == t).astype(int)


# ════════════════════════════════════════════════════════════════════════════
# REGRESIÓN PRINCIPAL: MUESTRA COMPLETA Y PYME
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("REGRESIÓN PRINCIPAL (replicando especificación del manuscrito)")
print("=" * 78)

# Sin trim (replicando exactamente la especificación del manuscrito)
panel_reg = panel.copy()
print(f"\n  Sin trim aplicado (replicando manuscrito)")
print(f"  Panel para regresión: {len(panel_reg):,}")


# ────────────────────────────────────────────────────────────────────────
# CHEQUEO 1: ¿En qué año se mide FEMINIZACION?
# ────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 78)
print("CHEQUEO 1 — Momento de medición de FEMINIZACION")
print("─" * 78)
print("\n  Comparación de coeficientes con FEMINIZ medida en 3 momentos distintos:")

resultados_chequeo1 = []
for label, fem_var in [("FEMINIZ_19", "FEMINIZ_19"),
                        ("FEMINIZ_22", "FEMINIZ_22"),
                        ("FEMINIZ_promedio", "FEMINIZ_promedio")]:
    covars = ["TRAY_Siempre", "TRAY_Adoptadora", "TRAY_Desertora",
              "PYME_d", "Micro_d", "ln_ANTIG", "GRUPO_EMP", fem_var]
    y, X, names = design_matrix(panel_reg, "SAL_TRAB_22", covars, ["SECTOR"])
    if y is None:
        continue
    m = OLSHC3(y, X, names)
    beta_fem = m.params[fem_var]
    prima_fem = (np.exp(beta_fem) - 1) * 100
    pval_fem = m.pvalues[fem_var]
    resultados_chequeo1.append({
        "Variante": label,
        "N": m.nobs,
        "R²": round(m.rsquared, 3),
        "β_FEMINIZ": round(beta_fem, 4),
        "SE": round(m.bse[fem_var], 4),
        "p-value": round(pval_fem, 4),
        "Prima %": round(prima_fem, 2),
    })

df_chk1 = pd.DataFrame(resultados_chequeo1)
print(f"\n{df_chk1.to_string(index=False)}")

# Elegir la que mejor replica el manuscrito (-0.301)
mejor_match = df_chk1.iloc[(df_chk1["β_FEMINIZ"] - (-0.301)).abs().argsort()[0]]
print(f"\n  La variante más cercana al coeficiente del manuscrito (-0.301) es:")
print(f"  → {mejor_match['Variante']}  (β = {mejor_match['β_FEMINIZ']}, prima = {mejor_match['Prima %']}%)")
fem_elegido = mejor_match["Variante"]


# ────────────────────────────────────────────────────────────────────────
# REGRESIÓN PRINCIPAL con la variante elegida
# ────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 78)
print(f"MODELO 1 — Muestra completa  (FEMINIZ = {fem_elegido})")
print("─" * 78)

covars_full = ["TRAY_Siempre", "TRAY_Adoptadora", "TRAY_Desertora",
                "PYME_d", "Micro_d", "ln_ANTIG", "GRUPO_EMP", fem_elegido]
y, X, names = design_matrix(panel_reg, "SAL_TRAB_22", covars_full, ["SECTOR"])
m_full = OLSHC3(y, X, names)
print_coef_table(m_full, "",
                  vars_highlight=["FEMINIZ", "TRAY_"])

print("\n" + "─" * 78)
print(f"MODELO 2 — Solo PYMEs  (FEMINIZ = {fem_elegido})")
print("─" * 78)

panel_pyme = panel_reg[panel_reg["TAMANO_CAT"] == "PYME"].copy()
covars_pyme = ["TRAY_Siempre", "TRAY_Adoptadora", "TRAY_Desertora",
                "ln_ANTIG", "GRUPO_EMP", fem_elegido]
y, X, names = design_matrix(panel_pyme, "SAL_TRAB_22", covars_pyme, ["SECTOR"])
m_pyme = OLSHC3(y, X, names)
print_coef_table(m_pyme, "",
                  vars_highlight=["FEMINIZ", "TRAY_"])

# Comparar con lo reportado en el manuscrito
print("\n" + "=" * 78)
print("COMPARACIÓN CON MANUSCRITO")
print("=" * 78)
print(f"\n  {'':<20}  {'Manuscrito':>15}  {'Replicado':>15}  {'Diferencia':>15}")
print("  " + "-" * 75)

# Muestra completa
beta_full_rep = m_full.params[fem_elegido]
beta_full_man = -0.301
print(f"  {'β_FEMIN completa':<20}  {beta_full_man:>+15.4f}  {beta_full_rep:>+15.4f}  {beta_full_rep-beta_full_man:>+15.4f}")
print(f"  {'N completa':<20}  {'2,690':>15}  {m_full.nobs:>15,}  {m_full.nobs-2690:>+15,}")

# PYME
beta_pyme_rep = m_pyme.params[fem_elegido]
beta_pyme_man = -0.259
print(f"  {'β_FEMIN PYME':<20}  {beta_pyme_man:>+15.4f}  {beta_pyme_rep:>+15.4f}  {beta_pyme_rep-beta_pyme_man:>+15.4f}")
print(f"  {'N PYME':<20}  {'1,183':>15}  {m_pyme.nobs:>15,}  {m_pyme.nobs-1183:>+15,}")


# ════════════════════════════════════════════════════════════════════════════
# CHEQUEO 2: Multicolinealidad entre FEMINIZACION y efectos fijos de sector
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("CHEQUEO 2 — ¿Cuánta variación de FEMINIZACION absorben los FE de sector?")
print("=" * 78)
print("\n  R² de una regresión auxiliar de FEMINIZACION ~ dummies_sector")

panel_aux = panel_reg.dropna(subset=[fem_elegido, "SECTOR"]).copy()
fem_var = panel_aux[fem_elegido].values

# Dummies de sector
sec_dummies = pd.get_dummies(panel_aux["SECTOR"], drop_first=True).astype(float).values
X_aux = np.hstack([np.ones((len(panel_aux), 1)), sec_dummies])
m_aux = OLSHC3(fem_var, X_aux, ["const"] + [f"d{i}" for i in range(sec_dummies.shape[1])])
print(f"  R² aux = {m_aux.rsquared:.3f}")
if m_aux.rsquared > 0.5:
    print(f"  ⚠ Más del 50% de la variación de FEMINIZ se explica por sector.")
    print(f"    El coeficiente condicional captura variación INTRASECTORIAL de feminización.")
else:
    print(f"  ✓ La variación intrasectorial de FEMINIZ es sustancial ({(1-m_aux.rsquared)*100:.0f}%).")
    print(f"    El efecto fijo de sector no está absorbiendo la mayoría del efecto.")

# Feminización media por sector
print("\n  Feminización media por sector:")
fem_sec = panel_aux.groupby("SECTOR")[fem_elegido].agg(["mean", "std", "count"]).round(3)
fem_sec.columns = ["media", "sd", "n"]
fem_sec = fem_sec.sort_values("media", ascending=False)
print(fem_sec.to_string())


# ════════════════════════════════════════════════════════════════════════════
# CHEQUEO 3: Especificación con terciles de feminización (dummies)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("CHEQUEO 3 — Especificación alternativa: FEMINIZACION en terciles (dummies)")
print("=" * 78)

# Terciles según valores muestrales
q1, q2 = panel_reg[fem_elegido].quantile([1/3, 2/3])
print(f"\n  Cortes de terciles de {fem_elegido}:")
print(f"    Tercil 1 (Masculinizada):   {fem_elegido} <= {q1:.3f}")
print(f"    Tercil 2 (Equilibrada):    {q1:.3f} < {fem_elegido} <= {q2:.3f}")
print(f"    Tercil 3 (Feminizada):     {fem_elegido} > {q2:.3f}")
print(f"  (El manuscrito declara cortes en 17.7% y 42.1%)")

panel_reg["FEMIN_TER"] = pd.cut(
    panel_reg[fem_elegido],
    bins=[-0.001, q1, q2, 1.001],
    labels=["Masculinizada", "Equilibrada", "Feminizada"]
)
# Dummies (Masculinizada = referencia)
panel_reg["FEMIN_equi"] = (panel_reg["FEMIN_TER"] == "Equilibrada").astype(int)
panel_reg["FEMIN_femi"] = (panel_reg["FEMIN_TER"] == "Feminizada").astype(int)

covars_ter = ["TRAY_Siempre", "TRAY_Adoptadora", "TRAY_Desertora",
               "PYME_d", "Micro_d", "ln_ANTIG", "GRUPO_EMP",
               "FEMIN_equi", "FEMIN_femi"]
y, X, names = design_matrix(panel_reg, "SAL_TRAB_22", covars_ter, ["SECTOR"])
m_ter = OLSHC3(y, X, names)
print_coef_table(m_ter, "\n  Modelo con terciles de feminización:",
                  vars_highlight=["FEMIN_"])

# Interpretación
beta_equi = m_ter.params["FEMIN_equi"]
beta_femi = m_ter.params["FEMIN_femi"]
print(f"\n  Interpretación:")
print(f"    Empresa equilibrada vs masculinizada: prima = {(np.exp(beta_equi)-1)*100:+.1f}%  (p={m_ter.pvalues['FEMIN_equi']:.3f})")
print(f"    Empresa feminizada vs masculinizada:  prima = {(np.exp(beta_femi)-1)*100:+.1f}%  (p={m_ter.pvalues['FEMIN_femi']:.3f})")
print(f"\n  Monotonicidad: ", end="")
if beta_femi < beta_equi < 0:
    print("✓ Se cumple (más feminización → salario más bajo)")
elif beta_femi < beta_equi:
    print("✓ Parcial (feminizada < equilibrada, pero equi ≥ 0)")
else:
    print("⚠ No se cumple — patrón no monotónico")


# ════════════════════════════════════════════════════════════════════════════
# GUARDAR RESULTADOS
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 78)
print("GUARDANDO RESULTADOS")
print("=" * 78)

# Tabla principal comparativa
resumen = pd.DataFrame([
    {
        "Modelo": "Muestra completa",
        "N manuscrito": 2690,
        "N replicado": m_full.nobs,
        "β_FEMIN manuscrito": -0.301,
        "β_FEMIN replicado": round(m_full.params[fem_elegido], 4),
        "SE replicado": round(m_full.bse[fem_elegido], 4),
        "p-value replicado": round(m_full.pvalues[fem_elegido], 4),
        "Prima % manuscrito": -26.0,
        "Prima % replicado": round((np.exp(m_full.params[fem_elegido])-1)*100, 2),
        "R² replicado": round(m_full.rsquared, 3),
    },
    {
        "Modelo": "Solo PYMEs",
        "N manuscrito": 1183,
        "N replicado": m_pyme.nobs,
        "β_FEMIN manuscrito": -0.259,
        "β_FEMIN replicado": round(m_pyme.params[fem_elegido], 4),
        "SE replicado": round(m_pyme.bse[fem_elegido], 4),
        "p-value replicado": round(m_pyme.pvalues[fem_elegido], 4),
        "Prima % manuscrito": -22.8,
        "Prima % replicado": round((np.exp(m_pyme.params[fem_elegido])-1)*100, 2),
        "R² replicado": round(m_pyme.rsquared, 3),
    },
])
resumen.to_csv(OUTPUTS / "verificacion_regresion_feminizacion.csv", index=False, encoding="utf-8-sig")
print(f"  ✓ {OUTPUTS / 'verificacion_regresion_feminizacion.csv'}")

df_chk1.to_csv(OUTPUTS / "chequeo1_momento_medicion_feminizacion.csv", index=False, encoding="utf-8-sig")
print(f"  ✓ {OUTPUTS / 'chequeo1_momento_medicion_feminizacion.csv'}")

print("\n✓ VERIFICACIÓN COMPLETADA")
