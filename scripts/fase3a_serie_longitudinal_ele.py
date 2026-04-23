"""
OPPRU-Chile · Piloto 2 · Análisis longitudinal v5.1
=====================================================
Revalida todas las cifras de la columna de opinión y del informe Piloto 2
corriendo la misma construcción de variables (con correcciones v5) en las
cuatro olas disponibles:

  - ELE-4 (año de referencia ≈ 2013, levantada 2015)
  - ELE-5 (año de referencia ≈ 2015, levantada 2017)
  - ELE-6 (año de referencia ≈ 2019, levantada 2020)
  - ELE-7 (año de referencia = 2022, levantada 2023)

Objetivos
---------
  1. Tasa general de actividad formativa (CAP_ALLY=1) ponderada por ola.
  2. Tasa por tamaño (Grande / PYME / Micro) por ola → ratio Grande/PYME.
  3. Coeficiente CAP_ALLY en SAL_TRAB (muestra completa y PYME) por ola.
  4. Tabla y figura longitudinales consolidadas.

Correcciones v5 aplicadas a TODAS las olas
-------------------------------------------
  • Parsing coma decimal en FE_TRANSVERSAL y variables numéricas.
  • Dotación física = (I151 + I160) / 12 cuando están disponibles
    (ELE-5, ELE-6, ELE-7). En ELE-4 sólo hay I020 (total agregado) y
    se aplica la misma división por 12.
  • OLS con errores robustos HC3 (numpy puro, sin statsmodels).

Autor: Felipe Ponce Bollmann · UCM · abril 2026
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATOS = ROOT / "datos"
RES = ROOT / "resultados"
FIG = ROOT / "figuras"
for d in [RES, FIG]:
    d.mkdir(parents=True, exist_ok=True)


def parse_comma(s):
    return pd.to_numeric(
        pd.Series(s).astype(str).str.replace(",", ".").replace("nan", np.nan),
        errors="coerce",
    )


class OLSHC3:
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


def design(data, dep, covars, cats=None):
    cats = cats or []
    df = data[[dep] + covars + cats].dropna().copy()
    for c in cats:
        vc = df[c].value_counts()
        df = df[df[c].isin(vc[vc >= 5].index)]
    X_parts = [np.ones((len(df), 1)), df[covars].astype(float).values]
    names = ["const"] + list(covars)
    for c in cats:
        d = pd.get_dummies(df[c], prefix=c, drop_first=True).astype(float)
        X_parts.append(d.values)
        names.extend(d.columns.tolist())
    return df[dep].astype(float).values, np.hstack(X_parts), names


def clasificar_tamano(x):
    if pd.isna(x):
        return np.nan
    if x == 1: return "Grande"
    if x in (2, 3, 4): return "PYME"
    if x == 5: return "Micro"
    return np.nan


def tasa_ponderada(frame):
    out = {}
    total_fe = frame["FE"].sum()
    for tam in ["Grande", "PYME", "Micro"]:
        sub = frame[frame["TAMANO_CAT"] == tam]
        tot_t = sub["FE"].sum()
        cap1 = sub.loc[sub["CAP_ALLY"] == 1, "FE"].sum()
        out[tam] = {
            "N_muestra": len(sub),
            "N_expandido": int(tot_t),
            "tasa_pct": round(cap1 / tot_t * 100, 2) if tot_t > 0 else np.nan,
        }
    cap1_tot = frame.loc[frame["CAP_ALLY"] == 1, "FE"].sum()
    out["General"] = {
        "N_muestra": len(frame),
        "N_expandido": int(total_fe),
        "tasa_pct": round(cap1_tot / total_fe * 100, 2),
    }
    return out


def cargar_ele4():
    """ELE-4 (ref 2013, levantada 2015): sufijo _ELE4, separador coma.
    Nota: D176 no existe en ELE-4; CAP_ALLY se construye con D097 y D106.
    Tamaño viene como texto: 'Grande', 'Mediana', 'Pequeña 1', 'Pequeña 2', 'Micro'."""
    df = pd.read_csv(DATOS / "BBDD_ELE4.csv", sep=",", encoding="latin-1", low_memory=False)

    def get(col):
        c = f"{col}_ELE4"
        return df[c] if c in df.columns else pd.Series([np.nan] * len(df))

    out = pd.DataFrame()
    out["D097"] = parse_comma(get("D097"))
    out["D106"] = parse_comma(get("D106"))
    out["I020"] = parse_comma(get("I020"))
    out["C041"] = parse_comma(get("C041"))
    out["C048"] = parse_comma(get("C048"))
    out["A068"] = parse_comma(get("A068"))
    out["A069"] = parse_comma(get("A069"))
    out["CIIU_FINAL"] = get("CIIUfinal").astype(str)
    out["FE"] = parse_comma(get("FE_Empresas"))

    # TAMANO categórico textual en ELE-4
    tam_texto = get("Tamaño").astype(str).str.strip()
    def mapear_tam_ele4(t):
        t = str(t).strip()
        if t == "Grande": return "Grande"
        if t in ("Mediana", "Pequeña 1", "Pequeña 2"): return "PYME"
        if t == "Micro": return "Micro"
        return np.nan
    out["TAMANO_CAT"] = tam_texto.apply(mapear_tam_ele4)

    # CAP_ALLY: NaN se trata como 0
    out["CAP_ALLY"] = ((out["D097"].fillna(0) == 1) | (out["D106"].fillna(0) == 1)).astype(int)

    # Dotación: ELE-4 solo tiene I020 (total persona-meses) → dividir por 12
    out["DOTACION"] = out["I020"] / 12

    valid_prod = (out["C041"] > 0) & (out["DOTACION"] > 0)
    out["PROD_LAB"] = np.where(valid_prod, np.log(out["C041"] * 1000 / out["DOTACION"]), np.nan)
    valid_sal = (out["C048"] > 0) & (out["DOTACION"] > 0)
    out["SAL_TRAB"] = np.where(valid_sal, np.log(out["C048"] * 1000 / out["DOTACION"]), np.nan)

    out["ANTIG"] = (2013 - out["A068"]).clip(lower=0)
    out["ln_ANTIG"] = np.log1p(out["ANTIG"])
    out["GRUPO_EMP"] = (out["A069"] == 1).astype(int)
    out["SECTOR"] = out["CIIU_FINAL"].astype(str).str.strip().str[0]
    out["OLA"] = "ELE-4"
    out["FE"] = out["FE"].fillna(1)
    return out


def cargar_ele5():
    """ELE-5 (ref 2015, levantada 2017): separador COMA."""
    df = pd.read_csv(DATOS / "BBDD-ELE5-Formato-Texto.csv", sep=",",
                     encoding="latin-1", low_memory=False)

    out = pd.DataFrame()
    for col in ["D097", "D176", "D106", "I151", "I160", "I020",
                "C041", "C048", "TAMANO", "A068", "A069"]:
        if col in df.columns:
            out[col] = parse_comma(df[col])
        else:
            out[col] = np.nan

    for fe_col in ["FE_transversal", "FE_Transversal", "FE_TRANSVERSAL",
                   "FE_Empresas", "FactorExp", "FE"]:
        if fe_col in df.columns:
            out["FE"] = parse_comma(df[fe_col])
            break
    else:
        out["FE"] = 1

    for tam_col in ["Tamano", "TAMANO", "Tamaño"]:
        if tam_col in df.columns:
            out["TAMANO"] = parse_comma(df[tam_col])
            break

    for ciiu_col in ["CIIU_FINAL", "CIIU_final", "CIIUfinal"]:
        if ciiu_col in df.columns:
            out["CIIU_FINAL"] = df[ciiu_col].astype(str)
            break
    else:
        out["CIIU_FINAL"] = ""

    # CAP_ALLY: NaN se trata como 0
    d097 = out["D097"].fillna(0) if "D097" in out.columns else 0
    d176 = out["D176"].fillna(0) if "D176" in out.columns else 0
    d106 = out["D106"].fillna(0) if "D106" in out.columns else 0
    out["CAP_ALLY"] = ((d097 == 1) | (d176 == 1) | (d106 == 1)).astype(int)

    dot = out["I151"].fillna(0) + out["I160"].fillna(0)
    use_i020 = dot == 0
    dot = dot.where(~use_i020, out["I020"].fillna(0))
    out["DOTACION"] = dot / 12

    valid_prod = (out["C041"] > 0) & (out["DOTACION"] > 0)
    out["PROD_LAB"] = np.where(valid_prod, np.log(out["C041"] * 1000 / out["DOTACION"]), np.nan)
    valid_sal = (out["C048"] > 0) & (out["DOTACION"] > 0)
    out["SAL_TRAB"] = np.where(valid_sal, np.log(out["C048"] * 1000 / out["DOTACION"]), np.nan)

    out["TAMANO_CAT"] = out["TAMANO"].apply(clasificar_tamano)
    out["ANTIG"] = (2015 - out["A068"]).clip(lower=0)
    out["ln_ANTIG"] = np.log1p(out["ANTIG"])
    out["GRUPO_EMP"] = (out["A069"] == 1).astype(int)
    out["SECTOR"] = out["CIIU_FINAL"].astype(str).str.strip().str[0]
    out["OLA"] = "ELE-5"
    out["FE"] = out["FE"].fillna(1)
    return out


def cargar_ele6():
    df = pd.read_csv(DATOS / "ele6_consolidada.csv", sep=";",
                     encoding="latin-1", low_memory=False)

    out = pd.DataFrame()
    for col in ["D097", "D176", "D106", "I151", "I160", "I020",
                "C041", "C048", "A068", "A069"]:
        if col in df.columns:
            out[col] = parse_comma(df[col])
        else:
            out[col] = np.nan

    out["TAMANO"] = parse_comma(df["Tamano"]) if "Tamano" in df.columns else np.nan
    out["FE"] = parse_comma(df["FE_TRANSVERSAL"])
    out["CIIU_FINAL"] = df["CIIU_FINAL"].astype(str)

    # CAP_ALLY: NaN se trata como 0
    d097 = out["D097"].fillna(0)
    d176 = out["D176"].fillna(0)
    d106 = out["D106"].fillna(0)
    out["CAP_ALLY"] = ((d097 == 1) | (d176 == 1) | (d106 == 1)).astype(int)

    dot = out["I151"].fillna(0) + out["I160"].fillna(0)
    use_i020 = dot == 0
    dot = dot.where(~use_i020, out["I020"].fillna(0))
    out["DOTACION"] = dot / 12

    valid_prod = (out["C041"] > 0) & (out["DOTACION"] > 0)
    out["PROD_LAB"] = np.where(valid_prod, np.log(out["C041"] * 1000 / out["DOTACION"]), np.nan)
    valid_sal = (out["C048"] > 0) & (out["DOTACION"] > 0)
    out["SAL_TRAB"] = np.where(valid_sal, np.log(out["C048"] * 1000 / out["DOTACION"]), np.nan)

    out["TAMANO_CAT"] = out["TAMANO"].apply(clasificar_tamano)
    out["ANTIG"] = (2019 - out["A068"]).clip(lower=0)
    out["ln_ANTIG"] = np.log1p(out["ANTIG"])
    out["GRUPO_EMP"] = (out["A069"] == 1).astype(int)
    out["SECTOR"] = out["CIIU_FINAL"].astype(str).str.strip().str[0]
    out["OLA"] = "ELE-6"
    out["FE"] = out["FE"].fillna(1)
    return out


def cargar_ele7():
    df = pd.read_csv(DATOS / "ele7-full.csv", sep=";",
                     encoding="latin-1", low_memory=False)

    out = pd.DataFrame()
    for col in ["D097", "D176", "D106", "I151", "I160",
                "C077", "C084", "TAMANO", "A068", "A069", "FE_TRANSVERSAL"]:
        if col in df.columns:
            out[col] = parse_comma(df[col])
        else:
            out[col] = np.nan

    out["CIIU_FINAL"] = df["CIIU_FINAL"].astype(str)
    out["FE"] = out["FE_TRANSVERSAL"]

    # CAP_ALLY: NaN se trata como 0 (no declarada asociación = no capacitación ally)
    # Consistente con las otras olas
    d097 = out["D097"].fillna(0)
    d176 = out["D176"].fillna(0)
    d106 = out["D106"].fillna(0)
    out["CAP_ALLY"] = ((d097 == 1) | (d176 == 1) | (d106 == 1)).astype(int)

    dot = out["I151"].fillna(0) + out["I160"].fillna(0)
    out["DOTACION"] = dot / 12

    valid_prod = (out["C077"] > 0) & (out["DOTACION"] > 0)
    out["PROD_LAB"] = np.where(valid_prod, np.log(out["C077"] * 1000 / out["DOTACION"]), np.nan)
    valid_sal = (out["C084"] > 0) & (out["DOTACION"] > 0)
    out["SAL_TRAB"] = np.where(valid_sal, np.log(out["C084"] * 1000 / out["DOTACION"]), np.nan)

    out["TAMANO_CAT"] = out["TAMANO"].apply(clasificar_tamano)
    out["ANTIG"] = (2022 - out["A068"]).clip(lower=0)
    out["ln_ANTIG"] = np.log1p(out["ANTIG"])
    out["GRUPO_EMP"] = (out["A069"] == 1).astype(int)
    out["SECTOR"] = out["CIIU_FINAL"].astype(str).str.strip().str[0]
    out["OLA"] = "ELE-7"
    out["FE"] = out["FE"].fillna(1)
    return out


def analizar_ola(frame, nombre_ola):
    # Muestra para tasas descriptivas: solo requiere tamaño asignado
    # (CAP_ALLY ya está imputado como 0 cuando NaN en los cargadores)
    mask_desc = frame["TAMANO_CAT"].notna() & frame["CAP_ALLY"].notna()
    da_desc = frame[mask_desc].copy()

    # Muestra analítica para regresiones: requiere SAL_TRAB válido
    # (replica criterio del Piloto 2 v5 original)
    mask_reg = (
        frame["TAMANO_CAT"].notna()
        & frame["CAP_ALLY"].notna()
        & frame["SAL_TRAB"].notna()
    )
    da = frame[mask_reg].copy()
    if len(da) > 100:
        # Trim 1-99 en SAL_TRAB para robustez
        p1, p99 = da["SAL_TRAB"].quantile([0.01, 0.99])
        da = da[(da["SAL_TRAB"] >= p1) & (da["SAL_TRAB"] <= p99)].copy()

    resultados = {
        "ola": nombre_ola,
        "n_muestra_descriptiva": len(da_desc),
        "n_expandido_descriptiva": int(da_desc["FE"].sum()),
        "n_muestra_analitica": len(da),
    }

    pyme = da[da["TAMANO_CAT"] == "PYME"]
    if len(pyme) > 10:
        sal_mes_pyme = np.exp(pyme["SAL_TRAB"].mean()) / 12
        resultados["salario_mensual_medio_pyme_CLP"] = int(round(sal_mes_pyme, 0))

    # TASAS sobre muestra descriptiva (amplia)
    tasas = tasa_ponderada(da_desc)
    resultados["tasa_general_pct"] = tasas["General"]["tasa_pct"]
    resultados["tasa_grande_pct"] = tasas["Grande"]["tasa_pct"]
    resultados["tasa_pyme_pct"] = tasas["PYME"]["tasa_pct"]
    resultados["tasa_micro_pct"] = tasas["Micro"]["tasa_pct"]
    resultados["ratio_grande_pyme"] = round(
        tasas["Grande"]["tasa_pct"] / tasas["PYME"]["tasa_pct"], 2
    ) if tasas["PYME"]["tasa_pct"] > 0 else np.nan
    resultados["n_grande_muestra"] = tasas["Grande"]["N_muestra"]
    resultados["n_pyme_muestra"] = tasas["PYME"]["N_muestra"]
    resultados["n_micro_muestra"] = tasas["Micro"]["N_muestra"]

    da["PYME_dummy"] = (da["TAMANO_CAT"] == "PYME").astype(int)
    da["Micro_dummy"] = (da["TAMANO_CAT"] == "Micro").astype(int)

    try:
        y, X, names = design(
            da, "SAL_TRAB",
            ["CAP_ALLY", "PYME_dummy", "Micro_dummy", "ln_ANTIG", "GRUPO_EMP"],
            ["SECTOR"],
        )
        m_full = OLSHC3(y, X, names)
        beta = m_full.params["CAP_ALLY"]
        resultados["beta_sal_completa"] = round(beta, 4)
        resultados["se_sal_completa"] = round(m_full.bse["CAP_ALLY"], 4)
        resultados["p_sal_completa"] = round(m_full.pvalues["CAP_ALLY"], 4)
        resultados["prima_sal_completa_pct"] = round((np.exp(beta) - 1) * 100, 2)
        resultados["n_reg_completa"] = m_full.nobs
    except Exception as e:
        resultados["error_completa"] = str(e)

    pyme_reg = da[da["TAMANO_CAT"] == "PYME"].copy()
    try:
        if len(pyme_reg) >= 100:
            y, X, names = design(
                pyme_reg, "SAL_TRAB",
                ["CAP_ALLY", "ln_ANTIG", "GRUPO_EMP"],
                ["SECTOR"],
            )
            m_pyme = OLSHC3(y, X, names)
            beta_p = m_pyme.params["CAP_ALLY"]
            resultados["beta_sal_pyme"] = round(beta_p, 4)
            resultados["se_sal_pyme"] = round(m_pyme.bse["CAP_ALLY"], 4)
            resultados["p_sal_pyme"] = round(m_pyme.pvalues["CAP_ALLY"], 4)
            resultados["prima_sal_pyme_pct"] = round((np.exp(beta_p) - 1) * 100, 2)
            resultados["ic95_pyme_inf"] = round((np.exp(m_pyme.ci_lo["CAP_ALLY"]) - 1) * 100, 2)
            resultados["ic95_pyme_sup"] = round((np.exp(m_pyme.ci_hi["CAP_ALLY"]) - 1) * 100, 2)
            resultados["n_reg_pyme"] = m_pyme.nobs
    except Exception as e:
        resultados["error_pyme"] = str(e)

    return resultados


print("=" * 78)
print("OPPRU-Chile · Piloto 2 v5.1 · Análisis longitudinal ELE-4/5/6/7")
print("=" * 78)

cargadores = [
    ("ELE-4 (2013)", 2013, cargar_ele4),
    ("ELE-5 (2015)", 2015, cargar_ele5),
    ("ELE-6 (2019)", 2019, cargar_ele6),
    ("ELE-7 (2022)", 2022, cargar_ele7),
]

filas = []
for etiqueta, anio, fn in cargadores:
    print(f"\n── Procesando {etiqueta} ──")
    try:
        df = fn()
        print(f"   Filas cargadas: {len(df):,}")
        r = analizar_ola(df, etiqueta)
        r["anio"] = anio
        filas.append(r)
        print(f"   Muestra descriptiva: {r['n_muestra_descriptiva']:,} empresas (expandido: {r['n_expandido_descriptiva']:,})")
        print(f"   Muestra analítica:   {r['n_muestra_analitica']:,} empresas")
        print(f"   Tasa general: {r['tasa_general_pct']:.2f}%")
        print(f"   Tasa Grande: {r.get('tasa_grande_pct', 'N/A')}%")
        print(f"   Tasa PYME:   {r.get('tasa_pyme_pct', 'N/A')}%")
        print(f"   Tasa Micro:  {r.get('tasa_micro_pct', 'N/A')}%")
        print(f"   Ratio G/PYME: {r.get('ratio_grande_pyme', 'N/A')}")
        if 'salario_mensual_medio_pyme_CLP' in r:
            print(f"   Salario mensual medio PYME: CLP {r['salario_mensual_medio_pyme_CLP']:,}")
        if 'beta_sal_pyme' in r:
            sig = "***" if r["p_sal_pyme"] < 0.01 else "**" if r["p_sal_pyme"] < 0.05 else "*" if r["p_sal_pyme"] < 0.10 else ""
            print(f"   β_SAL_PYME = {r['beta_sal_pyme']:+.4f}{sig}  (p={r['p_sal_pyme']:.4f}, prima={r['prima_sal_pyme_pct']:+.2f}%, IC95%: [{r['ic95_pyme_inf']:+.1f}%, {r['ic95_pyme_sup']:+.1f}%])")
    except Exception as e:
        print(f"   ✗ Error procesando {etiqueta}: {e}")
        import traceback
        traceback.print_exc()

tabla = pd.DataFrame(filas)
tabla.to_csv(RES / "tabla_longitudinal_v5.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 78)
print("TABLA LONGITUDINAL CONSOLIDADA (v5 con correcciones)")
print("=" * 78)
cols_resumen = [
    "ola", "anio", "n_muestra_descriptiva", "n_muestra_analitica",
    "tasa_general_pct", "tasa_grande_pct",
    "tasa_pyme_pct", "tasa_micro_pct", "ratio_grande_pyme",
    "beta_sal_pyme", "p_sal_pyme", "prima_sal_pyme_pct",
    "ic95_pyme_inf", "ic95_pyme_sup",
]
cols_disp = [c for c in cols_resumen if c in tabla.columns]
print(tabla[cols_disp].to_string(index=False))

print("\n✓ Tabla longitudinal guardada en", RES / "tabla_longitudinal_v5.csv")
