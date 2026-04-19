# Pilot Study 1 — Doctoral Research in Sociology
## Firm-level training, wages and fiscal sensitivity under the SENCE Training Tax Credit (Franquicia Tributaria)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: ELE-7](https://img.shields.io/badge/Data-ELE--7%202020--2022-blue)](https://www.economia.gob.cl)
[![Status: Working Paper](https://img.shields.io/badge/Status-Working%20Paper-orange)]()

---

## Overview

This repository contains the code, aggregated data outputs and documents for **Pilot Study 1**, part of a doctoral dissertation in Sociology at the **Universidad Complutense de Madrid (UCM)**, on workforce recertification and upskilling/reskilling strategies in Chile.

The study analyses the distribution of firm-level training activity, its association with wages and labour productivity, and the fiscal implications of eliminating the SENCE Training Tax Credit (Franquicia Tributaria SENCE), using microdata from the **Seventh Longitudinal Enterprise Survey (ELE-7, 2020-2022)**.

**Key findings:**
- Only 8.3% of formal firms declare any collaborative training activity (16.3% large firms vs. 4.8% SMEs vs. 3.9% micro).
- The training-wage premium in SMEs is ~15% (OLS, controlled for sector, age and group membership), rising from 11.3% in 2017 to 15.0% in 2022.
- The concentration ratio (large/SME) increased from 2.9x (2017) to 3.6x (2022): the Matthew Effect in human capital.
- Fiscal break-even analysis shows the elimination of the FT is fiscally regressive if more than 44% of the observed wage differential is causally attributable to training — within the 40-70% range estimated by the international literature.

---

## Repository Structure

```
oppru-chile-sence-ft-pilot1/
|
|-- data/
|   |-- raw/            # Data dictionaries and variable descriptors only
|   |                   # (ELE-7 microdata NOT included: confidential)
|   |-- processed/      # Aggregated, non-identifiable output tables (CSV)
|
|-- scripts/
|   |-- fase1_descriptivos_ele7.py       # Descriptive analysis & CAP_ALLY proxy
|   |-- fase2_regresiones_ols.py         # OLS wage and productivity models
|   |-- fase3a_serie_longitudinal_ele.py # Harmonised ELE-4 to ELE-7 series
|   |-- fase3b_simulacion_fiscal.py      # Fiscal break-even simulation
|
|-- figures/
|   |-- fig1_tasa_capacitacion_tamano.png
|   |-- fig2_serie_brecha_ele4_ele7.png
|   |-- fig3_prima_salarial_pyme.png
|   |-- fig4_breakeven_fiscal.png
|
|-- docs/
|   |-- policy_brief_pilot1.pdf
|   |-- columna_terceradosis_sence_ft.pdf
|
|-- README.md
|-- LICENSE
```

---

## Data

| Source | Description | Access |
|--------|-------------|--------|
| ELE-7 (Ministerio de Economia, 2024) | 7th Longitudinal Enterprise Survey, 2020-2022 | Public: economia.gob.cl |
| ELE-4 to ELE-6 (INE/Ministerio de Economia) | Earlier waves for harmonised series | Public: ine.gob.cl |
| DIPRES (2025) | SENCE FT impact evaluation | Public report |
| OECD (2018, 2019) | Social mobility & adult learning benchmarks | Public |

> **Important:** Raw ELE microdata files are NOT included in this repository due to confidentiality restrictions. The `data/raw/` folder contains only variable dictionaries and descriptors. To replicate the analysis, download the official ELE-7 CSV modules from economia.gob.cl and place them in `data/raw/`.

---

## Methods

- **Training proxy (CAP_ALLY):** Binary indicator derived from the inter-firm collaboration motives block (ELE-7 variable D097/equivalent). Captures declared collaborative training activity — a conservative lower bound, not a direct measure of SENCE FT usage.
- **OLS models:** Log wage bill per worker and log sales per worker, controlling for sector (CIIU 2-digit), firm age, group membership and size category.
- **Harmonised series (ELE-4 to ELE-7):** Best-available proxy for training in each wave, documented in `scripts/fase3a`; results constitute a *harmonised series*, not a strict panel.
- **Fiscal sensitivity (break-even):** Parametric simulation of net fiscal balance as a function of alpha (causal fraction of wage differential), including payroll taxes, health contributions and pension contributions (~20% of wage mass).

---

## Limitations

- CAP_ALLY is a proxy, not a direct measure of SENCE FT usage. The ELE-7 contains no variable identifying FT users.
- OLS models are associative, not causal. Causal identification requires an administrative linkage (SENCE registry x ELE via RUT) under an institutional agreement not yet publicly available.
- The harmonised series uses non-identical variables across waves; all differences are documented in `scripts/fase3a`.
- Fiscal simulation uses conservative default parameters; results should be updated with official DIPRES and SII figures.

---

## Citation

If you use this code or outputs, please cite:

```
Ponce Bollmann, F. (2026). Firm-level training, wages and fiscal sensitivity
under the SENCE Training Tax Credit. Pilot Study 1. Doctoral Research in Sociology.
Universidad Complutense de Madrid.
https://github.com/feponce-beep/oppru-chile-sence-ft-pilot1
```

---

## Author

**Felipe Ponce Bollmann**
Sociologist | Doctoral Researcher
Departamento de Sociologia — Universidad Complutense de Madrid (UCM)

---

## License

Code: MIT License (see LICENSE file).
Documents (policy brief, column): Creative Commons Attribution 4.0 International (CC BY 4.0).
Data outputs: see data source licenses above.
