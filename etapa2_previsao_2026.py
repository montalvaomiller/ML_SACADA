# ============================================================
# ETAPA 2 — Previsão 2026 (compatível com a Etapa 3) + tqdm
# Entrega: abas Previsao_2026 e Alocacao_2026
# Chaves esperadas pela Etapa 3: GRIFFE, LINHA, GRUPO_PRODUTO, FAIXA_PRECO, ANO_MES
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
BASE_DIR    = Path(r"C:\Users\monta\OneDrive\Documentos\Meta\METAxSACADA\ML-Sacada")
IN_CSV      = BASE_DIR / "saidas" / "staging_consolidado.csv"            # histórico de vendas
ESTOQUE_CSV = BASE_DIR / "dados"  / "estoque_grupo_por_filial.csv"       # opcional (regressores)
OUT_XLSX    = BASE_DIR / "saidas" / "previsao_2026.xlsx"

# Parâmetros
GRIFFE_DEFAULT         = "SACADA"   # usado se a coluna GRIFFE não existir
JANELA_SHARE_MESES     = 6          # últimos N meses para shares de alocação por filial
DESCONTO_MAP = {                    # regra simples por SITUACAO
    "ATUAL": 0.05,
    "ANTERIOR": 0.15,
    "OFF": 0.30,
    "SEM INFO": 0.10
}

# ===================== FUNÇÕES AUXILIARES =====================
def to_upper_strip(s):
    return s.astype(str).str.upper().str.strip()

def faixa_preco_from_val(v):
    try:
        b = int(np.floor((0 if pd.isna(v) else v) / 100.0) * 100)
    except Exception:
        b = 0
    return f"R${b}–R${b+99}"

def modo(series):
    vc = series.dropna().value_counts()
    return vc.index[0] if len(vc) else np.nan

# ===================== LOAD =====================
df = pd.read_csv(IN_CSV)
estoque = None
try:
    if ESTOQUE_CSV.exists():
        estoque = pd.read_csv(ESTOQUE_CSV)
except Exception:
    estoque = None

# Padroniza nomes e tipos básicos
df.rename(columns={
    "FILIAL": "FILIAL_2",           # se vier como FILIAL, padroniza para FILIAL_2
    "DATA_VENDA": "DATA",
    "QTDE": "QTDE_VENDIDA"
}, inplace=True)

# Colunas de texto
for c in ["FILIAL_2","GRIFFE","LINHA","GRUPO_PRODUTO","SITUACAO","COLECAO"]:
    if c in df.columns:
        df[c] = to_upper_strip(df[c])

# Garantias mínimas
if "GRIFFE" not in df.columns:
    df["GRIFFE"] = GRIFFE_DEFAULT
if "SITUACAO" not in df.columns:
    df["SITUACAO"] = "SEM INFO"

# Datas
df["DATA"] = pd.to_datetime(df["DATA"], errors="coerce")
df = df.dropna(subset=["DATA"]).sort_values("DATA")

# Preço unitário (se não houver, zera para não quebrar cálculos)
if "PRECO_UNIT" not in df.columns:
    df["PRECO_UNIT"] = 0.0

# FAIXA_PRECO (baseada no histórico)
df["FAIXA_PRECO"] = df["PRECO_UNIT"].apply(faixa_preco_from_val)

# Regressor de estoque/giro (opcional)
if estoque is not None:
    for c in ["FILIAL_2","GRUPO_PRODUTO"]:
        if c in estoque.columns:
            estoque[c] = to_upper_strip(estoque[c])
    if {"FILIAL_2","GRUPO_PRODUTO","QTD_ESTOQUE"}.issubset(estoque.columns):
        est = estoque.groupby(["FILIAL_2","GRUPO_PRODUTO"], as_index=False)["QTD_ESTOQUE"].sum()
        df = df.merge(est, on=["FILIAL_2","GRUPO_PRODUTO"], how="left")
    elif {"GRUPO_PRODUTO","QTD_ESTOQUE"}.issubset(estoque.columns):
        est = estoque.groupby(["GRUPO_PRODUTO"], as_index=False)["QTD_ESTOQUE"].sum()
        df = df.merge(est, on="GRUPO_PRODUTO", how="left")
    else:
        df["QTD_ESTOQUE"] = 0
else:
    df["QTD_ESTOQUE"] = 0

df["QTD_ESTOQUE"]  = df["QTD_ESTOQUE"].fillna(0)
df["GIRO_ESTOQUE"] = (df["QTDE_VENDIDA"] / df["QTD_ESTOQUE"].replace(0, np.nan)).fillna(0).clip(0, 10)

# ===================== AGREGAÇÃO MENSAL (previsão sem filial) =====================
# Etapa 3 une Previsao (sem filial) com Alocacao (com filial).
df["ANO_MES"] = df["DATA"].dt.to_period("M").astype(str)
df_mensal = (
    df.groupby(["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO", pd.Grouper(key="DATA", freq="MS")], dropna=False)
      .agg(QTDE_VENDIDA=("QTDE_VENDIDA","sum"),
           PRECO_MEDIO=("PRECO_UNIT","mean"),
           QTD_ESTOQUE=("QTD_ESTOQUE","mean"),
           GIRO_ESTOQUE=("GIRO_ESTOQUE","mean"))
      .reset_index()
      .rename(columns={"DATA":"DATA_REF"})
)

# Preencher regressors
df_mensal = df_mensal.sort_values(["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","DATA_REF"]).copy()
for col_fill in ["QTD_ESTOQUE","GIRO_ESTOQUE"]:
    df_mensal[col_fill] = (
        df_mensal.groupby(["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"])[col_fill]
                 .transform(lambda s: s.ffill().bfill())
                 .fillna(0)
    )

# ===================== PREVISÃO (Prophet + ARIMA) =====================
prev_list = []
group_iter = df_mensal.groupby(["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"])
total_groups = len(group_iter)

for keys, g in tqdm(group_iter, total=total_groups, desc="⏳ Gerando previsões", ncols=100):
    g = g.sort_values("DATA_REF").copy()
    if len(g) < 6:
        continue

    # Prophet
    p_df = g.rename(columns={"DATA_REF":"ds","QTDE_VENDIDA":"y"})[["ds","y","QTD_ESTOQUE","GIRO_ESTOQUE"]].copy()
    p_df = p_df.dropna(subset=["ds","y"])
    for reg in ["QTD_ESTOQUE","GIRO_ESTOQUE"]:
        p_df[reg] = p_df[reg].interpolate(limit_direction="both").fillna(0)

    try:
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.add_regressor("QTD_ESTOQUE")
        model.add_regressor("GIRO_ESTOQUE")
        model.fit(p_df)
    except Exception:
        model = None

    last_qtd  = float(p_df["QTD_ESTOQUE"].iloc[-1]) if len(p_df) else 0.0
    last_giro = float(p_df["GIRO_ESTOQUE"].iloc[-1]) if len(p_df) else 0.0
    fut_ds = pd.date_range(g["DATA_REF"].max() + pd.offsets.MonthBegin(), periods=12, freq="MS")

    if model is not None:
        future = model.make_future_dataframe(periods=12, freq="MS")
        future = future.merge(p_df[["ds","QTD_ESTOQUE","GIRO_ESTOQUE"]], on="ds", how="left")
        future["QTD_ESTOQUE"]  = future["QTD_ESTOQUE"].fillna(last_qtd)
        future["GIRO_ESTOQUE"] = future["GIRO_ESTOQUE"].fillna(last_giro)
        f_prop = model.predict(future)[["ds","yhat"]].tail(12).reset_index(drop=True)
    else:
        f_prop = pd.DataFrame({"ds": fut_ds, "yhat": np.nan})

    # ARIMA
    try:
        arima_fit = SARIMAX(g.set_index("DATA_REF")["QTDE_VENDIDA"], order=(1,1,1), seasonal_order=(1,1,0,12)).fit(disp=False)
        f_arima = pd.DataFrame({"ds": fut_ds, "yhat_arima": arima_fit.forecast(12).values})
    except Exception:
        f_arima = pd.DataFrame({"ds": fut_ds, "yhat_arima": np.nan})

    # Combinação
    comb = pd.DataFrame({"ds": fut_ds}).merge(f_prop, on="ds", how="left").merge(f_arima, on="ds", how="left")
    comb["QTDE_PREVISTA"] = comb[["yhat","yhat_arima"]].mean(axis=1, skipna=True).fillna(0).clip(lower=0).round()

    griffe, linha, grupo, faixa = keys
    comb["GRIFFE"]        = griffe
    comb["LINHA"]         = linha
    comb["GRUPO_PRODUTO"] = grupo
    comb["FAIXA_PRECO"]   = faixa
    comb["ANO_MES"]       = comb["ds"].dt.to_period("M").astype(str)

    prev_list.append(comb[["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","ANO_MES","QTDE_PREVISTA"]])

prev_df = pd.concat(prev_list, ignore_index=True) if prev_list else pd.DataFrame(
    columns=["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","ANO_MES","QTDE_PREVISTA"]
)

# ===================== PREÇO MÉDIO APLICADO & DESCONTO & VALOR =====================
# PRECO_MEDIO_APLICADO: média dos últimos 6 meses por chave
cutoff = (df["DATA"].max() - pd.DateOffset(months=JANELA_SHARE_MESES))

hist6 = (
    df[df["DATA"] >= cutoff]
    .groupby(["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"], dropna=False)["PRECO_UNIT"]
    .mean()
    .rename("PRECO_MEDIO_APLICADO")
    .reset_index()
)

# SITUACAO por chave (modo dos últimos 6 meses)
sit6 = (
    df[df["DATA"] >= cutoff]
    .groupby(["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"], dropna=False)["SITUACAO"]
    .agg(modo)
    .rename("SITUACAO_REF")
    .reset_index()
)

prev_df = prev_df.merge(hist6, on=["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"], how="left")
prev_df = prev_df.merge(sit6, on=["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"], how="left")

prev_df["PRECO_MEDIO_APLICADO"] = prev_df["PRECO_MEDIO_APLICADO"].fillna(
    prev_df.groupby(["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"])["PRECO_MEDIO_APLICADO"].transform("median")
).fillna(0)

prev_df["DESCONTO_SUGERIDO"] = prev_df["SITUACAO_REF"].map(DESCONTO_MAP).fillna(DESCONTO_MAP["SEM INFO"])
prev_df["VAL_VENDA_PREVISTA"] = prev_df["QTDE_PREVISTA"] * prev_df["PRECO_MEDIO_APLICADO"] * (1 - prev_df["DESCONTO_SUGERIDO"])

# ===================== ALOCAÇÃO POR FILIAL (shares últimos 6 meses) =====================
hist6_filial = (
    df[(df["DATA"] >= cutoff) & df["FILIAL_2"].notna()]
    .groupby(["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","FILIAL_2"], dropna=False)["QTDE_VENDIDA"]
    .sum()
    .reset_index()
)
tot_key = (
    hist6_filial.groupby(["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"], dropna=False)["QTDE_VENDIDA"]
    .sum().rename("TOT").reset_index()
)
shares = hist6_filial.merge(tot_key, on=["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"], how="left")
shares["PCT"] = np.where(shares["TOT"] > 0, shares["QTDE_VENDIDA"] / shares["TOT"], np.nan)

# fallback: sem histórico recente -> usa histórico geral
if shares["PCT"].isna().all():
    all_filiais = (
        df[df["FILIAL_2"].notna()]
        .groupby(["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","FILIAL_2"])["QTDE_VENDIDA"]
        .sum().reset_index()
    )
    tot_all = (
        all_filiais.groupby(["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"])["QTDE_VENDIDA"]
        .sum().rename("TOT").reset_index()
    )
    shares = all_filiais.merge(tot_all, on=["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"], how="left")
    shares["PCT"] = np.where(shares["TOT"] > 0, shares["QTDE_VENDIDA"] / shares["TOT"], 0)

shares = shares[["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","FILIAL_2","PCT"]].copy()

aloc_list = []
group_prev = prev_df.groupby(["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"])
for keys, gprev in tqdm(group_prev, total=len(group_prev), desc="📦 Alocando por filial", ncols=100):
    sub_share = shares.merge(
        pd.DataFrame([dict(zip(["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"], keys))]),
        on=["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"], how="inner"
    )
    if sub_share.empty:
        sub_share = pd.DataFrame([{
            "GRIFFE": keys[0], "LINHA": keys[1], "GRUPO_PRODUTO": keys[2], "FAIXA_PRECO": keys[3],
            "FILIAL_2": "SEM FILIAL", "PCT": 1.0
        }])
    gkeys = gprev[["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","ANO_MES","QTDE_PREVISTA"]].copy()
    sub = gkeys.merge(sub_share, on=["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"], how="left")
    sub["PCT"] = sub["PCT"].fillna(0)
    sub["QTDE_ALOCADA"] = (sub["QTDE_PREVISTA"] * sub["PCT"]).round()
    aloc_list.append(sub[["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","ANO_MES","FILIAL_2","QTDE_ALOCADA"]])

aloc_df = pd.concat(aloc_list, ignore_index=True) if aloc_list else pd.DataFrame(
    columns=["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","ANO_MES","FILIAL_2","QTDE_ALOCADA"]
)

# ===================== SALVAR =====================
with pd.ExcelWriter(OUT_XLSX) as xw:
    prev_out = prev_df[[
        "GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","ANO_MES",
        "QTDE_PREVISTA","PRECO_MEDIO_APLICADO","DESCONTO_SUGERIDO","VAL_VENDA_PREVISTA"
    ]].copy()
    prev_out.to_excel(xw, sheet_name="Previsao_2026", index=False)

    aloc_out = aloc_df[[
        "GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","ANO_MES","FILIAL_2","QTDE_ALOCADA"
    ]].copy()
    aloc_out.to_excel(xw, sheet_name="Alocacao_2026", index=False)

print("✅ Etapa 2 finalizada (compatível com Etapa 3) com barras de carregamento.")
print(f"Arquivo salvo em: {OUT_XLSX}")
