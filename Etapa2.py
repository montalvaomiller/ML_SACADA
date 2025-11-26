"""
ETAPA 2 — Previsão 2026 (vendas + estoque)

Entradas (em ML_SACADA/saidas):
    - staging_consolidado.csv         ← Saída da Etapa 1 (vendas limpas)
    - estoque_historico_mensal.csv    ← Saída da Etapa 1B (estoque por mês, filial, grupo)

Saída:
    - previsao_2026.xlsx
        • aba "Previsao_2026"
          → previsão mensal agregada por (GRIFFE, LINHA, GRUPO_PRODUTO, FAIXA_PRECO)
        • aba "Alocacao_2026"
          → previsão alocada por filial (FILIAL_2) via share histórico

Fluxo:
    1. Carrega vendas e estoque.
    2. Agrega vendas por mês/chave + cria regressoras de estoque em nível de rede.
    3. Modela cada série (GRIFFE, LINHA, GRUPO_PRODUTO, FAIXA_PRECO) com Prophet.
    4. Calcula preço médio aplicado, situação de referência, desconto sugerido.
    5. Calcula valor de venda previsto.
    6. Aloca quantidade prevista por filial com base no histórico recente.
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from prophet import Prophet
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ===================== CONFIGURAÇÕES =====================

BASE_DIR = Path(__file__).resolve().parent
SAIDAS_DIR = BASE_DIR / "saidas"

PATH_VENDAS  = SAIDAS_DIR / "staging_consolidado.csv"
PATH_ESTOQUE = SAIDAS_DIR / "estoque_historico_mensal.csv"
OUT_XLSX     = SAIDAS_DIR / "previsao_2026.xlsx"

# mínimo de meses de histórico para modelar uma série
MIN_MESES_HIST = 8
# janela (em meses) para cálculo de preço médio e shares de filial
JANELA_MESES_PRECO_SHARE = 6

# consideramos um m�s "completo" se temos vendas at� pelo menos este dia
MIN_DIA_MES_COMPLETO = 20
# janela usada para suavizar regressoras futuras (mediana dos �ltimos N meses)
JANELA_REGRESSORES_FUTURO = 3

# Descontos sugeridos por situação de referência
DESCONTO_MAP = {
    "ATUAL": 0.05,
    "ANTERIOR": 0.15,
    "OFF": 0.30,
    "SEM INFO": 0.10
}

TARGET_INICIO_ANOMES = 202601
TARGET_FIM_ANOMES = 202612
TARGET_INICIO_PERIOD = pd.Period(str(TARGET_INICIO_ANOMES), freq="M")
TARGET_FIM_PERIOD = pd.Period(str(TARGET_FIM_ANOMES), freq="M")


# ===================== FUNÇÕES AUXILIARES =====================

def log(msg: str):
    try:
        print(f"[ETAPA 2] {msg}")
    except UnicodeEncodeError:
        # fallback para consoles sem UTF-8
        print("[ETAPA 2] " + str(msg).encode("ascii", "ignore").decode("ascii"))


def to_upper_strip(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.upper()
         .str.strip()
         .replace({"NAN": np.nan, "": np.nan})
    )


def modo(series: pd.Series):
    vc = series.dropna().value_counts()
    return vc.index[0] if len(vc) else np.nan


def build_data_venda_from_anomes(df: pd.DataFrame) -> pd.Series:
    """Cria uma data representativa (1º dia do mês) a partir de ANO_MES=YYYYMM."""
    anomes = df["ANO_MES"].astype(str).str.strip()
    return pd.to_datetime(anomes + "01", format="%Y%m%d", errors="coerce")


# ===================== CARREGAMENTO E PREPARO =====================

def carregar_vendas(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de vendas não encontrado: {path}")
    log(f"Lendo vendas de: {path}")

    # low_memory=False para evitar DtypeWarning chatinho
    df = pd.read_csv(path, sep=";", low_memory=False)

    # normalização de textos principais
    for c in ["CANAL", "FILIAL_2", "COLECAO", "LINHA", "GRUPO_PRODUTO", "GRIFFE", "FAIXA_PRECO"]:
        if c in df.columns:
            df[c] = to_upper_strip(df[c])

    # DATA_VENDA
    if "DATA_VENDA" in df.columns:
        df["DATA_VENDA"] = pd.to_datetime(df["DATA_VENDA"], errors="coerce")
    else:
        # se não tiver DATA_VENDA, tenta construir a partir de ANO_MES
        if "ANO_MES" not in df.columns:
            raise ValueError("staging_consolidado.csv precisa ter DATA_VENDA ou ANO_MES.")
        df["DATA_VENDA"] = build_data_venda_from_anomes(df)

    # ANO_MES
    if "ANO_MES" not in df.columns:
        df["ANO_MES"] = df["DATA_VENDA"].dt.strftime("%Y%m")
    df["ANO_MES"] = df["ANO_MES"].astype(str).str.strip()

    # garante numericamente algumas colunas
    for c in ["VAL_VENDA", "VAL_VENDA_BRUTA", "VAL_ORIGINAL", "QTDE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # garante FILIAL_2
    if "FILIAL_2" not in df.columns:
        df["FILIAL_2"] = "SEM FILIAL"
    df["FILIAL_2"] = to_upper_strip(df["FILIAL_2"]).fillna("SEM FILIAL")

    # garante FAIXA_PRECO (se não existir, coloca P2)
    if "FAIXA_PRECO" not in df.columns:
        df["FAIXA_PRECO"] = "P2"

    # garante LINHA/GRUPO_PRODUTO como texto
    for c in ["LINHA", "GRUPO_PRODUTO"]:
        if c not in df.columns:
            df[c] = "SEM_INFO"
        df[c] = to_upper_strip(df[c]).fillna("SEM_INFO")

    return df


def carregar_estoque(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de estoque não encontrado: {path}")
    log(f"Lendo estoque de: {path}")
    df = pd.read_csv(path)

    if df.empty:
        log("⚠ Arquivo de estoque está vazio.")
        return df

    # normalização ANO_MES, FILIAL_2, GRUPO_PRODUTO
    for c in ["ANO_MES", "FILIAL_2", "GRUPO_PRODUTO"]:
        if c in df.columns:
            df[c] = to_upper_strip(df[c])

    # converte ANO_MES para string YYYYMM
    if "ANO_MES" in df.columns:
        df["ANO_MES"] = df["ANO_MES"].astype(str).str.strip()

    # estoque médio e ruptura
    for c in ["ESTOQUE_MEDIO", "RUPTURA_FLAG"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return df


def preparar_bases(vendas: pd.DataFrame, estoque: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega vendas por mês e junta com estoque em nível de rede.
    Cria GIRO_REDE como regressora.
    """

    if vendas.empty:
        log("⚠ Base de vendas está vazia após filtros.")
        return pd.DataFrame()

    # 1) vendas mensais por chave
    vendas["DATA_REF"] = vendas["DATA_VENDA"].values.astype("datetime64[M]")

    group_cols = ["GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO", "ANO_MES", "DATA_REF"]
    vendas_mensal = (
        vendas
        .groupby(group_cols, dropna=False)
        .agg(
            QTDE_VENDIDA=("QTDE", "sum"),
            VAL_VENDA=("VAL_VENDA", "sum"),
        )
        .reset_index()
    )
    vendas_mensal["PRECO_MEDIO"] = (
        vendas_mensal["VAL_VENDA"] / vendas_mensal["QTDE_VENDIDA"]
    ).replace([np.inf, -np.inf], np.nan)

    # 2) estoque rede mensal: soma em todas as filiais + flags de qualidade
    if estoque.empty:
        log("⚠ Estoque vazio: regressoras de estoque ficarão zeradas.")
        estoque_rede = pd.DataFrame(columns=[
            "ANO_MES", "GRUPO_PRODUTO", "ESTOQUE_REDE", "RUPTURA_FLAG_REDE",
            "SNAPSHOT_UNICO_MEAN", "IMPUTADO_MEAN"
        ])
    else:
        estoque_rede = (
            estoque
            .groupby(["ANO_MES", "GRUPO_PRODUTO"], dropna=False)
            .agg(
                ESTOQUE_REDE=("ESTOQUE_MEDIO", "sum"),
                RUPTURA_FLAG_REDE=("RUPTURA_FLAG", "max"),
                SNAPSHOT_UNICO_MEAN=("SNAPSHOT_UNICO", "mean"),
                IMPUTADO_MEAN=("IMPUTADO", "mean"),
            )
            .reset_index()
        )

    # 🔧 ALINHAR TIPOS DE CHAVE (ANO_MES e GRUPO_PRODUTO) 🔧
    for col in ["ANO_MES", "GRUPO_PRODUTO"]:
        if col in vendas_mensal.columns:
            vendas_mensal[col] = vendas_mensal[col].astype(str).str.strip().str.upper()
        if col in estoque_rede.columns:
            estoque_rede[col] = estoque_rede[col].astype(str).str.strip().str.upper()

    # 3) merge vendas + estoque
    df = vendas_mensal.merge(
        estoque_rede,
        on=["ANO_MES", "GRUPO_PRODUTO"],
        how="left"
    )

    df["ESTOQUE_REDE"] = df["ESTOQUE_REDE"].fillna(0)
    df["RUPTURA_FLAG_REDE"] = df["RUPTURA_FLAG_REDE"].fillna(0)
    df["SNAPSHOT_UNICO_MEAN"] = df["SNAPSHOT_UNICO_MEAN"].fillna(0)
    df["IMPUTADO_MEAN"] = df["IMPUTADO_MEAN"].fillna(0)

    # fator de qualidade para penalizar meses frágeis de estoque
    df["QUALIDADE_ESTOQUE"] = 1 - 0.5 * df["SNAPSHOT_UNICO_MEAN"] - 0.3 * df["IMPUTADO_MEAN"]
    df["QUALIDADE_ESTOQUE"] = df["QUALIDADE_ESTOQUE"].clip(lower=0, upper=1)
    df["ESTOQUE_REDE_AJUST"] = df["ESTOQUE_REDE"] * df["QUALIDADE_ESTOQUE"]

    # 4) giro rede
    df["GIRO_REDE"] = np.where(
        df["ESTOQUE_REDE_AJUST"] > 0,
        df["QTDE_VENDIDA"] / df["ESTOQUE_REDE_AJUST"],
        np.nan
    )
    df["GIRO_REDE"] = df["GIRO_REDE"].clip(lower=0, upper=10).fillna(0)

    df = df.sort_values(
        ["GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO", "DATA_REF"]
    ).reset_index(drop=True)

    return df


# ===================== MODELAGEM DE SÉRIE =====================

def modelar_serie(df_serie: pd.DataFrame,
                  griffe: str,
                  linha: str,
                  grupo: str,
                  faixa: str,
                  max_data_venda=None) -> pd.DataFrame:
    """
    Modela uma s?rie temporal (Prophet) para (GRIFFE, LINHA, GRUPO_PRODUTO, FAIXA_PRECO).
    Usa QTDE_VENDIDA como y e DATA_REF como ds, com regressoras de estoque/giro.
    """

    if df_serie.empty:
        return pd.DataFrame(columns=[
            "GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO",
            "DATA_REF", "ANO_MES", "QTDE_PREVISTA"
        ])

    # ordena e remove m?s claramente parcial (?ltimo m?s com poucos dias)
    df_serie = df_serie.sort_values("DATA_REF").copy()
    if max_data_venda is not None and not pd.isna(max_data_venda):
        ultima_data_ref = df_serie["DATA_REF"].max()
        if pd.notna(ultima_data_ref):
            if ultima_data_ref.to_period("M") == max_data_venda.to_period("M") and max_data_venda.day < MIN_DIA_MES_COMPLETO:
                df_serie = df_serie[df_serie["DATA_REF"] < ultima_data_ref].copy()

    if df_serie.empty:
        return pd.DataFrame(columns=[
            "GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO",
            "DATA_REF", "ANO_MES", "QTDE_PREVISTA"
        ])

    # regressoras futuras suavizadas pela mediana recente
    reg_hist = df_serie.tail(JANELA_REGRESSORES_FUTURO)
    ult_estoque = reg_hist["ESTOQUE_REDE"].median()
    ult_giro = reg_hist["GIRO_REDE"].median()
    if pd.isna(ult_estoque):
        ult_estoque = 0
    if pd.isna(ult_giro):
        ult_giro = 0

    df_p = df_serie.rename(columns={"DATA_REF": "ds", "QTDE_VENDIDA": "y"}).copy()
    df_p = df_p[["ds", "y", "ESTOQUE_REDE", "GIRO_REDE"]]

    if len(df_p) < MIN_MESES_HIST:
        media_qtde = df_p["y"].mean()
        idx = pd.period_range(TARGET_INICIO_PERIOD, TARGET_FIM_PERIOD, freq="M")
        f = pd.DataFrame({
            "ds": idx.to_timestamp(),
            "yhat": media_qtde,
        })
    else:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
        )
        m.add_regressor("ESTOQUE_REDE")
        m.add_regressor("GIRO_REDE")
        m.fit(df_p)

        last_period = df_serie["DATA_REF"].max().to_period("M")
        meses_futuros = max((TARGET_FIM_PERIOD - last_period).n, 0)
        if meses_futuros <= 0:
            return pd.DataFrame(columns=[
                "GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO",
                "DATA_REF", "ANO_MES", "QTDE_PREVISTA"
            ])

        future = m.make_future_dataframe(
            periods=meses_futuros,
            freq="MS"
        )
        future["ESTOQUE_REDE"] = ult_estoque
        future["GIRO_REDE"] = ult_giro

        forecast = m.predict(future)
        f = forecast[["ds", "yhat"]].copy()

    f["QTDE_PREVISTA"] = f["yhat"].clip(lower=0)
    f["ANO_MES"] = f["ds"].dt.strftime("%Y%m")

    f["ANO_MES_INT"] = f["ANO_MES"].astype(int)
    mask_horizonte = (
        (f["ANO_MES_INT"] >= TARGET_INICIO_ANOMES)
        & (f["ANO_MES_INT"] <= TARGET_FIM_ANOMES)
    )
    f = f[mask_horizonte].copy()
    f.drop(columns=["ANO_MES_INT"], inplace=True)

    f["DATA_REF"] = f["ds"]
    f["GRIFFE"] = griffe
    f["LINHA"] = linha
    f["GRUPO_PRODUTO"] = grupo
    f["FAIXA_PRECO"] = faixa

    return f[["GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO", "DATA_REF", "ANO_MES", "QTDE_PREVISTA"]]

# ===================== PÓS-MODELO: PREÇO, DESCONTO E VALOR =====================

def calcular_preco_e_desconto(vendas: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula PRECO_MEDIO_APLICADO, SITUACAO_REF, DESCONTO_SUGERIDO e VAL_VENDA_PREVISTA
    usando histórico dos últimos JANELA_MESES_PRECO_SHARE meses.
    """

    if prev.empty:
        return prev.assign(
            PRECO_MEDIO_APLICADO=np.nan,
            SITUACAO_REF=np.nan,
            DESCONTO_SUGERIDO=np.nan,
            VAL_VENDA_PREVISTA=np.nan
        )

    group_cols = ["GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO"]

    # janela temporal
    max_data = vendas["DATA_VENDA"].max()
    cutoff = max_data - pd.DateOffset(months=JANELA_MESES_PRECO_SHARE)
    vendas_rec = vendas[vendas["DATA_VENDA"] >= cutoff].copy()

    def calc_preco_medio(df_base: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula um preço médio robusto a outliers, usando:
        - PRECO_UNIT da base de vendas (já vindo da Etapa 1)
        - cap no 99º percentil para evitar registros absurdos
        - mediana por (GRIFFE, LINHA, GRUPO_PRODUTO, FAIXA_PRECO).
        """
        if df_base.empty:
            return pd.DataFrame(columns=group_cols + ["PRECO_MEDIO_APLICADO"])

        df2 = df_base.copy()
        # Garante que temos PRECO_UNIT; se não tiver, calcula a partir de VAL_VENDA/QTDE
        if "PRECO_UNIT" not in df2.columns:
            df2["PRECO_UNIT"] = np.where(
                df2["QTDE"] > 0,
                df2["VAL_VENDA"] / df2["QTDE"],
                np.nan,
            )

        # Cap global no 99º percentil para reduzir impacto de outliers extremos
        q99 = df2["PRECO_UNIT"].quantile(0.99)
        df2["PRECO_UNIT_CAP"] = df2["PRECO_UNIT"].clip(lower=0, upper=q99)

        agg = (
            df2
            .groupby(group_cols, dropna=False)["PRECO_UNIT_CAP"]
            .median()
            .reset_index()
            .rename(columns={"PRECO_UNIT_CAP": "PRECO_MEDIO_APLICADO"})
        )

        return agg

    hist_preco = calc_preco_medio(vendas_rec)
    hist_preco_full = calc_preco_medio(vendas).rename(
        columns={"PRECO_MEDIO_APLICADO": "PRECO_MEDIO_HIST"}
    )

    # situação de referência (modo)
    if "SITUACAO" in vendas_rec.columns:
        hist_sit = (
            vendas_rec
            .groupby(group_cols, dropna=False)["SITUACAO"]
            .agg(modo)
            .reset_index()
            .rename(columns={"SITUACAO": "SITUACAO_REF"})
        )
    else:
        hist_sit = pd.DataFrame(columns=group_cols + ["SITUACAO_REF"])

    if "SITUACAO" in vendas.columns:
        hist_sit_full = (
            vendas
            .groupby(group_cols, dropna=False)["SITUACAO"]
            .agg(modo)
            .reset_index()
            .rename(columns={"SITUACAO": "SITUACAO_REF_TOTAL"})
        )
    else:
        hist_sit_full = pd.DataFrame(columns=group_cols + ["SITUACAO_REF_TOTAL"])

    prev = prev.merge(hist_preco, on=group_cols, how="left")
    prev = prev.merge(hist_preco_full, on=group_cols, how="left")
    prev = prev.merge(hist_sit, on=group_cols, how="left")
    prev = prev.merge(hist_sit_full, on=group_cols, how="left")

    # fallback de preço: recente → histórico completo → mediana por grupo
    prev["PRECO_MEDIO_APLICADO"] = prev["PRECO_MEDIO_APLICADO"].fillna(prev["PRECO_MEDIO_HIST"])
    prev.drop(columns=["PRECO_MEDIO_HIST"], inplace=True, errors="ignore")

    # fallback de preço por mediana dentro do grupo
    prev["PRECO_MEDIO_APLICADO"] = prev["PRECO_MEDIO_APLICADO"].fillna(
        prev.groupby(group_cols)["PRECO_MEDIO_APLICADO"].transform("median")
    )
    prev["PRECO_MEDIO_APLICADO"] = prev["PRECO_MEDIO_APLICADO"].fillna(0)

    prev["SITUACAO_REF"] = prev["SITUACAO_REF"].fillna(prev.get("SITUACAO_REF_TOTAL"))
    if "SITUACAO_REF_TOTAL" in prev.columns:
        prev.drop(columns=["SITUACAO_REF_TOTAL"], inplace=True)
    prev["SITUACAO_REF"] = prev["SITUACAO_REF"].fillna("SEM INFO")

    # desconto sugerido
    prev["DESCONTO_SUGERIDO"] = prev["SITUACAO_REF"].map(DESCONTO_MAP).fillna(DESCONTO_MAP["SEM INFO"])

    # valor previsto
    prev["VAL_VENDA_PREVISTA"] = (
        prev["QTDE_PREVISTA"] *
        prev["PRECO_MEDIO_APLICADO"] *
        (1 - prev["DESCONTO_SUGERIDO"])
    )

    return prev


# ===================== ALOCAÇÃO POR FILIAL =====================

def alocar_por_filial(vendas: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """
    Usa o histórico dos últimos JANELA_MESES_PRECO_SHARE meses para calcular o share
    de QTDE por FILIAL_2 em cada (GRIFFE, LINHA, GRUPO_PRODUTO, FAIXA_PRECO).
    Aplica esses shares na QTDE_PREVISTA para cada mês.
    """

    if prev.empty:
        return pd.DataFrame(columns=[
            "GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO",
            "ANO_MES", "FILIAL_2", "QTDE_ALOCADA"
        ])

    max_data = vendas["DATA_VENDA"].max()
    cutoff = max_data - pd.DateOffset(months=JANELA_MESES_PRECO_SHARE)
    vendas_rec = vendas[vendas["DATA_VENDA"] >= cutoff].copy()

    if vendas_rec.empty:
        log("⚠ Sem histórico recente para alocação por filial, usando média total.")
        vendas_rec = vendas.copy()

    group_cols_share = ["GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO", "FILIAL_2"]

    shares = (
        vendas_rec
        .groupby(group_cols_share, dropna=False)["QTDE"]
        .sum()
        .reset_index(name="QTDE_FILIAL")
    )

    total_por_grupo = (
        shares
        .groupby(["GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO"], dropna=False)["QTDE_FILIAL"]
        .sum()
        .reset_index(name="QTDE_TOTAL")
    )

    shares = shares.merge(
        total_por_grupo,
        on=["GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO"],
        how="left"
    )

    shares["SHARE_FILIAL"] = np.where(
        shares["QTDE_TOTAL"] > 0,
        shares["QTDE_FILIAL"] / shares["QTDE_TOTAL"],
        0
    )

    # se por algum motivo algum grupo tiver todos shares zerados, coloca 1/N
    def normalizar_grupo(g):
        soma = g["SHARE_FILIAL"].sum()
        if soma <= 0:
            n = len(g)
            g["SHARE_FILIAL"] = 1 / n if n > 0 else 0
        else:
            g["SHARE_FILIAL"] = g["SHARE_FILIAL"] / soma
        return g

    shares = shares.groupby(
        ["GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO"],
        group_keys=False
    ).apply(normalizar_grupo)

    # agora aplicamos os shares na previsão
    prev_key_cols = ["GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO", "ANO_MES"]
    base = prev[prev_key_cols + ["QTDE_PREVISTA"]].copy()

    base = base.merge(
        shares[["GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO", "FILIAL_2", "SHARE_FILIAL"]],
        on=["GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO"],
        how="left"
    )

    base["SHARE_FILIAL"] = base["SHARE_FILIAL"].fillna(0)

    base["QTDE_ALOCADA"] = base["QTDE_PREVISTA"] * base["SHARE_FILIAL"]
    base["QTDE_ALOCADA"] = base["QTDE_ALOCADA"].round(0)

    # ajuste para garantir que a soma por grupo/mês bate a QTDE_PREVISTA
    def ajustar_soma(g):
        soma_aloc = g["QTDE_ALOCADA"].sum()
        delta = g["QTDE_PREVISTA"].iloc[0] - soma_aloc
        if abs(delta) > 0:
            idx_max = g["QTDE_ALOCADA"].idxmax()
            g.loc[idx_max, "QTDE_ALOCADA"] += delta
        return g

    base = base.groupby(prev_key_cols, group_keys=False).apply(ajustar_soma)

    return base[
        ["GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO",
         "ANO_MES", "FILIAL_2", "QTDE_ALOCADA"]
    ]

# ===================== AJUSTES DE PISO/CAP POR SÉRIE =====================

def ajustar_series_por_hist(vendas: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica piso/cap por série usando histórico recente (últimos 6 meses):
      - se hist_total > 0 e previsão_total = 0 -> piso com média mensal do histórico
      - se ratio > CAP_RATIO -> reescala previsão pelo fator (CAP_RATIO/ratio)
      - se ratio < MIN_RATIO -> reescala previsão pelo fator (MIN_RATIO/ratio)
    """
    if prev_df.empty:
        return prev_df

    CAP_RATIO = 3.0
    MIN_RATIO = 0.1

    max_data = vendas["DATA_VENDA"].max()
    cutoff = max_data - pd.DateOffset(months=6)
    vendas_rec = vendas[vendas["DATA_VENDA"] >= cutoff].copy()

    # histórico por chave
    vendas_rec["ANO_MES_STR"] = vendas_rec["DATA_VENDA"].dt.strftime("%Y%m")
    hist = (
        vendas_rec
        .groupby(["GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO"], dropna=False)
        .agg(
            QTDE_ULT6M=("QTDE", "sum"),
            MESES_ULT6M=("ANO_MES_STR", "nunique"),
        )
        .reset_index()
    )

    prev_tot = (
        prev_df
        .groupby(["GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO"], dropna=False)["QTDE_PREVISTA"]
        .sum()
        .reset_index()
        .rename(columns={"QTDE_PREVISTA": "QTDE_PREV_TOTAL"})
    )

    base = prev_tot.merge(hist, on=["GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO"], how="left").fillna(0)
    base["ratio"] = np.where(base["QTDE_ULT6M"] > 0, base["QTDE_PREV_TOTAL"] / base["QTDE_ULT6M"], np.nan)

    def calc_action(row):
        qtde_prev = row["QTDE_PREV_TOTAL"]
        qtde_hist = row["QTDE_ULT6M"]
        ratio = row["ratio"]
        if qtde_hist > 0 and qtde_prev == 0:
            return "FILL_ZERO", 1.0
        if pd.isna(ratio):
            return "KEEP", 1.0
        if ratio > CAP_RATIO:
            return "CAP", CAP_RATIO / ratio
        if ratio < MIN_RATIO:
            return "FLOOR", MIN_RATIO / ratio
        return "KEEP", 1.0

    actions = base.apply(lambda r: calc_action(r), axis=1)
    base[["ACTION", "FACTOR"]] = pd.DataFrame(actions.tolist(), index=base.index)

    prev_df = prev_df.copy()
    key_cols = ["GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO"]

    # rescale (cap/floor)
    rescale = base[base["ACTION"].isin(["CAP", "FLOOR"])]
    if not rescale.empty:
        prev_df = prev_df.merge(rescale[key_cols + ["FACTOR"]], on=key_cols, how="left")
        prev_df["FACTOR"] = prev_df["FACTOR"].fillna(1.0)
        prev_df["QTDE_PREVISTA"] = prev_df["QTDE_PREVISTA"] * prev_df["FACTOR"]
        prev_df.drop(columns=["FACTOR"], inplace=True)

    # fill zeros
    fill = base[base["ACTION"] == "FILL_ZERO"]
    if not fill.empty:
        fill_map = fill.set_index(key_cols)[["QTDE_ULT6M", "MESES_ULT6M"]]
        fill_map["MESES_ULT6M"] = fill_map["MESES_ULT6M"].replace(0, 6)
        fill_map["FILL_VAL"] = fill_map["QTDE_ULT6M"] / fill_map["MESES_ULT6M"]
        prev_df = prev_df.merge(fill_map[["FILL_VAL"]], left_on=key_cols, right_index=True, how="left")
        mask_fill = prev_df["FILL_VAL"].notna()
        prev_df.loc[mask_fill, "QTDE_PREVISTA"] = prev_df.loc[mask_fill, "FILL_VAL"]
        prev_df.drop(columns=["FILL_VAL"], inplace=True)

    return prev_df


# ===================== FUNÇÃO PRINCIPAL =====================

def main():
    t0 = time.time()
    log("Iniciando Etapa 2 - Previsao 2026")

    # 1) carregar bases
    vendas = carregar_vendas(PATH_VENDAS)
    estoque = carregar_estoque(PATH_ESTOQUE)
    max_data_venda = vendas["DATA_VENDA"].max()

    log(f"Vendas lidas: {len(vendas):,} linhas")
    log(f"Estoque lido: {len(estoque):,} linhas")
    log(f"Ultima data de venda carregada: {max_data_venda.date() if pd.notna(max_data_venda) else 'sem data'}")

    # 2) preparar base agregada (mensal + estoque)
    df_base = preparar_bases(vendas, estoque)
    log(f"Linhas na base agregada (mensal + estoque): {len(df_base):,}")

    if df_base.empty:
        log("❌ Base agregada vazia. Encerrando.")
        return

    # 3) modelar cada série
    prev_list = []
    grupos = (
        df_base[["GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO"]]
        .drop_duplicates()
        .sort_values(["GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO"])
    )

    log(f"Séries a modelar: {len(grupos):,}")

    for _, row in tqdm(grupos.iterrows(), total=len(grupos), desc="Modelando séries"):
        griffe = row["GRIFFE"]
        linha = row["LINHA"]
        grupo = row["GRUPO_PRODUTO"]
        faixa = row["FAIXA_PRECO"]

        df_serie = df_base[
            (df_base["GRIFFE"] == griffe)
            & (df_base["LINHA"] == linha)
            & (df_base["GRUPO_PRODUTO"] == grupo)
            & (df_base["FAIXA_PRECO"] == faixa)
        ].copy()

        prev_serie = modelar_serie(df_serie, griffe, linha, grupo, faixa, max_data_venda=max_data_venda)
        if not prev_serie.empty:
            prev_list.append(prev_serie)

    if prev_list:
        prev_df = pd.concat(prev_list, ignore_index=True)
    else:
        prev_df = pd.DataFrame(columns=[
            "GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO",
            "DATA_REF", "ANO_MES", "QTDE_PREVISTA"
        ])

    log(f"Linhas de previsão geradas: {len(prev_df):,}")

    # 4) preço médio aplicado, situação de referência e valor previsto
    prev_df = calcular_preco_e_desconto(vendas, prev_df)

    # 4.1) ajustes de piso/cap por série usando histórico recente
    prev_df = ajustar_series_por_hist(vendas, prev_df)

    # 5) alocação por filial
    aloc_df = alocar_por_filial(vendas, prev_df)

    # 6) salvar em Excel
    log(f"Salvando resultados em: {OUT_XLSX}")
    with pd.ExcelWriter(OUT_XLSX) as writer:
        prev_out = prev_df[[
            "GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO",
            "ANO_MES", "QTDE_PREVISTA",
            "PRECO_MEDIO_APLICADO", "SITUACAO_REF",
            "DESCONTO_SUGERIDO", "VAL_VENDA_PREVISTA"
        ]].copy()
        prev_out.to_excel(writer, sheet_name="Previsao_2026", index=False)

        aloc_out = aloc_df[[
            "GRIFFE", "LINHA", "GRUPO_PRODUTO", "FAIXA_PRECO",
            "ANO_MES", "FILIAL_2", "QTDE_ALOCADA"
        ]].copy()
        aloc_out.to_excel(writer, sheet_name="Alocacao_2026", index=False)

    log(f"✅ Etapa 2 concluída em {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
