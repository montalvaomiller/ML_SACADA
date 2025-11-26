import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ============================================================
#  Função de log
# ============================================================

def log(msg: str):
    print(f"[ETAPA 1] {msg}")


# ============================================================
#  Conversão numérica robusta
# ============================================================

def to_float_auto(x):
    """
    Converte string/numero em float tratando:
    - vírgula/ponto
    - espaços
    - valores vazios
    """
    if pd.isna(x):
        return np.nan
    try:
        s = str(x).strip()
        if s == "":
            return np.nan
        # trata formatos tipo '1.234,56' -> '1234.56'
        s = s.replace(".", "").replace(",", ".")
        return float(s)
    except Exception:
        return np.nan


# ============================================================
#  Pegar arquivo mais recente na pasta de vendas
# ============================================================

def get_latest_file(folder: str) -> str:
    files = [
        f
        for f in os.listdir(folder)
        if f.lower().endswith((".xlsx", ".xls", ".csv"))
        and not f.startswith("~$")
    ]
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo de venda encontrado em: {folder}")

    files = sorted(
        files,
        key=lambda x: os.path.getmtime(os.path.join(folder, x))
    )
    latest = files[-1]
    return os.path.join(folder, latest)


# ============================================================
#  Normalizar colunas de texto
# ============================================================

def normalize_str_cols(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.upper()
            )
    return df


# ============================================================
#  Mapeamento de SITUACAO
# ============================================================

def map_situacao(valor: str) -> str:
    if pd.isna(valor):
        return "SEM INFO"
    s = str(valor).upper()

    # Ajuste aqui se quiser mapear mais rótulos
    if any(k in s for k in ["ATUAL", "NOVA", "LANÇAMENTO"]):
        return "ATUAL"
    if any(k in s for k in ["ANTERIOR", "VELHA", "INVERNO ANTERIOR"]):
        return "ANTERIOR"
    if any(k in s for k in ["OFF", "PROMO", "LIQ"]):
        return "OFF"

    return "SEM INFO"


# ============================================================
#  Criar FAIXA_PRECO (P1/P2/P3) por LINHA + GRUPO_PRODUTO
# ============================================================

def faixa_preco_p123(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["FAIXA_PRECO"] = "SEM INFO"

    group_cols = ["LINHA", "GRUPO_PRODUTO"]

    for keys, idx in df.groupby(group_cols).groups.items():
        sub = df.loc[idx]
        preco = sub["PRECO_CHEIO_UNIT"]

        # Se quase não há variedade de preço, usa mediana
        preco_valid = preco.dropna()
        if preco_valid.nunique() < 3:
            med = preco_valid.median()
            df.loc[idx, "FAIXA_PRECO"] = np.where(preco <= med, "P1", "P2")
            continue

        try:
            q = pd.qcut(preco_valid, q=3, labels=["P1", "P2", "P3"], duplicates="drop")
            # Alinha ao índice original
            df.loc[preco_valid.index, "FAIXA_PRECO"] = q.astype(str)
        except Exception:
            med = preco_valid.median()
            df.loc[idx, "FAIXA_PRECO"] = np.where(preco <= med, "P1", "P2")

    return df


# ============================================================
#  INÍCIO DO SCRIPT
# ============================================================

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VENDAS_DIR = os.path.join(BASE_DIR, "Dados de Venda")
    SAIDAS_DIR = os.path.join(BASE_DIR, "saidas")
    os.makedirs(SAIDAS_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Escolher arquivo de vendas (CLI ou mais recente)
    # --------------------------------------------------------
    if len(sys.argv) > 1:
        venda_path = sys.argv[1]
        log(f"Arquivo informado via linha de comando: {venda_path}")
    else:
        venda_path = get_latest_file(VENDAS_DIR)
        log(f"Nenhum arquivo informado. Usando o mais recente em '{VENDAS_DIR}': {os.path.basename(venda_path)}")

    if not os.path.exists(venda_path):
        raise FileNotFoundError(f"Arquivo de vendas não encontrado: {venda_path}")

    # --------------------------------------------------------
    # Leitura
    # --------------------------------------------------------
    log(f"Lendo vendas de: {venda_path}")

    if venda_path.lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(venda_path, sheet_name=None, dtype=str)
        df = pd.concat(sheets.values(), ignore_index=True)
    else:
        # Caso seja CSV
        df = pd.read_csv(venda_path, dtype=str, sep=";", low_memory=False)

    log(f"Linhas carregadas (todas abas): {len(df):,}")

    # --------------------------------------------------------
    # Padronizar nomes de colunas
    # --------------------------------------------------------
    df.columns = df.columns.str.strip().str.upper()

    expected_cols = [
        "DATA_VENDA",
        "CANAL",
        "FILIAL_2",
        "COLECAO",
        "LINHA",
        "GRUPO_PRODUTO",
        "ANO_MES",
        "VAL_VENDA_BRUTA",
        "VAL_VENDA",
        "VAL_ORIGINAL",
        "VAL_CUSTO",
        "QTDE",
        "SITUACAO",
    ]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    # --------------------------------------------------------
    # Normalizar texto
    # --------------------------------------------------------
    df = normalize_str_cols(
        df,
        ["CANAL", "FILIAL_2", "COLECAO", "LINHA", "GRUPO_PRODUTO", "SITUACAO"]
    )

    # --------------------------------------------------------
    # Remover ATACADO
    # --------------------------------------------------------
    antes = len(df)
    df = df[~df["CANAL"].str.contains("ATACADO", na=False)].copy()
    log(f"Removidas {antes - len(df):,} linhas de CANAL contendo 'ATACADO'")

    # --------------------------------------------------------
    # Converter números
    # --------------------------------------------------------
    for col in ["VAL_VENDA_BRUTA", "VAL_VENDA", "VAL_ORIGINAL", "VAL_CUSTO", "QTDE"]:
        df[col] = df[col].apply(to_float_auto)

    # --------------------------------------------------------
    # Converter DATA_VENDA para datetime
    # (mantendo formato padrão do pandas, sem mudar mais nada)
    # --------------------------------------------------------
    df["DATA_VENDA"] = pd.to_datetime(df["DATA_VENDA"], errors="coerce")

    invalid_dates = df["DATA_VENDA"].isna().sum()
    if invalid_dates > 0:
        df = df.dropna(subset=["DATA_VENDA"])
        log(f"Removidas {invalid_dates:,} linhas com DATA_VENDA inválida")

    # --------------------------------------------------------
    # Recalcular ANO_MES (YYYYMM) a partir de DATA_VENDA
    # --------------------------------------------------------
    df["ANO_MES"] = df["DATA_VENDA"].dt.year * 100 + df["DATA_VENDA"].dt.month

    # --------------------------------------------------------
    # Remover QTDE <= 0
    # --------------------------------------------------------
    antes = len(df)
    df = df[df["QTDE"] > 0].copy()
    log(f"Removidas {antes - len(df):,} linhas com QTDE <= 0")

    # --------------------------------------------------------
    # Métricas de preço e custo (v2)
    # --------------------------------------------------------
    # Preço unitário e preço cheio unitário
    df["PRECO_UNIT"] = df["VAL_VENDA"] / df["QTDE"]
    df["PRECO_CHEIO_UNIT"] = np.where(
        df["VAL_ORIGINAL"] > 0,
        df["VAL_ORIGINAL"] / df["QTDE"],
        np.nan,
    )

        # --------------------------------------------------------
    # Métricas de preço e custo (v2)
    # --------------------------------------------------------
    # Preço unitário e preço cheio unitário
    df["PRECO_UNIT"] = df["VAL_VENDA"] / df["QTDE"]
    df["PRECO_CHEIO_UNIT"] = np.where(
        df["VAL_ORIGINAL"] > 0,
        df["VAL_ORIGINAL"] / df["QTDE"],
        np.nan,
    )

    # Mantemos VAL_CUSTO apenas como valor bruto vindo da base.
    # Como a escala de custo não está totalmente confiável e não é usada
    # na Etapa 2, evitamos forçar um cálculo de margem errado aqui.
    # Ainda assim, deixamos colunas de custo/margem para futura evolução.
    df.rename(columns={"VAL_CUSTO": "VAL_CUSTO_BRUTO"}, inplace=True)
    df["CUSTO_TOTAL"] = df["VAL_CUSTO_BRUTO"]
    df["DESC_VAL"] = df["VAL_ORIGINAL"] - df["VAL_VENDA"]
    df["MARGEM_UNIT"] = np.nan


    # --------------------------------------------------------
    # Remover preços inválidos (<= 0)
    # --------------------------------------------------------
    mask_price_bad = (df["PRECO_UNIT"] <= 0) | (df["PRECO_CHEIO_UNIT"] <= 0)
    if mask_price_bad.any():
        log(f"Removidas {mask_price_bad.sum():,} linhas com PRECO_UNIT/PRECO_CHEIO_UNIT <= 0")
        df = df[~mask_price_bad].copy()

    # --------------------------------------------------------
    # Calcular DESCONTO real e DESCONTO_LIMPO
    # --------------------------------------------------------
    mask_val_ok = df["VAL_ORIGINAL"] > 0
    df.loc[mask_val_ok, "DESCONTO"] = 1 - (
        df.loc[mask_val_ok, "VAL_VENDA"] / df.loc[mask_val_ok, "VAL_ORIGINAL"]
    )
    df.loc[~mask_val_ok, "DESCONTO"] = np.nan

    # Corta valores extremos para uso futuro em modelo, se desejar
    df["DESCONTO_LIMPO"] = df["DESCONTO"].clip(lower=0, upper=0.8)

    # --------------------------------------------------------
    # Normalizar SITUACAO + flags
    # --------------------------------------------------------
    df["SITUACAO"] = df["SITUACAO"].apply(map_situacao)

    df["IS_ATUAL"] = (df["SITUACAO"] == "ATUAL").astype(int)
    df["IS_ANTERIOR"] = (df["SITUACAO"] == "ANTERIOR").astype(int)
    df["IS_OFF"] = (df["SITUACAO"] == "OFF").astype(int)
    df["TREINO_MAIN"] = ((df["IS_ATUAL"] == 1) | (df["IS_ANTERIOR"] == 1)).astype(int)

    # --------------------------------------------------------
    # Garantir GRIFFE
    # --------------------------------------------------------
    if "GRIFFE" not in df.columns:
        df["GRIFFE"] = "SACADA"
    else:
        df["GRIFFE"] = df["GRIFFE"].fillna("SACADA")
        df["GRIFFE"] = df["GRIFFE"].replace("", "SACADA")

    df["GRIFFE"] = df["GRIFFE"].astype(str).str.strip().str.upper()

    # --------------------------------------------------------
    # Preencher vazios em chaves principais de forma explícita
    # --------------------------------------------------------
    df["FILIAL_2"] = df["FILIAL_2"].replace("", np.nan).fillna("SEM FILIAL")
    df["LINHA"] = df["LINHA"].replace("", np.nan).fillna("SEM LINHA")
    df["GRUPO_PRODUTO"] = df["GRUPO_PRODUTO"].replace("", np.nan).fillna("SEM GRUPO")

    # --------------------------------------------------------
    # Criar FAIXA_PRECO (P1/P2/P3) por LINHA + GRUPO_PRODUTO
    # usando PRECO_CHEIO_UNIT
    # --------------------------------------------------------
    df = faixa_preco_p123(df)

    # --------------------------------------------------------
    # CHAVE_MODELO para ajuda na Etapa 2/3
    # --------------------------------------------------------
    df["CHAVE_MODELO"] = (
        df["GRIFFE"]
        + " | "
        + df["LINHA"]
        + " | "
        + df["GRUPO_PRODUTO"]
        + " | "
        + df["FAIXA_PRECO"]
    )

    # --------------------------------------------------------
    # Salvar staging
    # --------------------------------------------------------
    staging_path = os.path.join(SAIDAS_DIR, "staging_consolidado.csv")
    df.to_csv(staging_path, index=False, sep=";")
    log(f"staging_consolidado salvo em: {staging_path}")
    log(f"Linhas finais: {len(df):,}")
    log("Etapa 1 finalizada com sucesso.")
