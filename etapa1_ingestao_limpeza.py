# etapa1_ingestao_limpeza.py
import pandas as pd
from pathlib import Path
import sys
import time
import traceback
import re
import numpy as np

# ================== I/O & ENCODING ==================
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ================== CONFIG ==================
BASE_DIR  = Path(r"C:\Users\monta\OneDrive\Documentos\Meta\METAxSACADA\ML-Sacada")
DADOS_DIR = BASE_DIR / "dados"
OUT_DIR   = BASE_DIR / "saidas"

# Colunas esperadas (usaremos apenas as que existirem no arquivo)
COLS = [
    "DATA_VENDA","CANAL","FILIAL_2","GRIFFE","COLECAO","LINHA",
    "GRUPO_PRODUTO","ANO_MES","VAL_VENDA_BRUTA","VAL_VENDA",
    "VAL_ORIGINAL","VAL_CUSTO","QTDE","DESCONTO","SITUACAO"  # <- NOVO
]

# ================== HELPERS ==================
def to_float_auto(s: pd.Series) -> pd.Series:
    """Converte strings numéricas (pt/EN) em float."""
    def parse_one(x):
        x = "" if x is None else str(x).strip()
        if x == "" or x.lower() in {"nan","none"}:
            return np.nan
        x = re.sub(r"[^0-9.,\-]", "", x)
        if "," in x and "." not in x:
            return float(x.replace(".", "").replace(",", "."))
        if "." in x and "," not in x:
            return float(x)
        last_c = x.rfind(","); last_d = x.rfind(".")
        if last_c > last_d:  # vírgula é decimal
            return float(x.replace(".", "").replace(",", "."))
        else:                # ponto é decimal
            return float(x.replace(",", ""))
    return s.apply(parse_one)

def escolher_arquivo() -> Path:
    """1) Usa caminho passado em argv; 2) senão, pega o .xlsx mais recente em /dados."""
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            print(f"📂 Arquivo de entrada (via argumento): {p}")
            return p
        raise FileNotFoundError(f"Arquivo passado por argumento não existe: {p}")
    candidatos = sorted(DADOS_DIR.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidatos:
        raise FileNotFoundError(f"Nenhum .xlsx encontrado em {DADOS_DIR}")
    print(f"📂 Arquivo de entrada (auto): {candidatos[0].name}")
    print(f"   Caminho: {candidatos[0]}")
    return candidatos[0]

def _normalizar_situacao(col: pd.Series) -> pd.Series:
    """Normaliza a coluna SITUACAO para valores canônicos."""
    mapa = {
        "ATUAL":"ATUAL",
        "OFF":"OFF",
        "ANTERIOR":"ANTERIOR",
        "SEM INFO":"SEM INFO",
        "SEM_INFO":"SEM INFO",
        "SEM-INFO":"SEM INFO",
        "SEMINFO":"SEM INFO",
        "SEM INF0":"SEM INFO",
        "SEM INF0":"SEM INFO",
        "SEM INF0RMACAO":"SEM INFO",
        "":"SEM INFO", "NAN":"SEM INFO", "NONE":"SEM INFO"
    }
    return col.map(lambda x: mapa.get(x, x)).fillna("SEM INFO")

def limpar_padronizar(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza colunas, tipos e remove trocas (QTDE <= 0)."""
    keep = [c for c in COLS if c in df.columns]
    if not keep:
        return pd.DataFrame(columns=[c.upper() for c in COLS])

    df = df[keep].copy()
    df.columns = [c.upper() for c in df.columns]

    # texto
    for c in ["CANAL","FILIAL_2","GRIFFE","COLECAO","LINHA","GRUPO_PRODUTO","SITUACAO"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.upper()

    # normalização de SITUACAO
    if "SITUACAO" in df.columns:
        df["SITUACAO"] = _normalizar_situacao(df["SITUACAO"])
        # flags úteis (p/ modelos e filtros)
        df["IS_ATUAL"]    = (df["SITUACAO"] == "ATUAL").astype(int)
        df["IS_ANTERIOR"] = (df["SITUACAO"] == "ANTERIOR").astype(int)
        df["IS_OFF"]      = (df["SITUACAO"] == "OFF").astype(int)
        df["TREINO_MAIN"] = df["SITUACAO"].isin(["ATUAL","ANTERIOR"]).astype(int)

    # datas
    if "DATA_VENDA" in df.columns:
        df["DATA_VENDA"] = pd.to_datetime(df["DATA_VENDA"], dayfirst=True, errors="coerce")

    # ANO_MES
    if "ANO_MES" in df.columns:
        df["ANO_MES"] = df["ANO_MES"].astype(str).str.extract(r"(\d{6})")[0]
    else:
        df["ANO_MES"] = pd.to_datetime(df.get("DATA_VENDA", pd.NaT), errors="coerce").dt.strftime("%Y%m")

    # numéricos
    for c in ["VAL_VENDA_BRUTA","VAL_VENDA","VAL_ORIGINAL","VAL_CUSTO"]:
        if c in df.columns:
            df[c] = to_float_auto(df[c])

    # quantidades e desconto
   # QTDE
    if "QTDE" in df.columns:
        df["QTDE"] = pd.to_numeric(df["QTDE"], errors="coerce").fillna(0).astype(int)
    else:
        df["QTDE"] = 0

    # DESCONTO
    if "DESCONTO" in df.columns:
        df["DESCONTO"] = pd.to_numeric(df["DESCONTO"], errors="coerce").fillna(0.0)
    else:
        df["DESCONTO"] = 0.0


    # remover trocas/devoluções
    df = df[df["QTDE"] > 0].copy()

    # cálculos auxiliares (unitários)
    df["PRECO_UNIT"]        = (df["VAL_VENDA"]     / df["QTDE"]).where(df["QTDE"] != 0)
    df["PRECO_CHEIO_UNIT"]  = (df["VAL_ORIGINAL"]  / df["QTDE"]).where(df["QTDE"] != 0)
    df["CUSTO_TOTAL"]       = df["VAL_CUSTO"] * df["QTDE"]
    df["DESC_VAL"]          = (df["VAL_ORIGINAL"] - df["VAL_VENDA"]).clip(lower=0)
    df["MARGEM_UNIT"]       = df["PRECO_UNIT"] - df["VAL_CUSTO"]

    return df.drop_duplicates()

def _faixa_quantil_series(series: pd.Series) -> pd.Series:
    """
    Classifica a série em P1/P2/P3 por tercis (q=3), com fallback quando há pouca variação.
    Usa duplicates='drop' para evitar erros quando quantis se repetem.
    """
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(index=series.index, dtype="object")
    try:
        q = pd.qcut(x, q=3, labels=["P1","P2","P3"], duplicates="drop")
        if q.dtype == "category" and len(q.cat.categories) == 3:
            return q.astype("object")
    except Exception:
        pass
    med = x.median()
    bins = [-np.inf, med, np.inf]
    labels = ["P1","P2"]
    out = pd.cut(x, bins=bins, labels=labels, include_lowest=True)
    return out.astype("object")

def adicionar_faixa_preco(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona FAIXA_PRECO por (LINHA, GRUPO_PRODUTO) usando PRECO_CHEIO_UNIT (VAL_ORIGINAL/QTDE)."""
    if not {"LINHA","GRUPO_PRODUTO","PRECO_CHEIO_UNIT"}.issubset(df.columns):
        df = df.copy()
        df["FAIXA_PRECO"] = np.nan
        return df
    df = df.copy()
    df["FAIXA_PRECO"] = (
        df.groupby(["LINHA","GRUPO_PRODUTO"], dropna=False)["PRECO_CHEIO_UNIT"]
          .transform(_faixa_quantil_series)
    )
    return df

def _agregado_base(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    if not keys:
        return pd.DataFrame(columns=[
            "ANO_MES","FILIAL_2","GRIFFE","COLECAO","LINHA","GRUPO_PRODUTO",
            "QTDE_MES","VAL_VENDA_MES","VAL_ORIG_MES","CUSTO_MES","DESC_MES",
            "PRECO_MEDIO_MES","MARGEM_TOTAL","MARGEM_MEDIA"
        ])
    agg = df.groupby(keys, dropna=False).agg(
        QTDE_MES      = ("QTDE","sum"),
        VAL_VENDA_MES = ("VAL_VENDA","sum"),
        VAL_ORIG_MES  = ("VAL_ORIGINAL","sum"),
        CUSTO_MES     = ("CUSTO_TOTAL","sum"),
        DESC_MES      = ("DESC_VAL","sum")
    ).reset_index()
    agg["PRECO_MEDIO_MES"] = (agg["VAL_VENDA_MES"] / agg["QTDE_MES"]).where(agg["QTDE_MES"] != 0)
    agg["MARGEM_TOTAL"]    = agg["VAL_VENDA_MES"] - agg["CUSTO_MES"]
    agg["MARGEM_MEDIA"]    = (agg["MARGEM_TOTAL"] / agg["QTDE_MES"]).where(agg["QTDE_MES"] != 0)
    return agg

def agregar_mensal(df: pd.DataFrame, include_situacao: bool = False) -> pd.DataFrame:
    """Agrega a base no nível mensal por dimensões principais, opcionalmente com SITUACAO."""
    keys = ["ANO_MES","FILIAL_2","GRIFFE","COLECAO","LINHA","GRUPO_PRODUTO"]
    if include_situacao and "SITUACAO" in df.columns:
        keys.append("SITUACAO")
    keys = [k for k in keys if k in df.columns]
    return _agregado_base(df, keys)

def agregar_mensal_por_faixa(df: pd.DataFrame, include_situacao: bool = False) -> pd.DataFrame:
    """Agregado mensal incluindo FAIXA_PRECO (útil para relatório executivo), opcionalmente com SITUACAO."""
    keys = ["ANO_MES","FILIAL_2","GRIFFE","COLECAO","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"]
    if include_situacao and "SITUACAO" in df.columns:
        keys.append("SITUACAO")
    keys = [k for k in keys if k in df.columns]
    return _agregado_base(df, keys)

# ================== PIPELINE ==================
def main():
    print("🔧 Etapa 1 — Ingestão & Limpeza + FAIXA_PRECO + SITUACAO")
    print("Dica: se aparecer erro de engine, instale:  pip install openpyxl")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    input_xlsx = escolher_arquivo()

    # Abre o Excel e lista abas
    try:
        xls = pd.ExcelFile(input_xlsx, engine="openpyxl")
        sheets = xls.sheet_names
    except Exception:
        print("❌ Falha ao abrir o Excel (engine/openpyxl). Detalhes:")
        traceback.print_exc()
        raise

    print(f"🗂️  Abas encontradas: {', '.join(sheets)}")

    frames = []
    for s in sheets:
        t_aba = time.time()
        try:
            df = pd.read_excel(input_xlsx, sheet_name=s, dtype=str, engine="openpyxl")
            n0 = len(df)
            print(f"   • Lendo aba {s:<15} ... {n0:>7} linhas")
            df["__SHEET__"] = s
            frames.append(df)
            print(f"     ✔ OK ({time.time() - t_aba:.1f}s)")
        except Exception:
            print(f"     ❌ Erro ao ler a aba '{s}'. Detalhes:")
            traceback.print_exc()

    if not frames:
        raise RuntimeError("Nenhuma aba pôde ser lida com sucesso.")

    # Concatena e processa
    raw = pd.concat(frames, ignore_index=True)
    print(f"🔎 Total bruto concatenado: {len(raw):,} linhas")

    try:
        stage  = limpar_padronizar(raw)
        stage  = adicionar_faixa_preco(stage)

        # agregados sem e com SITUACAO
        mensal            = agregar_mensal(stage, include_situacao=False)
        mensal_faixa      = agregar_mensal_por_faixa(stage, include_situacao=False)
        mensal_sit        = agregar_mensal(stage, include_situacao=True)
        mensal_faixa_sit  = agregar_mensal_por_faixa(stage, include_situacao=True)

        print(f"🧽 Após limpeza (sem trocas): {len(stage):,} linhas")
        if "SITUACAO" in stage.columns:
            print("🔎 Distribuição SITUACAO (top 10):")
            print(stage["SITUACAO"].value_counts(dropna=False).head(10))
        print("🏷️  FAIXA_PRECO - amostra:")
        print(stage[["LINHA","GRUPO_PRODUTO","PRECO_CHEIO_UNIT","FAIXA_PRECO"]].head(8))
        print(f"📆 Linhas agregadas (mensal): {len(mensal):,}")
        print(f"📆 Linhas agregadas (mensal por faixa): {len(mensal_faixa):,}")
        print(f"📆 Linhas agregadas (mensal + situação): {len(mensal_sit):,}")
        print(f"📆 Linhas agregadas (mensal por faixa + situação): {len(mensal_faixa_sit):,}")
    except Exception:
        print("❌ Erro durante limpeza/agregação:")
        traceback.print_exc()
        raise

    # Salvar saídas
    try:
        stage.to_csv(OUT_DIR / "staging_consolidado.csv", index=False, encoding="utf-8")
        mensal.to_csv(OUT_DIR / "base_mensal.csv", index=False, encoding="utf-8")
        mensal_faixa.to_csv(OUT_DIR / "base_mensal_por_faixa.csv", index=False, encoding="utf-8")
        # novos arquivos com SITUACAO
        mensal_sit.to_csv(OUT_DIR / "base_mensal_por_situacao.csv", index=False, encoding="utf-8")
        mensal_faixa_sit.to_csv(OUT_DIR / "base_mensal_por_faixa_situacao.csv", index=False, encoding="utf-8")

        print(f"📁 Salvo: {OUT_DIR / 'staging_consolidado.csv'}")
        print(f"📁 Salvo: {OUT_DIR / 'base_mensal.csv'}")
        print(f"📁 Salvo: {OUT_DIR / 'base_mensal_por_faixa.csv'}")
        print(f"📁 Salvo: {OUT_DIR / 'base_mensal_por_situacao.csv'}")
        print(f"📁 Salvo: {OUT_DIR / 'base_mensal_por_faixa_situacao.csv'}")
    except Exception:
        print("❌ Erro ao salvar CSVs:")
        traceback.print_exc()
        raise

    print(f"✅ Etapa 1 concluída em {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
