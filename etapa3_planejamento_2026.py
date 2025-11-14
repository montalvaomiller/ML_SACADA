# etapa3_planejamento_2026.py
import pandas as pd
import numpy as np
from pathlib import Path

# ================== CONFIG ==================
BASE_DIR  = Path(r"C:\Users\monta\OneDrive\Documentos\Meta\METAxSACADA\ML-Sacada")
SAIDAS    = BASE_DIR / "saidas"
IN_PREV   = SAIDAS / "previsao_2026.xlsx"              # gerado na Etapa 2 (com FAIXA_PRECO)
OUT_XLSX  = SAIDAS / "planejamento_comercial_2026.xlsx"

# Margem alvo padrão (se não houver tabela externa)
MARGEM_ALVO_DEFAULT = 0.55

# Tabela externa de margens (opcional). Pode incluir FAIXA_PRECO também.
# Exemplo de cabeçalho: GRIFFE,LINHA,GRUPO_PRODUTO,FAIXA_PRECO,MARGEM_ALVO
MARGENS_EXTERNAS_CSV = None  # ex.: SAIDAS / "margens_alvo.csv"

# ================== HELPERS ==================
def ensure_numeric(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _pick_keys_for_merge(df_left: pd.DataFrame, df_right: pd.DataFrame, preferred):
    """
    Escolhe dinamicamente as chaves de junção com base no que existe nas duas tabelas,
    respeitando a ordem de 'preferred'.
    """
    common = [k for k in preferred if (k in df_left.columns and k in df_right.columns)]
    if not common:
        # fallback mínimo: usa o que houver em comum
        common = [c for c in df_left.columns if c in df_right.columns]
        # e mantém apenas colunas-chave típicas, se existirem
        preferred_set = {"GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","ANO_MES"}
        common = [c for c in common if c in preferred_set] or common[:1]
    return common

def _safe_str_strip_upper(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

# ================== I/O ==================
def carregar_previsao_alocacao(path_xlsx: Path):
    """Lê as abas essenciais da Etapa 2, com proteção a colunas faltantes."""
    prev = pd.read_excel(path_xlsx, sheet_name="Previsao_2026")
    aloc = pd.read_excel(path_xlsx, sheet_name="Alocacao_2026")

    # Tipos
    prev = ensure_numeric(prev, ["QTDE_PREVISTA", "PRECO_MEDIO_APLICADO", "VAL_VENDA_PREVISTA", "DESCONTO_SUGERIDO"])
    aloc = ensure_numeric(aloc, ["QTDE_ALOCADA"])

    # Normaliza chaves (sem upper para não romper nomes; apenas strip)
    key_cols_prev = [c for c in ["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","ANO_MES"] if c in prev.columns]
    key_cols_aloc = [c for c in ["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","FILIAL_2","ANO_MES"] if c in aloc.columns]
    prev = _safe_str_strip_upper(prev, key_cols_prev)
    aloc = _safe_str_strip_upper(aloc, key_cols_aloc)

    return prev, aloc

def carregar_margens_externas(csv_path):
    if not csv_path or not Path(csv_path).exists():
        return None
    m = pd.read_csv(csv_path, dtype=str)
    m = ensure_numeric(m, ["MARGEM_ALVO"])
    for c in ["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"]:
        if c in m.columns:
            m[c] = m[c].astype(str).str.strip()
    return m

# ================== REGRAS DE NEGÓCIO ==================
def aplicar_margem_alvo(plano, margens_externas=None):
    # margem padrão
    plano["MARGEM_ALVO"] = MARGEM_ALVO_DEFAULT

    # aplica margens externas se houver (prioriza regra mais específica, com FAIXA_PRECO)
    if margens_externas is not None and not margens_externas.empty:
        if set(["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"]).issubset(margens_externas.columns):
            join_keys = ["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"]
        else:
            join_keys = [c for c in ["GRIFFE","LINHA","GRUPO_PRODUTO"] if c in margens_externas.columns]
        if join_keys:
            use_cols = [c for c in (join_keys + ["MARGEM_ALVO"]) if c in margens_externas.columns]
            plano = plano.merge(margens_externas[use_cols], on=join_keys, how="left", suffixes=("","_EXT"))
            if "MARGEM_ALVO_EXT" in plano.columns:
                plano["MARGEM_ALVO"] = np.where(plano["MARGEM_ALVO_EXT"].notna(), plano["MARGEM_ALVO_EXT"], plano["MARGEM_ALVO"])
                plano.drop(columns=["MARGEM_ALVO_EXT"], inplace=True, errors="ignore")

    # clamp (evita valores fora de [0.05, 0.85])
    plano["MARGEM_ALVO"] = plano["MARGEM_ALVO"].clip(lower=0.05, upper=0.85)
    return plano

def estimar_custo(plano):
    """
    Se não houver custo no input, usa fallback = 50% do preço médio.
    Normaliza escala usando razão custo/preço por (GRIFFE, LINHA, GRUPO_PRODUTO, FAIXA_PRECO).
    """
    if "CUSTO_UNIT_HIST" not in plano.columns:
        plano["CUSTO_UNIT_HIST"] = np.nan

    # fallback direto
    preco_col = "PRECO_MEDIO_APLICADO" if "PRECO_MEDIO_APLICADO" in plano.columns else None
    if preco_col is None:
        # se por algum motivo não existir, cria com 0 para não quebrar
        plano["PRECO_MEDIO_APLICADO"] = 0.0
        preco_col = "PRECO_MEDIO_APLICADO"

    plano["CUSTO_UNIT_HIST"] = np.where(
        plano["CUSTO_UNIT_HIST"].isna(),
        plano[preco_col] * 0.50,
        plano["CUSTO_UNIT_HIST"]
    )

    eps = 1e-12
    plano["R_COST_PRICE"] = plano["CUSTO_UNIT_HIST"] / (plano[preco_col] + eps)

    grp_keys = [c for c in ["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"] if c in plano.columns]
    if grp_keys:
        med = plano.groupby(grp_keys, dropna=False)["R_COST_PRICE"].median().rename("R_MED")
        plano = plano.merge(med, on=grp_keys, how="left")
        mask_100 = plano["R_MED"] > 20
        plano.loc[mask_100, "CUSTO_UNIT_HIST"] = plano.loc[mask_100, "CUSTO_UNIT_HIST"] / 100.0

        plano["R_COST_PRICE"] = plano["CUSTO_UNIT_HIST"] / (plano[preco_col] + eps)
        med2 = plano.groupby(grp_keys, dropna=False)["R_COST_PRICE"].median().rename("R_MED2")
        plano = plano.drop(columns=["R_MED"], errors="ignore").merge(med2, on=grp_keys, how="left")
        mask_10 = plano["R_MED2"] > 20
        plano.loc[mask_10, "CUSTO_UNIT_HIST"] = plano.loc[mask_10, "CUSTO_UNIT_HIST"] / 10.0
        plano.drop(columns=["R_COST_PRICE","R_MED2"], inplace=True, errors="ignore")
    else:
        # sem chaves para normalizar, apenas remove colunas auxiliares
        plano.drop(columns=["R_COST_PRICE"], inplace=True, errors="ignore")

    return plano

def calcular_precos_e_valores(plano):
    """
    Calcula PRECO_SUGERIDO, VAL_BASE, VAL_AJUST, LUCRO_AJUST com guardas.
    """
    eps = 1e-9
    if "MARGEM_ALVO" not in plano.columns:
        plano["MARGEM_ALVO"] = MARGEM_ALVO_DEFAULT
    if "QTDE_ALOCADA" not in plano.columns:
        plano["QTDE_ALOCADA"] = 0

    # preço sugerido para meta de margem
    plano["PRECO_SUGERIDO"] = plano["CUSTO_UNIT_HIST"] / (1 - plano["MARGEM_ALVO"] + eps)
    # guarda 1: não pode ser abaixo de custo + 5%
    plano["PRECO_SUGERIDO"] = np.maximum(plano["PRECO_SUGERIDO"], plano["CUSTO_UNIT_HIST"] * 1.05)

    # guarda 2: teto = 3x preço histórico (se existir)
    ref_col = "PRECO_MEDIO_APLICADO" if "PRECO_MEDIO_APLICADO" in plano.columns else None
    if ref_col:
        limite_up = plano[ref_col] * 3.0
        mask_expl = plano["PRECO_SUGERIDO"] > limite_up
        plano.loc[mask_expl, "PRECO_SUGERIDO"] = limite_up
        if "STATUS_PROD" not in plano.columns:
            plano["STATUS_PROD"] = ""
        plano.loc[mask_expl, "STATUS_PROD"] = "REVER PREÇO (OUTLIER)"

    # valores base e ajustado
    preco_ref = ref_col if ref_col else "PRECO_SUGERIDO"  # se não tiver histórico, usa o sugerido
    plano["VAL_BASE"]    = plano["QTDE_ALOCADA"] * plano[preco_ref]
    plano["VAL_AJUST"]   = plano["QTDE_ALOCADA"] * plano["PRECO_SUGERIDO"]
    plano["CUSTO_TOTAL"] = plano["QTDE_ALOCADA"] * plano["CUSTO_UNIT_HIST"]
    plano["LUCRO_BASE"]  = plano["VAL_BASE"]  - plano["CUSTO_TOTAL"]
    plano["LUCRO_AJUST"] = plano["VAL_AJUST"] - plano["CUSTO_TOTAL"]

    # guarda 3: lucro não pode ser maior que a receita
    bad = plano["LUCRO_AJUST"] > plano["VAL_AJUST"]
    plano.loc[bad, "LUCRO_AJUST"] = np.nan
    return plano

def classificar_status(plano):
    eps = 1e-9
    plano["MARGEM_PROJETADA"] = (plano["PRECO_SUGERIDO"] - plano["CUSTO_UNIT_HIST"]) / (plano["PRECO_SUGERIDO"] + eps)

    base_status = np.select(
        [
            (plano["MARGEM_PROJETADA"] >= plano["MARGEM_ALVO"]) & (plano["QTDE_ALOCADA"] > 0),
            (plano["QTDE_ALOCADA"] <= 0)
        ],
        ["PRODUZIR", "BAIXA PRIORIDADE"],
        default="REVER PREÇO"
    )

    if "STATUS_PROD" not in plano.columns:
        plano["STATUS_PROD"] = ""
    plano["STATUS_PROD"] = np.where(
        plano["STATUS_PROD"].eq("REVER PREÇO (OUTLIER)"),
        "REVER PREÇO (OUTLIER)",
        base_status
    )

    q33, q66 = plano["VAL_AJUST"].fillna(0).quantile([0.33, 0.66])
    plano["PRIORIDADE"] = np.select(
        [
            plano["VAL_AJUST"] <= q33,
            (plano["VAL_AJUST"] > q33) & (plano["VAL_AJUST"] <= q66),
            plano["VAL_AJUST"] > q66
        ],
        ["BAIXA","MÉDIA","ALTA"],
        default="MÉDIA"
    )
    return plano

def gerar_outliers(plano):
    out = plano[
        (("PRECO_MEDIO_APLICADO" in plano.columns) & (plano["PRECO_SUGERIDO"] >= plano["PRECO_MEDIO_APLICADO"] * 3.0)) |
        (("PRECO_MEDIO_APLICADO" in plano.columns) & (plano["CUSTO_UNIT_HIST"] >= plano["PRECO_MEDIO_APLICADO"] * 5.0)) |
        (plano["LUCRO_AJUST"] > plano["VAL_AJUST"])
    ].copy()
    return out

def resumos(plano):
    # Resumo por filial
    grp_cols_filial = [c for c in ["ANO_MES","GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","FILIAL_2"] if c in plano.columns]
    resumo_filial = plano.groupby(grp_cols_filial, dropna=False).agg(
        QTDE=("QTDE_ALOCADA","sum"),
        VAL_BASE=("VAL_BASE","sum"),
        VAL_AJUST=("VAL_AJUST","sum"),
        LUCRO_BASE=("LUCRO_BASE","sum"),
        LUCRO_AJUST=("LUCRO_AJUST","sum")
    ).reset_index()

    # Resumo geral por faixa
    grp_cols_geral = [c for c in ["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"] if c in plano.columns]
    if not grp_cols_geral:
        grp_cols_geral = ["FAIXA_PRECO"] if "FAIXA_PRECO" in plano.columns else []
    resumo_geral = plano.groupby(grp_cols_geral, dropna=False).agg(
        QTDE=("QTDE_ALOCADA","sum"),
        VAL_AJUST=("VAL_AJUST","sum"),
        MARGEM_MEDIA=("MARGEM_ALVO","mean"),
        PRECO_MEDIO=("PRECO_SUGERIDO","mean")
    ).reset_index()
    return resumo_filial, resumo_geral

# ================== MAIN ==================
def main():
    SAIDAS.mkdir(parents=True, exist_ok=True)
    print("🧭 Etapa 3 — Planejamento Comercial 2026 (robusto e dinâmico)")

    # 1) Ler previsão e alocação
    prev, aloc = carregar_previsao_alocacao(IN_PREV)

    # 2) Montar base unificada por item/filial/mês (incluindo FAIXA_PRECO)
    expected_keys = ["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","ANO_MES"]
    keys_prev = [k for k in expected_keys if k in prev.columns]
    keys_aloc = [k for k in expected_keys if k in aloc.columns]

    join_keys = _pick_keys_for_merge(aloc, prev, preferred=expected_keys)
    print(f"🔑 Chaves de junção: {join_keys}")

    cols_prev_keep = list(dict.fromkeys(join_keys + [c for c in ["QTDE_PREVISTA","PRECO_MEDIO_APLICADO","VAL_VENDA_PREVISTA","DESCONTO_SUGERIDO"] if c in prev.columns]))
    cols_aloc_keep = list(dict.fromkeys(join_keys + [c for c in ["FILIAL_2","QTDE_ALOCADA"] if c in aloc.columns]))

    missing_prev = [c for c in cols_prev_keep if c not in prev.columns]
    missing_aloc = [c for c in cols_aloc_keep if c not in aloc.columns]
    if missing_prev:
        print(f"[WARN] Colunas ausentes em Previsao_2026: {missing_prev}")
    if missing_aloc:
        print(f"[WARN] Colunas ausentes em Alocacao_2026: {missing_aloc}")

    base = aloc[cols_aloc_keep].merge(prev[cols_prev_keep], on=join_keys, how="left")
    if base.empty:
        print("[WARN] Merge vazio. Verifique chaves/colunas das abas Previsao_2026 e Alocacao_2026.")

    # 3) Margem-alvo (aceita regra por faixa caso exista no CSV)
    marg_ext = carregar_margens_externas(MARGENS_EXTERNAS_CSV)
    base = aplicar_margem_alvo(base, marg_ext)

    # 4) Estimar custo histórico (com normalizador de escala por faixa)
    base = estimar_custo(base)

    # 5) Calcular preços/valores com guardas
    base = calcular_precos_e_valores(base)

    # 6) Classificar status e prioridade
    base = classificar_status(base)

    # 7) Outliers
    outliers = gerar_outliers(base)

    # 8) Resumos
    resumo_filial, resumo_geral = resumos(base)

    # 9) Salvar Excel
    try:
        import openpyxl  # garantir engine
        with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
            base.to_excel(xw, sheet_name="Planejamento_Comercial_2026", index=False)
            resumo_filial.to_excel(xw, sheet_name="Resumo_por_Filial", index=False)
            resumo_geral.to_excel(xw, sheet_name="Resumo_Geral_Faixa", index=False)
            outliers.to_excel(xw, sheet_name="Outliers", index=False)
        print(f"[OK] Etapa 3 concluída: {OUT_XLSX}")
    except Exception as e:
        print(f"[INFO] openpyxl indisponível ({e}). Salvando CSVs.")
        base.to_csv(SAIDAS / "planejamento_comercial_2026.csv", index=False, encoding="utf-8")
        resumo_filial.to_csv(SAIDAS / "resumo_filial_2026.csv", index=False, encoding="utf-8")
        resumo_geral.to_csv(SAIDAS / "resumo_geral_faixa_2026.csv", index=False, encoding="utf-8")
        outliers.to_csv(SAIDAS / "planejamento_outliers_2026.csv", index=False, encoding="utf-8")
        print(f"[OK] CSVs salvos em: {SAIDAS}")

if __name__ == "__main__":
    main()
