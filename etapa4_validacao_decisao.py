# etapa4_validacao_decisao.py (hardening + sensível a SITUACAO)
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile, shutil, sys, traceback

# ================== CONFIG ==================
BASE_DIR = Path(r"C:\Users\monta\OneDrive\Documentos\Meta\METAxSACADA\ML-Sacada")
IN_DIR   = BASE_DIR / "saidas"
IN_XLSX  = IN_DIR / "planejamento_comercial_2026.xlsx"
OUT_XLSX = IN_DIR / "validacao_decisao_2026.xlsx"

# Segmento base para normalização
BASE_SEG_KEYS = ["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"]

# pesos para situação (se existir)
PESOS_SITUACAO = {"ATUAL": 1.00, "ANTERIOR": 0.90, "OFF": 0.60, "SEM INFO": 0.70}

# ================== HELPERS ==================
def _safe_max(s: pd.Series, fallback=1.0):
    m = s.max()
    return m if pd.notnull(m) and m > 0 else fallback

def _norm_by_group(df: pd.DataFrame, key_cols, value_col, out_col):
    key_cols = [c for c in key_cols if c in df.columns]
    if not key_cols or value_col not in df.columns:
        df[out_col] = 0.0
        return df
    gmax = df.groupby(key_cols, dropna=False)[value_col].transform(_safe_max)
    df[out_col] = (df[value_col] / gmax).fillna(0.0).clip(0.0, 1.0)
    return df

def _safe_write_excel(path: Path, sheets: dict):
    """Escreve em arquivo temporário e move para o destino no final."""
    tmpdir = Path(tempfile.mkdtemp())
    tmpfile = tmpdir / (path.stem + "_tmp.xlsx")
    try:
        import openpyxl  # garante engine
        with pd.ExcelWriter(tmpfile, engine="openpyxl") as xw:
            for name, df in sheets.items():
                df.to_excel(xw, sheet_name=name, index=False)
        shutil.move(str(tmpfile), str(path))  # move atômico
    finally:
        try: shutil.rmtree(tmpdir, ignore_errors=True)
        except: pass

def _peso_situacao(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series([1.0] * 0)
    s = series.astype(str).str.upper().str.strip()
    return s.map(PESOS_SITUACAO).fillna(0.85)  # fallback levemente penalizado

# ================== PIPELINE ==================
def main():
    IN_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Valida entrada ----
    if not IN_XLSX.exists():
        print(f"❌ Arquivo de entrada não encontrado: {IN_XLSX}")
        sys.exit(1)

    try:
        xl = pd.ExcelFile(IN_XLSX)
        if "Planejamento_Comercial_2026" not in xl.sheet_names:
            print(f"❌ Aba 'Planejamento_Comercial_2026' não existe. Abas: {xl.sheet_names}")
            sys.exit(1)
        df = pd.read_excel(IN_XLSX, sheet_name="Planejamento_Comercial_2026")
    except Exception:
        print("❌ Falha ao ler o Excel de entrada:")
        traceback.print_exc()
        sys.exit(1)

    if df.empty:
        print("❌ A planilha de entrada está vazia (0 linhas). Abortando para evitar 0KB.")
        sys.exit(1)

    # ---- Tipos numéricos (robusto) ----
    num_cols = [
        "QTDE_ALOCADA","CUSTO_UNIT_HIST","MARGEM_ALVO",
        "PRECO_MEDIO_APLICADO","PRECO_SUGERIDO",
        "VAL_BASE","VAL_AJUST","LUCRO_BASE","LUCRO_AJUST"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        else:
            df[c] = 0

    # Padroniza chaves
    for c in ["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","FILIAL_2","ANO_MES","SITUACAO"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # ---- Métricas auxiliares ----
    eps = 1e-9
    # margem real e projetada
    df["MARGEM_REAL"] = np.where(
        df["PRECO_MEDIO_APLICADO"] > 0,
        (df["PRECO_MEDIO_APLICADO"] - df["CUSTO_UNIT_HIST"]) / (df["PRECO_MEDIO_APLICADO"] + eps),
        0
    )
    df["MARGEM_PROJETADA"] = np.where(
        df["PRECO_SUGERIDO"] > 0,
        (df["PRECO_SUGERIDO"] - df["CUSTO_UNIT_HIST"]) / (df["PRECO_SUGERIDO"] + eps),
        0
    )
    df["GAP_MARGEM"] = df["MARGEM_PROJETADA"] - df["MARGEM_ALVO"]

    # deltas de preço
    df["DELTA_PRECO"] = df["PRECO_SUGERIDO"] - df["PRECO_MEDIO_APLICADO"]
    df["VAR_PCT_PRECO"] = np.where(
        df["PRECO_MEDIO_APLICADO"] > 0,
        df["DELTA_PRECO"] / (df["PRECO_MEDIO_APLICADO"] + eps),
        np.nan
    )

    # ---- Status (mantém outlier quando marcado) ----
    base_status = np.select(
        [
            (df["MARGEM_PROJETADA"] >= df["MARGEM_ALVO"]) & (df["QTDE_ALOCADA"] > 0),
            (df["QTDE_ALOCADA"] <= 0)
        ],
        ["PRODUZIR", "BAIXA PRIORIDADE"],
        default="REVER PREÇO"
    )
    if "STATUS_PROD" not in df.columns:
        df["STATUS_PROD"] = base_status
    else:
        df["STATUS_PROD"] = np.where(
            df["STATUS_PROD"].astype(str).eq("REVER PREÇO (OUTLIER)"),
            "REVER PREÇO (OUTLIER)",
            base_status
        )

    # ---- Normalizações por segmento (dinâmico) ----
    seg_keys = [c for c in BASE_SEG_KEYS if c in df.columns]
    df = _norm_by_group(df, seg_keys, "QTDE_ALOCADA", "QTDE_NORM_SEG")
    df = _norm_by_group(df, seg_keys, "LUCRO_AJUST", "LUCRO_NORM_SEG")

    # ---- Peso por SITUACAO no SCORE (se houver) ----
    if "SITUACAO" in df.columns:
        df["PESO_SITUACAO"] = _peso_situacao(df["SITUACAO"])
    else:
        df["PESO_SITUACAO"] = 1.0

    # ---- SCORE & prioridade ----
    # Base: margem (40%), volume (30%), lucro (30%), ajustado por SITUACAO
    score_base = (df["MARGEM_PROJETADA"].clip(0,1) * 0.4) + (df["QTDE_NORM_SEG"] * 0.3) + (df["LUCRO_NORM_SEG"] * 0.3)
    df["SCORE"] = (score_base * df["PESO_SITUACAO"]).clip(0, 1.5)

    if df["SCORE"].notna().sum() >= 3:
        q33, q66 = df["SCORE"].quantile([0.33, 0.66])
    else:
        q33, q66 = 0.33, 0.66
    df["PRIORIDADE"] = np.select(
        [
            df["SCORE"] <= q33,
            (df["SCORE"] > q33) & (df["SCORE"] <= q66),
            df["SCORE"] > q66
        ],
        ["BAIXA", "MÉDIA", "ALTA"],
        default="MÉDIA"
    )

    # ---- Resumos ----
    resumo_griffe_linha = df.groupby(
        [c for c in ["GRIFFE","LINHA"] if c in df.columns],
        dropna=False
    ).agg(
        QTDE_TOTAL=("QTDE_ALOCADA","sum"),
        RECEITA_TOTAL=("VAL_AJUST","sum"),
        LUCRO_TOTAL=("LUCRO_AJUST","sum"),
        MARGEM_MEDIA=("MARGEM_PROJETADA","mean"),
        PCT_PRODUZIR=("STATUS_PROD", lambda x: (x == "PRODUZIR").mean() * 100),
        PCT_REVER_PRECO=("STATUS_PROD", lambda x: (x.astype(str).str.contains("REVER PREÇO")).mean() * 100)
    ).reset_index()

    resumo_faixa = df.groupby(
        [c for c in ["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"] if c in df.columns],
        dropna=False
    ).agg(
        QTDE_TOTAL=("QTDE_ALOCADA","sum"),
        RECEITA_TOTAL=("VAL_AJUST","sum"),
        LUCRO_TOTAL=("LUCRO_AJUST","sum"),
        PRECO_SUG_MEDIO=("PRECO_SUGERIDO","mean"),
        MARGEM_MEDIA=("MARGEM_PROJETADA","mean"),
        PCT_PRODUZIR=("STATUS_PROD", lambda x: (x == "PRODUZIR").mean() * 100)
    ).reset_index()

    resumo_filial_faixa = df.groupby(
        [c for c in ["FILIAL_2","GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"] if c in df.columns],
        dropna=False
    ).agg(
        QTDE_TOTAL=("QTDE_ALOCADA","sum"),
        RECEITA_TOTAL=("VAL_AJUST","sum"),
        LUCRO_TOTAL=("LUCRO_AJUST","sum"),
        MARGEM_MEDIA=("MARGEM_PROJETADA","mean")
    ).reset_index()

    resumo_prioridade = df.groupby(
        [c for c in ["PRIORIDADE","STATUS_PROD","FAIXA_PRECO"] if c in df.columns],
        dropna=False
    ).agg(
        ITENS=("QTDE_ALOCADA","count"),
        QTDE_TOTAL=("QTDE_ALOCADA","sum"),
        VAL_AJUST_TOTAL=("VAL_AJUST","sum")
    ).reset_index()

    # ---- Alertas ----
    alertas = df[df["MARGEM_PROJETADA"] < df["MARGEM_ALVO"]][
        [c for c in [
            "GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","FILIAL_2","ANO_MES","SITUACAO",
            "MARGEM_PROJETADA","MARGEM_ALVO","GAP_MARGEM",
            "QTDE_ALOCADA","PRECO_MEDIO_APLICADO","PRECO_SUGERIDO","DELTA_PRECO","VAR_PCT_PRECO",
            "VAL_AJUST"
        ] if c in df.columns]
    ].sort_values([c for c in ["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","FILIAL_2","ANO_MES"] if c in df.columns])

    # ---- Top prioridades ----
    top_prior = (
        df.sort_values(["PRIORIDADE","SCORE","VAL_AJUST"], ascending=[True, False, False])
          .query("PRIORIDADE == 'ALTA'")
          .copy()
    )

    # ---- Consistência ----
    if set(["GRIFFE","LINHA","GRUPO_PRODUTO"]).issubset(df.columns):
        total_sem_faixa = df.groupby(["GRIFFE","LINHA","GRUPO_PRODUTO"], dropna=False)["QTDE_ALOCADA"].sum().rename("QTDE_TOTAL_NO_FAIXA")
        total_com_faixa = (
            df.groupby(["GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO"], dropna=False)["QTDE_ALOCADA"].sum()
              .groupby(level=[0,1,2]).sum().rename("QTDE_TOTAL_COM_FAIXA")
        )
        consist = total_sem_faixa.to_frame().join(total_com_faixa, how="outer").reset_index()
        consist["DELTA_QTDE"] = (consist["QTDE_TOTAL_COM_FAIXA"] - consist["QTDE_TOTAL_NO_FAIXA"]).fillna(0)
    else:
        consist = pd.DataFrame(columns=["QTDE_TOTAL_NO_FAIXA","QTDE_TOTAL_COM_FAIXA","DELTA_QTDE"])

    # ---- Dicionário de campos ----
    dict_campos = pd.DataFrame({
        "CAMPO": [
            "MARGEM_REAL","MARGEM_PROJETADA","GAP_MARGEM",
            "DELTA_PRECO","VAR_PCT_PRECO",
            "QTDE_NORM_SEG","LUCRO_NORM_SEG","PESO_SITUACAO","SCORE","PRIORIDADE"
        ],
        "DESCRICAO": [
            "Margem usando preço médio aplicado.",
            "Margem usando preço sugerido.",
            "Diferença (proj - alvo).",
            "PRECO_SUGERIDO - PRECO_MEDIO_APLICADO.",
            "Variação percentual do preço vs aplicado.",
            "QTDE normalizada no segmento (0..1).",
            "Lucro ajustado normalizado no segmento (0..1).",
            "Peso de situação (ATUAL, ANTERIOR, OFF...).",
            "Rank composto (margem/qtde/lucro * peso situação).",
            "Classe por tercis do SCORE."
        ]
    })

    # ---- Gravação blindada ----
    try:
        _safe_write_excel(
            OUT_XLSX,
            {
                "Decisoes_Produto_2026": df,
                "Resumo_Griffe_Linha_2026": resumo_griffe_linha,
                "Resumo_Faixa_2026": resumo_faixa,
                "Resumo_Filial_Faixa_2026": resumo_filial_faixa,
                "Resumo_Prioridade_2026": resumo_prioridade,
                "Top_Prioridades_2026": top_prior,
                "Alertas_2026": alertas,
                "Check_Consistencia": consist,
                "Dicionario_Campos": dict_campos,
            }
        )
        print(f"[OK] Etapa 4 concluída: {OUT_XLSX}")
    except Exception:
        print("❌ Falha ao gravar o Excel (arquivo pode estar aberto no Excel). Salvando CSVs de fallback…")
        traceback.print_exc()
        df.to_csv(IN_DIR / "validacao_decisoes_produto_2026.csv", index=False, encoding="utf-8-sig")
        resumo_griffe_linha.to_csv(IN_DIR / "validacao_resumo_griffe_linha_2026.csv", index=False, encoding="utf-8-sig")
        resumo_faixa.to_csv(IN_DIR / "validacao_resumo_faixa_2026.csv", index=False, encoding="utf-8-sig")
        resumo_filial_faixa.to_csv(IN_DIR / "validacao_resumo_filial_faixa_2026.csv", index=False, encoding="utf-8-sig")
        resumo_prioridade.to_csv(IN_DIR / "validacao_resumo_prioridade_2026.csv", index=False, encoding="utf-8-sig")
        top_prior.to_csv(IN_DIR / "validacao_top_prioridades_2026.csv", index=False, encoding="utf-8-sig")
        alertas.to_csv(IN_DIR / "validacao_alertas_2026.csv", index=False, encoding="utf-8-sig")
        consist.to_csv(IN_DIR / "validacao_check_consistencia.csv", index=False, encoding="utf-8-sig")
        dict_campos.to_csv(IN_DIR / "validacao_dicionario_campos.csv", index=False, encoding="utf-8-sig")
        print(f"[OK] CSVs salvos em: {IN_DIR}")

if __name__ == "__main__":
    main()
