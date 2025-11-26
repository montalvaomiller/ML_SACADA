import pandas as pd
import numpy as np
from pathlib import Path
import re
import sys
import time
import traceback

# ================== CONFIG BÁSICA ==================

BASE_DIR  = Path(__file__).resolve().parent        # pasta ML_SACADA
INPUT_DIR = BASE_DIR / "Dados de Estoque"          # onde estão ESTOQUE_*.XLSX
OUT_DIR   = BASE_DIR / "saidas"                    # mesma pasta de saída da Etapa 1
OUT_DIR.mkdir(parents=True, exist_ok=True)

# usar só fotos de estoque destes dias do mês (padrão: 1 e 15)
KEEP_DAYS = (1, 15)

# caminho da base de vendas (Etapa 1) para alinhar filiais, se existir
STAGING_VENDAS = OUT_DIR / "staging_consolidado.csv"


# ================== FUNÇÃO DE LOG ==================

def log(msg: str):
    print(f"[ETAPA 1B] {msg}")


# ================== HELPERS ==================

def infer_date_from_filename(name: str):
    """
    Tenta inferir a data a partir do nome do arquivo.
    Ex.: ESTOQUE_15072023.XLSX → 15/07/2023
    """
    m = re.search(r"(\d{2})(\d{2})(\d{4})", name)
    if not m:
        return pd.NaT
    d, m_, y = m.groups()
    try:
        return pd.Timestamp(year=int(y), month=int(m_), day=int(d))
    except Exception:
        return pd.NaT


def carregar_arquivo_estoque(path: Path) -> pd.DataFrame:
    """
    Lê um snapshot de estoque e devolve um DataFrame padronizado com:
    - GRIFFE, FILIAL_2, GRUPO_PRODUTO, ESTOQUE
    - DATA_REF (datetime), ANO_MES (YYYYMM), DIA_CORTE (int)
    """
    log(f"Lendo arquivo de estoque: {path.name}")
    df = pd.read_excel(path, dtype={"ESTOQUE": float}, engine="openpyxl")

    # padronizar nomes de colunas
    df.columns = [c.strip().upper() for c in df.columns]

    # garantir colunas essenciais
    required = {"GRIFFE", "FILIAL", "GRUPO_PRODUTO", "ESTOQUE"}
    missing = required - set(df.columns)
    if missing:
        log(f"⚠ Arquivo {path.name} sem colunas obrigatórias: {missing}. Ignorando.")
        return pd.DataFrame(columns=["GRIFFE", "FILIAL_2", "GRUPO_PRODUTO",
                                     "ESTOQUE", "DATA_REF", "ANO_MES", "DIA_CORTE"])

    # texto em maiúsculas / sem espaços extras
    for col in ["GRIFFE", "FILIAL", "GRUPO_PRODUTO"]:
        df[col] = df[col].astype(str).str.strip().str.upper()

    # só griffe SACADA (alinhado com staging_consolidado)
    df = df[df["GRIFFE"] == "SACADA"]
    if df.empty:
        log(f"   → Após filtrar GRIFFE=SACADA, não sobrou nada em {path.name}.")
        return pd.DataFrame(columns=["GRIFFE", "FILIAL_2", "GRUPO_PRODUTO",
                                     "ESTOQUE", "DATA_REF", "ANO_MES", "DIA_CORTE"])

    # FILIAL_2 = FILIAL padronizada (mesmo nome usado na Etapa 1)
    df["FILIAL_2"] = df["FILIAL"]

    # ESTOQUE como numérico (mantemos NaN; não forçamos 0 para não marcar ruptura falsa)
    df["ESTOQUE"] = pd.to_numeric(df["ESTOQUE"], errors="coerce")

    # DATA_REF
    if "DATA_SALDO" in df.columns:
        data_ref = pd.to_datetime(df["DATA_SALDO"], dayfirst=True, errors="coerce")
        if data_ref.notna().any():
            df["DATA_REF"] = data_ref
        else:
            df["DATA_REF"] = infer_date_from_filename(path.name)
    else:
        df["DATA_REF"] = infer_date_from_filename(path.name)

    df["DATA_REF"] = pd.to_datetime(df["DATA_REF"], errors="coerce")

    # se continuou sem data, descarta
    df = df[df["DATA_REF"].notna()].copy()
    if df.empty:
        log(f"⚠ Não foi possível determinar DATA_REF em {path.name}. Ignorando.")
        return pd.DataFrame(columns=["GRIFFE", "FILIAL_2", "GRUPO_PRODUTO",
                                     "ESTOQUE", "DATA_REF", "ANO_MES", "DIA_CORTE"])

    # ANO_MES no padrão YYYYMM
    df["ANO_MES"] = df["DATA_REF"].dt.strftime("%Y%m")
    df["DIA_CORTE"] = df["DATA_REF"].dt.day

    cols_keep = ["GRIFFE", "FILIAL_2", "GRUPO_PRODUTO",
                 "ESTOQUE", "DATA_REF", "ANO_MES", "DIA_CORTE"]
    return df[cols_keep].copy()


def carregar_todos_snapshots() -> pd.DataFrame:
    """
    Lê todos os ESTOQUE_*.xls* da pasta INPUT_DIR e concatena em um único DataFrame.
    """
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Pasta de estoque não encontrada: {INPUT_DIR}")

    arquivos = sorted(list(INPUT_DIR.glob("ESTOQUE_*.xls*")))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo ESTOQUE_*.xls* encontrado em {INPUT_DIR}")

    log(f"Encontrados {len(arquivos)} arquivos de estoque.")
    frames = []
    for p in arquivos:
        try:
            df = carregar_arquivo_estoque(p)
            if not df.empty:
                log(f"   → {len(df):,} linhas úteis em {p.name}")
                frames.append(df)
        except Exception:
            log(f"❌ Erro ao processar {p.name}:")
            traceback.print_exc()

    if not frames:
        log("⚠ Nenhum dado de estoque válido após leitura de todos os arquivos.")
        return pd.DataFrame(columns=["GRIFFE", "FILIAL_2", "GRUPO_PRODUTO",
                                     "ESTOQUE", "DATA_REF", "ANO_MES", "DIA_CORTE"])

    estoque_all = pd.concat(frames, ignore_index=True)
    log(f"Total consolidado de estoque: {len(estoque_all):,} linhas.")
    return estoque_all


def alinhar_filiais_com_vendas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Se existir staging_consolidado.csv da Etapa 1, restringe o estoque apenas às
    filiais que existem nas vendas (FILIAL_2). Isso evita filiais puramente
    logísticas que não entram na previsão.
    """
    if not STAGING_VENDAS.exists():
        log("🔎 staging_consolidado.csv não encontrado. Mantendo todas as filiais do estoque.")
        return df

    try:
        log(f"Lendo staging_consolidado para alinhar filiais: {STAGING_VENDAS}")
        vendas = pd.read_csv(STAGING_VENDAS, sep=";")
        if "FILIAL_2" not in vendas.columns:
            log("⚠ staging_consolidado.csv sem coluna FILIAL_2. Não será usado para filtro.")
            return df
        filiais_validas = set(vendas["FILIAL_2"].astype(str).str.upper().str.strip().unique())
        antes = len(df)
        df = df[df["FILIAL_2"].isin(filiais_validas)].copy()
        log(f"Filtrando por filiais presentes em vendas: {antes:,} → {len(df):,} linhas.")
        return df
    except Exception:
        log("⚠ Erro ao ler staging_consolidado.csv. Ignorando filtro de filiais.")
        traceback.print_exc()
        return df


def agregar_mensal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega o estoque mensal por (ANO_MES, FILIAL_2, GRUPO_PRODUTO)
    do jeito correto:
      1) soma o estoque por dia (DATA_REF) → ESTOQUE_DIA
      2) calcula média/mín/máx e snapshots no mês
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "ANO_MES", "FILIAL_2", "GRUPO_PRODUTO",
            "ESTOQUE_MEDIO", "ESTOQUE_MIN", "ESTOQUE_MAX",
            "SNAPSHOTS", "RUPTURA_FLAG", "SNAPSHOT_UNICO", "IMPUTADO"
        ])

    df = df.copy()
    df["DATA_REF"] = pd.to_datetime(df["DATA_REF"], errors="coerce")
    df = df[df["DATA_REF"].notna()].copy()

    # garantir DIA_CORTE
    df["DIA_CORTE"] = df["DATA_REF"].dt.day

    # 1) nível diário: soma o estoque do grupo naquele dia (somando todos os SKUs)
    daily_full = (
        df
        .groupby(["ANO_MES", "FILIAL_2", "GRUPO_PRODUTO", "DATA_REF"], dropna=False)
        .agg(ESTOQUE_DIA=("ESTOQUE", "sum"))
        .reset_index()
    )
    daily_full["DIA_CORTE"] = daily_full["DATA_REF"].dt.day

    # 2) aplica filtro de dias (1 e 15, por exemplo)
    if KEEP_DAYS:
        antes = len(daily_full)
        daily = daily_full[daily_full["DIA_CORTE"].isin(KEEP_DAYS)].copy()
        if daily.empty:
            log(f"⚠ KEEP_DAYS={KEEP_DAYS} esvaziou o mês. Revertendo e usando todos os dias disponíveis.")
            daily = daily_full.copy()
        else:
            log(f"Aplicando filtro KEEP_DAYS={KEEP_DAYS}: {antes:,} → {len(daily):,} linhas (nível diário).")
    else:
        daily = daily_full.copy()

    if daily.empty:
        log("⚠ Após aplicar KEEP_DAYS, nenhum dado diário restante.")
        return pd.DataFrame(columns=[
            "ANO_MES", "FILIAL_2", "GRUPO_PRODUTO",
            "ESTOQUE_MEDIO", "ESTOQUE_MIN", "ESTOQUE_MAX",
            "SNAPSHOTS", "RUPTURA_FLAG", "SNAPSHOT_UNICO", "IMPUTADO"
        ])

    # 3) agrega para nível mensal por filial + grupo
    agg = (
        daily
        .groupby(["ANO_MES", "FILIAL_2", "GRUPO_PRODUTO"], dropna=False)
        .agg(
            ESTOQUE_MEDIO=("ESTOQUE_DIA", "mean"),
            ESTOQUE_MIN=("ESTOQUE_DIA", "min"),
            ESTOQUE_MAX=("ESTOQUE_DIA", "max"),
            SNAPSHOTS=("ESTOQUE_DIA", "size"),              # nº de dias usados (1 ou 2)
            RUPTURA_FLAG=("ESTOQUE_DIA", lambda s: int((s <= 0).any()))
        )
        .reset_index()
    )

    # garantir numérico
    for col in ["ESTOQUE_MEDIO", "ESTOQUE_MIN", "ESTOQUE_MAX", "SNAPSHOTS", "RUPTURA_FLAG"]:
        agg[col] = pd.to_numeric(agg[col], errors="coerce")

    # flags de qualidade
    agg["SNAPSHOT_UNICO"] = (agg["SNAPSHOTS"] == 1).astype(int)
    agg["IMPUTADO"] = 0

    # imputação para SNAPSHOTS=1: usa mediana dos últimos 3 meses (por FILIAL_2+GRUPO_PRODUTO)
    agg["ANO_MES_INT"] = pd.to_numeric(agg["ANO_MES"], errors="coerce")

    def impute_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("ANO_MES_INT").copy()
        med_hist = g["ESTOQUE_MEDIO"].rolling(window=3, min_periods=1).median().shift(1)
        mask = g["SNAPSHOTS"] == 1
        g.loc[mask & med_hist.notna(), "ESTOQUE_MEDIO"] = med_hist[mask]
        g.loc[mask & med_hist.notna(), "ESTOQUE_MIN"] = med_hist[mask]
        g.loc[mask & med_hist.notna(), "ESTOQUE_MAX"] = med_hist[mask]
        g.loc[mask & med_hist.notna(), "IMPUTADO"] = 1
        return g

    agg = agg.groupby(["FILIAL_2", "GRUPO_PRODUTO"], group_keys=False).apply(impute_group)

    # preencher NaN remanescentes de métricas numéricas com 0 após imputação
    for col in ["ESTOQUE_MEDIO", "ESTOQUE_MIN", "ESTOQUE_MAX", "SNAPSHOTS", "RUPTURA_FLAG", "SNAPSHOT_UNICO", "IMPUTADO"]:
        agg[col] = pd.to_numeric(agg[col], errors="coerce").fillna(0)

    # ordena
    agg = agg.sort_values(["ANO_MES_INT", "FILIAL_2", "GRUPO_PRODUTO"]).reset_index(drop=True)
    agg.drop(columns=["ANO_MES_INT"], inplace=True)

    return agg


# ================== MAIN ==================

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    t0 = time.time()
    log("Iniciando ETAPA 1B — Estoque histórico mensal")

    # 1) carregar todos os snapshots
    estoque_all = carregar_todos_snapshots()

    if estoque_all.empty:
        log("⚠ estoque_all vazio após leitura. Salvando arquivo vazio para debug e encerrando.")
        out_consolidado = OUT_DIR / "estoque_consolidado_clean.csv"
        estoque_all.to_csv(out_consolidado, index=False, encoding="utf-8-sig")
        log(f"📁 Salvo consolidado (vazio) em: {out_consolidado}")
        return

    # 2) alinhar filiais com a base de vendas (se possível)
    estoque_all = alinhar_filiais_com_vendas(estoque_all)

    if estoque_all.empty:
        log("⚠ Após alinhar com vendas, estoque_all ficou vazio. Salvando arquivo vazio e encerrando.")
        out_consolidado = OUT_DIR / "estoque_consolidado_clean.csv"
        estoque_all.to_csv(out_consolidado, index=False, encoding="utf-8-sig")
        log(f"📁 Salvo consolidado (vazio) em: {out_consolidado}")
        return

    # 3) salvar consolidado linha a linha (debug)
    out_consolidado = OUT_DIR / "estoque_consolidado_clean.csv"
    estoque_all.to_csv(out_consolidado, index=False, encoding="utf-8-sig")
    log(f"📁 Salvo consolidado limpo em: {out_consolidado}")

    # 4) agregação mensal para input da Etapa 2
    hist_mensal = agregar_mensal(estoque_all)

    out_hist = OUT_DIR / "estoque_historico_mensal.csv"
    hist_mensal.to_csv(out_hist, index=False, encoding="utf-8-sig")
    log(f"📁 Salvo histórico mensal em: {out_hist}")
    log(f"Linhas em estoque_historico_mensal: {len(hist_mensal):,}")

    log(f"✅ ETAPA 1B concluída em {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
