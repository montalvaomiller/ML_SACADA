# limpa_e_agrega_estoque_por_filial_feedback.py
import pandas as pd
import re, unicodedata, time
from pathlib import Path
from tqdm import tqdm  # <== barra de progresso (instale: pip install tqdm)

# ========================= CONFIG =========================
INPUT_DIR = r"C:\Users\monta\OneDrive\Documentos\Meta\METAxSACADA\ML-Sacada\dados_estoque"
OUT_DIR   = r"C:\Users\monta\OneDrive\Documentos\Meta\METAxSACADA\ML-Sacada\saidas"
KEEP_DAYS = (1, 15)
SAVE_INDIVIDUAIS_CLEAN = False
# ==========================================================

ALLOWED_FILIAIS = [
"E-COMMERCE SACADA","EST EXP. ATAC. OHBOY NOVO","EST EXP. ATAC. SACADA",
"OUTLET CATARINA-SP","OUTLET PREMIUM-RJ","SACADA BARRA SALVADOR-BA",
"SACADA BARRA SHOPPING-RJ","SACADA BOULEVARD SHOPP-PA","SACADA CENTRO-RJ",
"SACADA COPACABANA-RJ","SACADA FLAMBOYANT SHOP-GO","SACADA ICARAI-RJ",
"SACADA IGUATEMI FORTAL-CE","SACADA IGUATEMI SALVA-BA","SACADA IPANEMA(550)-RJ",
"SACADA JUIZ DE FORA-MG","SACADA LARGO MACHADO-RJ","SACADA MOEMA-SP",
"SACADA OSCAR FREIRE-SP","SACADA PATIO SAVASSI-MG","SACADA PLAZA C.FORTE-PE",
"SACADA PLAZA NITEROI-RJ","SACADA RDB-RJ","SACADA RDL-RJ","SACADA SALVADOR SHOPP-BA",
"SACADA SHOPP. ELDORADO-SP","SACADA SHOPP. LEBLON-RJ","SACADA SHOPP. RECIFE-PE",
"SACADA SHOPP. RIO SUL-RJ","SACADA SHOPP.GOIANIA-GO","SACADA SHOPP.RIO MAR-PE",
"SACADA SHOPP.RIO MAR-SE","SACADA SHOPP.TIJUCA-RJ","SACADA SHOPP.VITORIA-ES","SEM_FILIAL"
]

def normalize_text(s):
    if pd.isna(s): return ""
    s = str(s).upper().strip()
    s = re.sub(r"[–—−‐-‒]", "-", s)
    s = unicodedata.normalize("NFKD", s).encode("ASCII","ignore").decode("ASCII")
    s = s.replace(" SHOPPING", " SHOPP.").replace("SHOPPING","SHOPP.")
    s = s.replace(" SHOP."," SHOPP.").replace("SHOPP .","SHOPP.")
    s = re.sub(r"\s+\.",".",s)
    s = re.sub(r"\s+"," ",s)
    s = re.sub(r"\s*-\s*","-",s)
    return s

ALLOWED_NORM_MAP = {normalize_text(v): v for v in ALLOWED_FILIAIS}

DATE_PATTERNS = [
    re.compile(r"(20\d{2})[-_/\.]?(0[1-9]|1[0-2])[-_/\.]?([0-3]\d)"),
    re.compile(r"([0-3]\d)[-_/\.]?(0[1-9]|1[0-2])[-_/\.]?(20\d{2})"),
]

def extract_date_from_name(name):
    base = Path(name).stem
    for pat in DATE_PATTERNS:
        m = pat.search(base)
        if m:
            g = m.groups()
            if pat is DATE_PATTERNS[0]:
                y,mn,d = g
            else:
                d,mn,y = g
            return f"{y}-{mn}-{d}"
    return None

def clean_one_excel(path_xlsx: Path):
    df = pd.read_excel(path_xlsx, dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]
    ren = {}
    for c in df.columns:
        uc = c.upper()
        if "GRIFF" in uc or "GRIFE" in uc: ren[c] = "GRIFFE"
        if "FILIAL" in uc or "LOJA" in uc: ren[c] = "FILIAL"
    if ren: df = df.rename(columns=ren)

    if "GRIFFE" in df.columns:
        df = df[df["GRIFFE"].str.upper() == "SACADA"]
    if "FILIAL" in df.columns:
        df["_FILIAL_NORM"] = df["FILIAL"].map(normalize_text)
        df = df[df["_FILIAL_NORM"].isin(ALLOWED_NORM_MAP.keys())].copy()
        df["FILIAL"] = df["_FILIAL_NORM"].map(ALLOWED_NORM_MAP)
        df.drop(columns=["_FILIAL_NORM"], inplace=True)
    else:
        df["FILIAL"] = "SEM_FILIAL"

    df["PERIODO"] = extract_date_from_name(path_xlsx.name)
    df["MES_REF"] = pd.to_datetime(df["PERIODO"], errors="coerce").dt.strftime("%Y-%m")
    df["DIA_CORTE"] = pd.to_datetime(df["PERIODO"], errors="coerce").dt.day
    df["ARQUIVO_ORIGEM"] = path_xlsx.name
    return df

def quantidade_to_float(s):
    s = (s.astype(str)
         .str.replace(".","",regex=False)
         .str.replace(",",".",regex=False)
         .str.extract(r"([-+]?\d*\.?\d+)")[0]
         .astype(float)
         .fillna(0.0))
    return s

def main():
    start = time.time()
    folder = Path(INPUT_DIR)
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(list(folder.glob("*.xlsx")) + list(folder.glob("*.xls")))
    if not files:
        print(f"Nenhum arquivo .xlsx/.xls em {folder}")
        return

    all_frames = []
    print(f"🔹 Iniciando limpeza de {len(files)} arquivos...\n")

    for p in tqdm(files, desc="Processando arquivos", ncols=100):
        try:
            df = clean_one_excel(p)
            if KEEP_DAYS: df = df[df["DIA_CORTE"].isin(KEEP_DAYS)]
            if not df.empty:
                all_frames.append(df)
                if SAVE_INDIVIDUAIS_CLEAN:
                    p_out = p.with_name(p.stem + "_CLEAN.xlsx")
                    df.to_excel(p_out, index=False)
        except Exception as e:
            print(f"\n❌ Erro em {p.name}: {e}")

    if not all_frames:
        print("Nenhum dado válido após limpeza.")
        return

    base = pd.concat(all_frames, ignore_index=True)
    consolidado_csv = out_dir / "estoque_consolidado_clean.csv"
    base.to_csv(consolidado_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ Consolidado limpo salvo em: {consolidado_csv} ({len(base):,} linhas)\n")

    # ---- agregação ----
    grupo_candidates = ("GRUPO_PRODUTO","GRUPO PRODUTO","GRUPO","CATEGORIA")
    qty_candidates   = ("ESTOQUE","QTD","QTDE","QUANTIDADE","QUANT","ESTOQUE_ATUAL","ESTOQUE TOTAL")

    def find_col(df, cands):
        for c in df.columns:
            up = c.upper().strip()
            for cand in cands:
                if cand in up:
                    return c
        return None

    col_g = find_col(base, grupo_candidates)
    col_q = find_col(base, qty_candidates)
    if not col_g or not col_q:
        print("❌ Colunas de grupo ou quantidade não encontradas.")
        return

    df = base[["FILIAL", col_g, col_q]].copy()
    df[col_g] = df[col_g].astype(str).str.upper().str.strip()
    df[col_q] = quantidade_to_float(df[col_q])

    agg = df.groupby(["FILIAL", col_g], dropna=False, as_index=False)[col_q].sum()
    agg = agg.rename(columns={col_g:"GRUPO_PRODUTO", col_q:"QTD_ESTOQUE"})
    total = agg.groupby("FILIAL", as_index=False)["QTD_ESTOQUE"].sum().rename(columns={"QTD_ESTOQUE":"TOTAL_FILIAL"})
    agg = agg.merge(total, on="FILIAL", how="left")
    agg["PCT_FILIAL"] = (agg["QTD_ESTOQUE"] / agg["TOTAL_FILIAL"]).round(4)

    out1 = out_dir / "estoque_grupo_por_filial.csv"
    out2 = out_dir / "estoque_grupo_por_filial_pivot.csv"
    agg.to_csv(out1, index=False, encoding="utf-8-sig")
    agg.pivot(index="FILIAL", columns="GRUPO_PRODUTO", values="QTD_ESTOQUE").fillna(0).reset_index().to_csv(out2, index=False, encoding="utf-8-sig")

    elapsed = round(time.time()-start,1)
    print(f"\n✅ Arquivos gerados:\n- {out1}\n- {out2}")
    print(f"⏱️ Tempo total: {elapsed} s\n")

if __name__ == "__main__":
    main()
