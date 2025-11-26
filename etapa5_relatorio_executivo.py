# etapa5_relatorio_executivo_v6.py
# Relatório Executivo com gráficos por FAIXA_PRECO + escrita segura (v6)
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile, shutil, traceback

# ================== CONFIG ==================
BASE_DIR = Path(__file__).resolve().parent
IN_DIR   = BASE_DIR / "saidas"
IN_XLSX  = IN_DIR / "validacao_decisao_2026.xlsx"
OUT_XLSX = IN_DIR / "relatorio_executivo_2026.xlsx"

FAIXA_ORDER = ["P1","P2","P3"]  # ordem executiva

# ================== UTILS ==================
def safe_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def as_ano_mes_series(s):
    # aceita int/float/str e devolve YYYYMM (string) consistente
    s = s.astype(str).str.extract(r"(\d{6})")[0]
    return s

def col_name(idx):
    """Converte índice de coluna 0-based para rótulo Excel (A..Z, AA..)."""
    name = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx-1, 26)
        name = chr(65 + rem) + name
    return name

def _safe_write_excel(path: Path, build_fn):
    """
    Cria um arquivo temporário, executa build_fn(xw) para montar o workbook,
    fecha o writer e só então move o .xlsx para 'path'.
    """
    tmpdir = Path(tempfile.mkdtemp())
    tmpfile = tmpdir / (path.stem + "_tmp.xlsx")
    try:
        import xlsxwriter  # garante engine
        with pd.ExcelWriter(tmpfile, engine="xlsxwriter") as xw:
            build_fn(xw)   # monta todas as abas e gráficos
        shutil.move(str(tmpfile), str(path))
        print(f"[OK] Relatório executivo salvo em: {path}")
    except Exception:
        print("❌ Falha ao gravar o Excel (arquivo pode estar aberto ou houve erro em gráfico).")
        traceback.print_exc()
        raise
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except:
            pass

# ================== BLOCOS DE DADOS ==================
def montar_kpis(df):
    kpis_geral = pd.DataFrame({
        "KPI": [
            "Receita total (ajustada)", "Lucro total (ajustado)",
            "Margem projetada média (%)", "Itens PRODUZIR (%)",
            "Itens REVER PREÇO (%)", "Itens BAIXA PRIORIDADE (%)"
        ],
        "VALOR": [
            df.get("VAL_AJUST", pd.Series(dtype=float)).sum(),
            df.get("LUCRO_AJUST", pd.Series(dtype=float)).sum(),
            df.get("MARGEM_PROJETADA", pd.Series(dtype=float)).mean() * 100,
            (df.get("STATUS_PROD","").astype(str)=="PRODUZIR").mean() * 100,
            (df.get("STATUS_PROD","").astype(str)=="REVER PREÇO").mean() * 100,
            (df.get("STATUS_PROD","").astype(str)=="BAIXA PRIORIDADE").mean() * 100,
        ]
    })

    group_cols = [c for c in ["GRIFFE","LINHA"] if c in df.columns]
    if not group_cols:
        kpis_gl = pd.DataFrame(columns=["GRIFFE","LINHA","RECEITA_AJUST","LUCRO_AJUST","MARGEM_MEDIA","PCT_PRODUZIR","PCT_REVER_PRECO","PCT_BAIXA"])
    else:
        kpis_gl = df.groupby(group_cols, dropna=False).agg(
            RECEITA_AJUST=("VAL_AJUST","sum"),
            LUCRO_AJUST=("LUCRO_AJUST","sum"),
            MARGEM_MEDIA=("MARGEM_PROJETADA","mean"),
            PCT_PRODUZIR=("STATUS_PROD", lambda x: (x=="PRODUZIR").mean()*100),
            PCT_REVER_PRECO=("STATUS_PROD", lambda x: (x=="REVER PREÇO").mean()*100),
            PCT_BAIXA=("STATUS_PROD", lambda x: (x=="BAIXA PRIORIDADE").mean()*100)
        ).reset_index()
        kpis_gl["MARGEM_MEDIA"] = kpis_gl["MARGEM_MEDIA"] * 100

    return kpis_geral, kpis_gl

def montar_tops(df, top_n=20):
    cols_keep = [
        "GRIFFE","LINHA","GRUPO_PRODUTO","FILIAL_2","ANO_MES","FAIXA_PRECO",
        "QTDE_ALOCADA","PRECO_SUGERIDO","VAL_AJUST","LUCRO_AJUST",
        "MARGEM_PROJETADA","MARGEM_ALVO","STATUS_PROD","PRIORIDADE"
    ]
    for c in cols_keep:
        if c not in df.columns:
            df[c] = np.nan
    top = df[cols_keep].copy()
    top["ANO_MES"] = as_ano_mes_series(top["ANO_MES"].astype(str)) if "ANO_MES" in top.columns else np.nan
    top = top.sort_values("LUCRO_AJUST", ascending=False).head(top_n)
    return top

def montar_riscos(df):
    riscos = df[df.get("MARGEM_PROJETADA",0) < df.get("MARGEM_ALVO",0)].copy()
    if not riscos.empty and "ANO_MES" in riscos.columns:
        riscos["ANO_MES"] = as_ano_mes_series(riscos["ANO_MES"].astype(str))
    riscos = riscos.sort_values([c for c in ["GRIFFE","LINHA","GRUPO_PRODUTO","FILIAL_2","ANO_MES"] if c in riscos.columns])
    cols = [
        "GRIFFE","LINHA","GRUPO_PRODUTO","FAIXA_PRECO","FILIAL_2","ANO_MES",
        "QTDE_ALOCADA","MARGEM_PROJETADA","MARGEM_ALVO","PRECO_SUGERIDO","VAL_AJUST","LUCRO_AJUST"
    ]
    for c in cols:
        if c not in riscos.columns:
            riscos[c] = np.nan
    return riscos[cols]

def montar_prioridades(df):
    prior_linha = df.groupby(["LINHA","PRIORIDADE"], dropna=False).size().reset_index(name="QTD") if "LINHA" in df.columns else pd.DataFrame(columns=["LINHA","PRIORIDADE","QTD"])
    prior_griffe = df.groupby(["GRIFFE","PRIORIDADE"], dropna=False).size().reset_index(name="QTD") if "GRIFFE" in df.columns else pd.DataFrame(columns=["GRIFFE","PRIORIDADE","QTD"])
    pivot_linha = prior_linha.pivot(index="LINHA", columns="PRIORIDADE", values="QTD").fillna(0).reset_index() if not prior_linha.empty else pd.DataFrame(columns=["LINHA","BAIXA","MÉDIA","ALTA"])
    pivot_griffe = prior_griffe.pivot(index="GRIFFE", columns="PRIORIDADE", values="QTD").fillna(0).reset_index() if not prior_griffe.empty else pd.DataFrame(columns=["GRIFFE","BAIXA","MÉDIA","ALTA"])
    receita_griffe = df.groupby("GRIFFE", dropna=False)["VAL_AJUST"].sum().reset_index().sort_values("VAL_AJUST", ascending=False) if "GRIFFE" in df.columns else pd.DataFrame(columns=["GRIFFE","VAL_AJUST"])
    return pivot_linha, pivot_griffe, receita_griffe

# ================== MAIN ==================
def main():
    # ====== Carregar
    df = pd.read_excel(IN_XLSX, sheet_name="Decisoes_Produto_2026")
    df = safe_num(df, [
        "QTDE_ALOCADA","CUSTO_UNIT_HIST","MARGEM_ALVO","PRECO_MEDIO_APLICADO",
        "PRECO_SUGERIDO","VAL_BASE","VAL_AJUST","LUCRO_BASE","LUCRO_AJUST",
        "MARGEM_REAL","MARGEM_PROJETADA"
    ])
    # normaliza ANO_MES e FAIXA_PRECO
    if "ANO_MES" in df.columns:
        df["ANO_MES"] = as_ano_mes_series(df["ANO_MES"].astype(str))
    if "FAIXA_PRECO" in df.columns:
        df["FAIXA_PRECO"] = pd.Categorical(df["FAIXA_PRECO"], categories=FAIXA_ORDER, ordered=True)

    # ====== Bases principais
    kpis_geral, kpis_gl = montar_kpis(df)
    top20 = montar_tops(df, top_n=20)
    riscos = montar_riscos(df)
    prior_linha, prior_griffe, receita_griffe = montar_prioridades(df)

    # ====== Agregações para gráficos (existentes)
    base_scatter = df[[c for c in [
        "GRIFFE","LINHA","MARGEM_ALVO","MARGEM_PROJETADA","STATUS_PROD",
        "PRECO_SUGERIDO","QTDE_ALOCADA","VAL_AJUST","LUCRO_AJUST","ANO_MES"
    ] if c in df.columns]].copy()

    if "GRIFFE" in df.columns:
        rec_lucro_griffe = df.groupby("GRIFFE", dropna=False).agg(
            RECEITA_AJUST=("VAL_AJUST","sum"),
            LUCRO_AJUST=("LUCRO_AJUST","sum"),
        ).reset_index().sort_values("RECEITA_AJUST", ascending=False)
    else:
        rec_lucro_griffe = pd.DataFrame(columns=["GRIFFE","RECEITA_AJUST","LUCRO_AJUST"])

    status_dist = df["STATUS_PROD"].value_counts(dropna=False).rename_axis("STATUS_PROD").reset_index(name="QTD") if "STATUS_PROD" in df.columns else pd.DataFrame(columns=["STATUS_PROD","QTD"])

    if "LINHA" in df.columns:
        risco_pct_linha = (
            df.assign(_risco=(df["MARGEM_PROJETADA"] < df["MARGEM_ALVO"]))
              .groupby("LINHA", dropna=False)["_risco"].mean().mul(100)
              .reset_index(name="PCT_ABAIXO_META")
              .sort_values("PCT_ABAIXO_META", ascending=False)
        )
    else:
        risco_pct_linha = pd.DataFrame(columns=["LINHA","PCT_ABAIXO_META"])

    if "ANO_MES" in df.columns:
        receita_mensal = (df.groupby("ANO_MES", dropna=False)["VAL_AJUST"]
                            .sum().reset_index().sort_values("ANO_MES"))
    else:
        receita_mensal = pd.DataFrame({"ANO_MES": [], "VAL_AJUST": []})

    # ---- Base para distribuição mensal de peças com filtro de filial
    if set(["ANO_MES","FILIAL_2","QTDE_ALOCADA"]).issubset(df.columns):
        base_pecas = (
            df.groupby(["ANO_MES", "FILIAL_2"], dropna=False)["QTDE_ALOCADA"]
              .sum().reset_index()
              .sort_values(["ANO_MES", "FILIAL_2"])
        )
        meses_unicos   = df["ANO_MES"].dropna().drop_duplicates().sort_values().astype(str).tolist()
        filiais_unicas = df["FILIAL_2"].dropna().drop_duplicates().sort_values().astype(str).tolist()
    else:
        base_pecas = pd.DataFrame(columns=["ANO_MES","FILIAL_2","QTDE_ALOCADA"])
        meses_unicos, filiais_unicas = [], []

    # ====== Novas agregações por FAIXA_PRECO
    if set(["LINHA","FAIXA_PRECO","VAL_AJUST"]).issubset(df.columns):
        receita_linha_faixa = (
            df.groupby(["LINHA","FAIXA_PRECO"], dropna=False)["VAL_AJUST"].sum()
              .reset_index().sort_values(["LINHA","FAIXA_PRECO"])
        )
    else:
        receita_linha_faixa = pd.DataFrame(columns=["LINHA","FAIXA_PRECO","VAL_AJUST"])

    if "FAIXA_PRECO" in df.columns and "MARGEM_PROJETADA" in df.columns:
        margem_media_faixa = (
            df.groupby(["FAIXA_PRECO"], dropna=False)["MARGEM_PROJETADA"].mean()
              .reset_index().sort_values("FAIXA_PRECO")
        )
    else:
        margem_media_faixa = pd.DataFrame(columns=["FAIXA_PRECO","MARGEM_PROJETADA"])

    if set(["ANO_MES","FAIXA_PRECO","QTDE_ALOCADA"]).issubset(df.columns):
        volume_mensal_faixa = (
            df.groupby(["ANO_MES","FAIXA_PRECO"], dropna=False)["QTDE_ALOCADA"].sum()
              .reset_index().sort_values(["ANO_MES","FAIXA_PRECO"])
        )
    else:
        volume_mensal_faixa = pd.DataFrame(columns=["ANO_MES","FAIXA_PRECO","QTDE_ALOCADA"])

    if set(["LINHA","FAIXA_PRECO","MARGEM_PROJETADA"]).issubset(df.columns):
        margem_linha_faixa = (
            df.groupby(["LINHA","FAIXA_PRECO"], dropna=False)["MARGEM_PROJETADA"].mean()
              .reset_index().sort_values(["LINHA","FAIXA_PRECO"])
        )
    else:
        margem_linha_faixa = pd.DataFrame(columns=["LINHA","FAIXA_PRECO","MARGEM_PROJETADA"])

    # Mix de faixas por GRIFFE (participação em QTDE)
    if set(["GRIFFE","FAIXA_PRECO","QTDE_ALOCADA"]).issubset(df.columns):
        grp = (
            df.groupby(["GRIFFE","FAIXA_PRECO"], dropna=False)["QTDE_ALOCADA"]
              .sum()
              .reset_index(name="QTD")
        )
        tot_por_griffe = grp.groupby("GRIFFE", dropna=False)["QTD"].transform("sum").replace(0, np.nan)
        grp["PCT"] = (grp["QTD"] / tot_por_griffe).fillna(0)
        grp["FAIXA_PRECO"] = pd.Categorical(grp["FAIXA_PRECO"], categories=FAIXA_ORDER, ordered=True)
        mix_griffe_faixa = grp.sort_values(["GRIFFE","FAIXA_PRECO"]).reset_index(drop=True)
    else:
        mix_griffe_faixa = pd.DataFrame(columns=["GRIFFE","FAIXA_PRECO","QTD","PCT"])

    # ================== CONSTRUÇÃO DO EXCEL (ESCRITA SEGURA) ==================
    def build_workbook(xw):
        # --- Abas de dados (originais)
        kpis_geral.to_excel(xw, sheet_name="KPIs_Gerais", index=False)
        kpis_gl.to_excel(xw, sheet_name="KPIs_Griffe_Linha", index=False)
        top20.to_excel(xw, sheet_name="Top_Produtos", index=False)
        riscos.to_excel(xw, sheet_name="Riscos", index=False)
        prior_linha.to_excel(xw, sheet_name="Prioridades_Linha", index=False)
        prior_griffe.to_excel(xw, sheet_name="Prioridades_Griffe", index=False)
        receita_griffe.to_excel(xw, sheet_name="Receita_Griffe", index=False)
        base_scatter.to_excel(xw, sheet_name="Base", index=False)
        rec_lucro_griffe.to_excel(xw, sheet_name="Receita_Lucro_Griffe", index=False)
        status_dist.to_excel(xw, sheet_name="Status_Distrib", index=False)
        risco_pct_linha.to_excel(xw, sheet_name="Risco_Linha", index=False)
        receita_mensal.to_excel(xw, sheet_name="Receita_Mensal", index=False)
        base_pecas.to_excel(xw, sheet_name="Base_Pecas", index=False)

        # --- Abas de dados (novas por FAIXA_PRECO)
        receita_linha_faixa.to_excel(xw, sheet_name="Receita_Linha_Faixa", index=False)
        margem_media_faixa.to_excel(xw, sheet_name="Margem_Faixa", index=False)
        volume_mensal_faixa.to_excel(xw, sheet_name="Volume_Mensal_Faixa", index=False)
        margem_linha_faixa.to_excel(xw, sheet_name="Margem_Linha_Faixa", index=False)
        status_faixa_raw = (
            df.groupby(["FAIXA_PRECO","STATUS_PROD"], dropna=False).size()
              .reset_index(name="QTD").sort_values(["FAIXA_PRECO","STATUS_PROD"])
            if set(["FAIXA_PRECO","STATUS_PROD"]).issubset(df.columns) else pd.DataFrame(columns=["FAIXA_PRECO","STATUS_PROD","QTD"])
        )
        status_faixa_raw.to_excel(xw, sheet_name="Status_Faixa", index=False)
        mix_griffe_faixa.to_excel(xw, sheet_name="Mix_Griffe_Faixa", index=False)

        wb = xw.book

        # === Formats ===
        header_fmt = wb.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1})
        money_fmt  = wb.add_format({'num_format': 'R$ #,##0', 'border': 0})
        pct_fmt    = wb.add_format({'num_format': '0.0"%"', 'border': 0})
        int_fmt    = wb.add_format({'num_format': '#,##0', 'border': 0})
        high_fmt   = wb.add_format({'bg_color': '#FFF2CC'})  # destaque leve

        def apply_table_style(ws_name, df_obj):
            try:
                ws = xw.sheets[ws_name]
            except KeyError:
                return None
            nrows = len(df_obj)
            ncols = len(df_obj.columns)
            if ncols == 0:
                return ws
            ws.set_row(0, None, header_fmt)
            ws.freeze_panes(1, 1)
            ws.autofilter(0, 0, max(nrows,1), max(ncols-1,0))
            ws.set_column(0, ncols-1, 14)

            for j, col in enumerate(df_obj.columns):
                width = 14
                if col in ("GRIFFE","LINHA","GRUPO_PRODUTO","STATUS_PROD","PRIORIDADE","FILIAL_2"):
                    width = 18
                if col in ("KPI",):
                    width = 32
                if col in ("ANO_MES",):
                    width = 12
                ws.set_column(j, j, width)

                if any(k in col for k in ("VAL_", "LUCRO_", "PRECO_")):
                    ws.set_column(j, j, None, money_fmt)
                elif col.startswith("PCT_") or "MARGEM" in col or col.endswith("(%)") or col=="PCT":
                    ws.set_column(j, j, None, pct_fmt)
                elif col.startswith("QTDE") or col.startswith("QTD"):
                    ws.set_column(j, j, None, int_fmt)
            return ws

        # Aplicar estilo em todas as abas de dados
        for sheet_name, data_obj in [
            ("KPIs_Gerais", kpis_geral),
            ("KPIs_Griffe_Linha", kpis_gl),
            ("Top_Produtos", top20),
            ("Riscos", riscos),
            ("Prioridades_Linha", prior_linha),
            ("Prioridades_Griffe", prior_griffe),
            ("Receita_Griffe", receita_griffe),
            ("Base", base_scatter),
            ("Receita_Lucro_Griffe", rec_lucro_griffe),
            ("Status_Distrib", status_dist),
            ("Risco_Linha", risco_pct_linha),
            ("Receita_Mensal", receita_mensal),
            ("Base_Pecas", base_pecas),
            ("Receita_Linha_Faixa", receita_linha_faixa),
            ("Margem_Faixa", margem_media_faixa),
            ("Volume_Mensal_Faixa", volume_mensal_faixa),
            ("Margem_Linha_Faixa", margem_linha_faixa),
            ("Status_Faixa", status_faixa_raw),
            ("Mix_Griffe_Faixa", mix_griffe_faixa),
        ]:
            apply_table_style(sheet_name, data_obj)

        # ===== Aba interativa: Pecas_Mes_Filtro =====
        wsPF = wb.add_worksheet("Pecas_Mes_Filtro")
        wsPF.hide_gridlines(2)
        wsPF.write("A1", "Filial:")
        wsPF.write("A3", "Mês")
        wsPF.write("B3", "Peças (QTDE_ALOCADA)")
        wsPF.set_row(2, None, header_fmt)

        lista_filiais = ["TODAS"] + ([x for x in filiais_unicas] if filiais_unicas else [])
        start_row_list = 2  # H2 em diante
        wsPF.write("H1", "ListaFiliais")
        for i, val in enumerate(lista_filiais, start=start_row_list):
            wsPF.write(i-1, 7, val)  # (linha-1, col 7=H)
        wsPF.set_column("H:H", 20, None, {'hidden': True})

        start_row_meses = 4
        for i, mes in enumerate(meses_unicos, start=start_row_meses):
            wsPF.write(i-1, 0, str(mes))

        last_row_filiais = start_row_list + len(lista_filiais) - 1
        wsPF.data_validation("B1", {
            "validate": "list",
            "source": f"=Pecas_Mes_Filtro!$H${start_row_list}:$H${last_row_filiais}",
            "input_title": "Filial",
            "input_message": "Selecione a filial para filtrar o gráfico",
        })
        wsPF.write("B1", "TODAS")

        n_meses = len(meses_unicos)
        for idx in range(n_meses):
            row = start_row_meses + idx  # linha Excel (1-based)
            wsPF.write_formula(
                row-1, 1,  # coluna B
                f'=IF($B$1="TODAS",'
                f'  SUMIF(Base_Pecas!$A:$A,$A{row},Base_Pecas!$C:$C),'
                f'  SUMIFS(Base_Pecas!$C:$C,Base_Pecas!$A:$A,$A{row},Base_Pecas!$B:$B,$B$1))'
            )

        wsPF.set_column("A:A", 12)
        wsPF.set_column("B:B", 22)
        wsPF.freeze_panes(3, 0)

        chartP = wb.add_chart({'type': 'line'})
        if n_meses > 0:
            last_row_series = start_row_meses + n_meses - 1
            chartP.add_series({
                'name':       'Peças por mês',
                'categories': f"=Pecas_Mes_Filtro!$A${start_row_meses}:$A${last_row_series}",
                'values':     f"=Pecas_Mes_Filtro!$B${start_row_meses}:$B${last_row_series}",
                'marker':     {'type': 'automatic'},
            })
        chartP.set_title({'name': 'Distribuição de peças por mês (filtrável por Filial)'})
        chartP.set_x_axis({'name': 'Mês'})
        chartP.set_y_axis({'name': 'Peças', 'num_format': '#,##0'})
        wsPF.insert_chart('D2', chartP, {'x_scale': 1.5, 'y_scale': 1.3})

        # ===== Aba de gráficos (executivo)
        wsG = wb.add_worksheet("Graficos")

        def _sheet_exists(name):
            try:
                _ = xw.sheets[name]
                return True
            except KeyError:
                return False

        def _insert_chart_safe(target_cell, chart, x_scale=1.2, y_scale=1.2):
            try:
                wsG.insert_chart(target_cell, chart, {'x_scale': x_scale, 'y_scale': y_scale})
            except Exception:
                traceback.print_exc()

        # 1) Prioridade por LINHA (stacked)
        if _sheet_exists("Prioridades_Linha") and len(prior_linha) > 0:
            wsPL = xw.sheets["Prioridades_Linha"]
            last_row_pl = len(prior_linha) + 1
            chart1 = wb.add_chart({'type': 'column', 'subtype': 'stacked'})
            for cat in ["BAIXA","MÉDIA","ALTA"]:
                if cat in prior_linha.columns:
                    col_idx = list(prior_linha.columns).index(cat)
                    chart1.add_series({
                        'name':       f"={wsPL.get_name()}!${col_name(col_idx)}$1",
                        'categories': f"={wsPL.get_name()}!$A$2:$A${last_row_pl}",
                        'values':     f"={wsPL.get_name()}!${col_name(col_idx)}$2:${col_name(col_idx)}${last_row_pl}",
                    })
            chart1.set_title({'name': 'Prioridade por LINHA'})
            chart1.set_x_axis({'name': 'LINHA'})
            chart1.set_y_axis({'name': 'QTD', 'num_format': '#,##0'})
            _insert_chart_safe('B2', chart1, 1.3, 1.2)

        # 2) Receita ajustada por GRIFFE
        if _sheet_exists("Receita_Griffe") and len(receita_griffe) > 0:
            wsRG = xw.sheets["Receita_Griffe"]
            last_row_rg = len(receita_griffe) + 1
            chart2 = wb.add_chart({'type': 'column'})
            chart2.add_series({
                'name':       'Receita Ajustada',
                'categories': f"={wsRG.get_name()}!$A$2:$A${last_row_rg}",
                'values':     f"={wsRG.get_name()}!$B$2:$B${last_row_rg}",
            })
            chart2.set_title({'name': 'Receita Ajustada por GRIFFE'})
            chart2.set_x_axis({'name': 'GRIFFE'})
            chart2.set_y_axis({'name': 'R$', 'num_format': 'R$ #,##0'})
            _insert_chart_safe('B20', chart2, 1.3, 1.2)

        # 3) Margem Projetada vs Margem Alvo (Scatter)
        if _sheet_exists("Base") and len(base_scatter) > 1 and set(["MARGEM_ALVO","MARGEM_PROJETADA"]).issubset(base_scatter.columns):
            wsB = xw.sheets["Base"]
            last_row_b = len(base_scatter) + 1
            cols_b = list(base_scatter.columns)
            col_alvo = col_name(cols_b.index("MARGEM_ALVO"))
            col_proj = col_name(cols_b.index("MARGEM_PROJETADA"))
            chart3 = wb.add_chart({'type': 'scatter', 'subtype': 'straight'})
            chart3.add_series({
                'name': 'Proj vs Alvo',
                'categories': f"={wsB.get_name()}!${col_alvo}$2:${col_alvo}${last_row_b}",
                'values':     f"={wsB.get_name()}!${col_proj}$2:${col_proj}${last_row_b}",
                'marker': {'type': 'circle', 'size': 4},
            })
            chart3.set_title({'name': 'Margem Projetada vs Margem Alvo'})
            chart3.set_x_axis({'name': 'Margem Alvo (%)', 'num_format': '0"%"'})
            chart3.set_y_axis({'name': 'Margem Projetada (%)', 'num_format': '0"%"'})
            _insert_chart_safe('J2', chart3, 1.2, 1.2)

        # 4) Receita x Lucro por GRIFFE
        if _sheet_exists("Receita_Lucro_Griffe") and len(rec_lucro_griffe) > 0:
            wsRLG = xw.sheets["Receita_Lucro_Griffe"]
            last_row_rlg = len(rec_lucro_griffe) + 1
            chart4 = wb.add_chart({'type': 'column'})
            chart4.add_series({
                'name':       'Receita Ajustada',
                'categories': f"={wsRLG.get_name()}!$A$2:$A${last_row_rlg}",
                'values':     f"={wsRLG.get_name()}!$B$2:$B${last_row_rlg}",
            })
            chart4.add_series({
                'name':       'Lucro Ajustado',
                'categories': f"={wsRLG.get_name()}!$A$2:$A${last_row_rlg}",
                'values':     f"={wsRLG.get_name()}!$C$2:$C${last_row_rlg}",
            })
            chart4.set_title({'name': 'Receita x Lucro por GRIFFE'})
            chart4.set_x_axis({'name': 'GRIFFE'})
            chart4.set_y_axis({'name': 'R$', 'num_format': 'R$ #,##0'})
            chart4.set_legend({'position': 'bottom'})
            _insert_chart_safe('J20', chart4, 1.2, 1.2)

        # 5) Distribuição STATUS_PROD (Pizza)
        if _sheet_exists("Status_Distrib") and len(status_dist) > 0:
            wsSD = xw.sheets["Status_Distrib"]
            last_row_sd = len(status_dist) + 1
            chart5 = wb.add_chart({'type': 'pie'})
            chart5.add_series({
                'name':       'Distribuição STATUS_PROD',
                'categories': f"={wsSD.get_name()}!$A$2:$A${last_row_sd}",
                'values':     f"={wsSD.get_name()}!$B$2:$B${last_row_sd}",
                'data_labels': {'percentage': True},
            })
            chart5.set_title({'name': 'STATUS_PROD (Mix da Coleção)'})
            _insert_chart_safe('B38', chart5, 1.1, 1.1)

        # 6) % abaixo da meta por LINHA (Barra)
        if _sheet_exists("Risco_Linha") and len(risco_pct_linha) > 0:
            wsRLP = xw.sheets["Risco_Linha"]
            last_row_rl = len(risco_pct_linha) + 1
            chart6 = wb.add_chart({'type': 'bar'})
            chart6.add_series({
                'name':       '% abaixo da meta',
                'categories': f"={wsRLP.get_name()}!$A$2:$A${last_row_rl}",
                'values':     f"={wsRLP.get_name()}!$B$2:$B${last_row_rl}",
                'data_labels': {'value': True},
            })
            chart6.set_title({'name': '% de itens abaixo da meta por LINHA'})
            chart6.set_x_axis({'name': '% de itens', 'num_format': '0"%"', 'major_unit': 10})
            chart6.set_y_axis({'name': 'LINHA'})
            _insert_chart_safe('B56', chart6, 1.3, 1.2)

        # 7) Receita Mensal (Linha)
        if _sheet_exists("Receita_Mensal") and len(receita_mensal) > 0:
            wsRM = xw.sheets["Receita_Mensal"]
            last_row_rm = len(receita_mensal) + 1
            chart7 = wb.add_chart({'type': 'line'})
            chart7.add_series({
                'name':       'Receita Ajustada',
                'categories': f"={wsRM.get_name()}!$A$2:$A${last_row_rm}",
                'values':     f"={wsRM.get_name()}!$B$2:$B${last_row_rm}",
                'marker': {'type': 'automatic'},
            })
            chart7.set_title({'name': 'Receita Ajustada por Mês'})
            chart7.set_x_axis({'name': 'ANO_MES'})
            chart7.set_y_axis({'name': 'R$', 'num_format': 'R$ #,##0'})
            _insert_chart_safe('J38', chart7, 1.2, 1.2)

        # ======= NOVOS GRÁFICOS POR FAIXA =======

        # 8) Receita por LINHA × FAIXA (empilhado)
        if len(receita_linha_faixa) > 0:
            pivot_rlf = receita_linha_faixa.pivot(index="LINHA", columns="FAIXA_PRECO", values="VAL_AJUST").fillna(0).reset_index()
            # ordena colunas das faixas na ordem executiva
            cols_faixa = [c for c in FAIXA_ORDER if c in pivot_rlf.columns]
            pivot_rlf = pivot_rlf.reindex(columns=["LINHA"] + cols_faixa)
            pivot_rlf.to_excel(xw, sheet_name="Receita_Linha_Faixa_Pivot", index=False)
            wsRLFP = xw.sheets["Receita_Linha_Faixa_Pivot"]
            last_row_rlfp = len(pivot_rlf) + 1
            chart8 = wb.add_chart({'type': 'column', 'subtype': 'stacked'})
            for j, col in enumerate(pivot_rlf.columns[1:], start=1):
                colL = col_name(j)
                chart8.add_series({
                    'name':       f"={wsRLFP.get_name()}!${colL}$1",
                    'categories': f"={wsRLFP.get_name()}!$A$2:$A${last_row_rlfp}",
                    'values':     f"={wsRLFP.get_name()}!${colL}$2:${colL}${last_row_rlfp}",
                })
            chart8.set_title({'name': 'Receita por LINHA × FAIXA'})
            chart8.set_x_axis({'name': 'LINHA'})
            chart8.set_y_axis({'name': 'R$', 'num_format': 'R$ #,##0'})
            _insert_chart_safe('N2', chart8, 1.2, 1.2)

        # 9) Margem média por FAIXA (coluna)
        if len(margem_media_faixa) > 0:
            wsMF = xw.sheets["Margem_Faixa"]
            last_row_mf = len(margem_media_faixa) + 1
            chart9 = wb.add_chart({'type': 'column'})
            chart9.add_series({
                'name': 'Margem média',
                'categories': f"={wsMF.get_name()}!$A$2:$A${last_row_mf}",
                'values':     f"={wsMF.get_name()}!$B$2:$B${last_row_mf}",
                'data_labels': {'value': True},
            })
            chart9.set_title({'name': 'Margem Média por Faixa'})
            chart9.set_x_axis({'name': 'Faixa'})
            chart9.set_y_axis({'name': 'Margem', 'num_format': '0"%"'})
            _insert_chart_safe('N20', chart9, 1.2, 1.2)

        # 10) Volume mensal por FAIXA (linha)
        if len(volume_mensal_faixa) > 0:
            pivot_vmf = volume_mensal_faixa.pivot(index="ANO_MES", columns="FAIXA_PRECO", values="QTDE_ALOCADA").fillna(0).reset_index()
            cols_faixa = [c for c in FAIXA_ORDER if c in pivot_vmf.columns]
            pivot_vmf = pivot_vmf.reindex(columns=["ANO_MES"] + cols_faixa)
            pivot_vmf.to_excel(xw, sheet_name="Volume_Mensal_Faixa_Pivot", index=False)
            wsVMFP = xw.sheets["Volume_Mensal_Faixa_Pivot"]
            last_row_vmfp = len(pivot_vmf) + 1
            chart10 = wb.add_chart({'type': 'line'})
            for j, col in enumerate(pivot_vmf.columns[1:], start=1):
                colL = col_name(j)
                chart10.add_series({
                    'name':       f"={wsVMFP.get_name()}!${colL}$1",
                    'categories': f"={wsVMFP.get_name()}!$A$2:$A${last_row_vmfp}",
                    'values':     f"={wsVMFP.get_name()}!${colL}$2:${colL}${last_row_vmfp}",
                    'marker': {'type': 'automatic'},
                })
            chart10.set_title({'name': 'Volume Mensal por Faixa'})
            chart10.set_x_axis({'name': 'Mês'})
            chart10.set_y_axis({'name': 'Peças', 'num_format': '#,##0'})
            _insert_chart_safe('N38', chart10, 1.2, 1.2)

        # 11) Margem média por LINHA × FAIXA (cluster)
        if len(margem_linha_faixa) > 0:
            pivot_mlf = margem_linha_faixa.pivot(index="LINHA", columns="FAIXA_PRECO", values="MARGEM_PROJETADA").fillna(0).reset_index()
            cols_faixa = [c for c in FAIXA_ORDER if c in pivot_mlf.columns]
            pivot_mlf = pivot_mlf.reindex(columns=["LINHA"] + cols_faixa)
            pivot_mlf.to_excel(xw, sheet_name="Margem_Linha_Faixa_Pivot", index=False)
            wsMLFP = xw.sheets["Margem_Linha_Faixa_Pivot"]
            last_row_mlff = len(pivot_mlf) + 1
            chart11 = wb.add_chart({'type': 'column'})  # cluster
            for j, col in enumerate(pivot_mlf.columns[1:], start=1):
                colL = col_name(j)
                chart11.add_series({
                    'name':       f"={wsMLFP.get_name()}!${colL}$1",
                    'categories': f"={wsMLFP.get_name()}!$A$2:$A${last_row_mlff}",
                    'values':     f"={wsMLFP.get_name()}!${colL}$2:${colL}${last_row_mlff}",
                })
            chart11.set_title({'name': 'Margem Média por LINHA × FAIXA'})
            chart11.set_x_axis({'name': 'LINHA'})
            chart11.set_y_axis({'name': 'Margem', 'num_format': '0"%"'})
            _insert_chart_safe('B74', chart11, 1.3, 1.2)

        # 12) Status por FAIXA (empilhado)
        status_faixa_pivot = (
            status_faixa_raw.pivot(index="FAIXA_PRECO", columns="STATUS_PROD", values="QTD")
            .fillna(0).reset_index()
            if len(status_faixa_raw) > 0 else pd.DataFrame()
        )
        if len(status_faixa_pivot) > 0:
            status_cols = [c for c in ["PRODUZIR","REVER PREÇO","REVER PREÇO (OUTLIER)","BAIXA PRIORIDADE"] if c in status_faixa_pivot.columns]
            status_faixa_pivot = status_faixa_pivot.reindex(columns=["FAIXA_PRECO"] + status_cols)
            status_faixa_pivot.to_excel(xw, sheet_name="Status_Faixa_Pivot", index=False)
            wsSFP = xw.sheets["Status_Faixa_Pivot"]
            last_row_sfpp = len(status_faixa_pivot) + 1
            chart12 = wb.add_chart({'type': 'column', 'subtype': 'stacked'})
            for j, col in enumerate(status_faixa_pivot.columns[1:], start=1):
                colL = col_name(j)
                chart12.add_series({
                    'name':       f"={wsSFP.get_name()}!${colL}$1",
                    'categories': f"={wsSFP.get_name()}!$A$2:$A${last_row_sfpp}",
                    'values':     f"={wsSFP.get_name()}!${colL}$2:${colL}${last_row_sfpp}",
                })
            chart12.set_title({'name': 'Status por Faixa de Preço'})
            chart12.set_x_axis({'name': 'Faixa'})
            chart12.set_y_axis({'name': 'Itens', 'num_format': '#,##0'})
            _insert_chart_safe('J74', chart12, 1.2, 1.2)

        # 13) Mix de Faixas por GRIFFE (empilhado em %)
        if len(mix_griffe_faixa) > 0:
            pivot_mgf = mix_griffe_faixa.pivot(index="GRIFFE", columns="FAIXA_PRECO", values="PCT").fillna(0).reset_index()
            cols_faixa = [c for c in FAIXA_ORDER if c in pivot_mgf.columns]
            pivot_mgf = pivot_mgf.reindex(columns=["GRIFFE"] + cols_faixa)
            pivot_mgf.to_excel(xw, sheet_name="Mix_Griffe_Faixa_Pivot", index=False)
            wsMGFP = xw.sheets["Mix_Griffe_Faixa_Pivot"]
            last_row_mgfp = len(pivot_mgf) + 1
            chart13 = wb.add_chart({'type': 'column', 'subtype': 'stacked'})
            for j, col in enumerate(pivot_mgf.columns[1:], start=1):
                colL = col_name(j)
                chart13.add_series({
                    'name':       f"={wsMGFP.get_name()}!${colL}$1",
                    'categories': f"={wsMGFP.get_name()}!$A$2:$A${last_row_mgfp}",
                    'values':     f"={wsMGFP.get_name()}!${colL}$2:${colL}${last_row_mgfp}",
                })
            chart13.set_title({'name': 'Mix de Faixas por GRIFFE'})
            chart13.set_x_axis({'name': 'GRIFFE'})
            chart13.set_y_axis({'name': 'Participação', 'num_format': '0"%"'})
            _insert_chart_safe('N74', chart13, 1.2, 1.2)

        # ===== Realces úteis =====
        if len(top20) > 0:
            wsTop = xw.sheets["Top_Produtos"]
            cols_t = list(top20.columns)
            if "PRIORIDADE" in cols_t:
                cprio = col_name(cols_t.index("PRIORIDADE"))
                last_row = len(top20) + 1
                wsTop.conditional_format(f"A2:{col_name(len(cols_t)-1)}{last_row}", {
                    'type': 'formula',
                    'criteria': f'=${cprio}2="ALTA"',
                    'format': high_fmt
                })

    # gravação segura
    try:
        _safe_write_excel(OUT_XLSX, build_workbook)
    except Exception:
        print("[INFO] Salvando CSVs de fallback por causa do erro acima.")
        # CSVs essenciais (para não te deixar na mão)
        kpis_geral.to_csv(IN_DIR / "rel_kpis_gerais.csv", index=False, encoding="utf-8-sig")
        kpis_gl.to_csv(IN_DIR / "rel_kpis_griffe_linha.csv", index=False, encoding="utf-8-sig")
        top20.to_csv(IN_DIR / "rel_top_produtos.csv", index=False, encoding="utf-8-sig")
        riscos.to_csv(IN_DIR / "rel_riscos.csv", index=False, encoding="utf-8-sig")
        prior_linha.to_csv(IN_DIR / "rel_prioridades_linha.csv", index=False, encoding="utf-8-sig")
        prior_griffe.to_csv(IN_DIR / "rel_prioridades_griffe.csv", index=False, encoding="utf-8-sig")
        receita_griffe.to_csv(IN_DIR / "rel_receita_griffe.csv", index=False, encoding="utf-8-sig")
        base_scatter.to_csv(IN_DIR / "rel_base_scatter.csv", index=False, encoding="utf-8-sig")
        rec_lucro_griffe.to_csv(IN_DIR / "rel_receita_lucro_griffe.csv", index=False, encoding="utf-8-sig")
        status_dist.to_csv(IN_DIR / "rel_status_distrib.csv", index=False, encoding="utf-8-sig")
        risco_pct_linha.to_csv(IN_DIR / "rel_risco_pct_linha.csv", index=False, encoding="utf-8-sig")
        receita_mensal.to_csv(IN_DIR / "rel_receita_mensal.csv", index=False, encoding="utf-8-sig")
        base_pecas.to_csv(IN_DIR / "rel_base_pecas.csv", index=False, encoding="utf-8-sig")
        receita_linha_faixa.to_csv(IN_DIR / "rel_receita_linha_faixa.csv", index=False, encoding="utf-8-sig")
        margem_media_faixa.to_csv(IN_DIR / "rel_margem_faixa.csv", index=False, encoding="utf-8-sig")
        volume_mensal_faixa.to_csv(IN_DIR / "rel_volume_mensal_faixa.csv", index=False, encoding="utf-8-sig")
        margem_linha_faixa.to_csv(IN_DIR / "rel_margem_linha_faixa.csv", index=False, encoding="utf-8-sig")
        mix_griffe_faixa.to_csv(IN_DIR / "rel_mix_griffe_faixa.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
