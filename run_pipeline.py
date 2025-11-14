# run_pipeline.py (força UTF-8 ao capturar logs das etapas)
import subprocess
import datetime
from pathlib import Path
import sys
import os

# ====== CONFIG ======
BASE_DIR    = Path(r"C:\Users\monta\OneDrive\Documentos\Meta\METAxSACADA\ML-Sacada")
SCRIPTS_DIR = BASE_DIR / "scripts"
SAIDAS_DIR  = BASE_DIR / "saidas"
LOG_FILE    = SAIDAS_DIR / "pipeline_log.txt"

ETAPAS = [
    ("Etapa 1 - Ingestão e Limpeza", "etapa1_ingestao_limpeza.py"),
    ("Etapa 2 - Previsão 2026", "etapa2_previsao_2026.py"),
    ("Etapa 3 - Planejamento Comercial", "etapa3_planejamento_2026.py"),
    ("Etapa 4 - Validação e Decisão", "etapa4_validacao_decisao.py"),
    ("Etapa 5 - Relatório Executivo", "etapa5_relatorio_executivo.py"),
]

# Ambiente com UTF-8 para os subprocessos (evita erros de decode/encode)
CHILD_ENV = os.environ.copy()
CHILD_ENV["PYTHONIOENCODING"] = "utf-8"

def log(msg: str):
    SAIDAS_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    # imprime no console (tenta UTF-8, senão substitui)
    try:
        sys.stdout.write(line)
    except UnicodeEncodeError:
        sys.stdout.write(line.encode("utf-8", "replace").decode("utf-8", "replace"))
    # grava no arquivo em UTF-8
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)

def run_step(nome, script):
    log(f"🔹 Iniciando {nome}...")
    start = datetime.datetime.now()

    script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        log(f"❌ Script não encontrado: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    # Forçar encoding UTF-8 ao capturar stdout/stderr
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",       # substitui caracteres inválidos
        env=CHILD_ENV,
    )

    duration = (datetime.datetime.now() - start).total_seconds()

    if result.stdout:
        log(result.stdout.rstrip("\n"))
    if result.stderr:
        log("[stderr]")
        log(result.stderr.rstrip("\n"))

    if result.returncode == 0:
        log(f"✅ {nome} concluída em {duration:.1f}s")
        return True
    else:
        log(f"⚠️ Erro em {nome}: código {result.returncode}")
        return False

def main():
    log("🚀 Início do pipeline META x SACADA")
    for nome, script in ETAPAS:
        if not run_step(nome, script):
            log(f"⛔ Interrompendo pipeline em {nome}")
            break
    log("🏁 Pipeline finalizado.\n")

if __name__ == "__main__":
    # tenta configurar stdout do próprio runner para UTF-8 (opcional)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
