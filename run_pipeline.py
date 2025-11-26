import subprocess
import datetime
from pathlib import Path
import sys
import os

# ================== CONFIG ==================
BASE_DIR = Path(__file__).resolve().parent
SAIDAS_DIR = BASE_DIR / "saidas"
LOG_FILE = SAIDAS_DIR / "pipeline_log.txt"

# Scripts na ordem do pipeline local
ETAPAS = [
    ("Etapa 1 - Ingestao e Limpeza", "Etapa 1.py"),
    ("Etapa 1B - Estoque Mensal", "Etapa1B.py"),
    ("Etapa 2 - Previsao 2026", "Etapa2.py"),
    ("Etapa 3 - Planejamento Comercial", "etapa3_planejamento_2026.py"),
    ("Etapa 4 - Validacao e Decisao", "etapa4_validacao_decisao.py"),
    ("Etapa 5 - Relatorio Executivo", "etapa5_relatorio_executivo.py"),
]

# Ambiente com UTF-8 para subprocessos
CHILD_ENV = os.environ.copy()
CHILD_ENV["PYTHONIOENCODING"] = "utf-8"


def log(msg: str):
    SAIDAS_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    try:
        sys.stdout.write(line)
    except UnicodeEncodeError:
        sys.stdout.write(line.encode("ascii", "ignore").decode("ascii"))
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def run_step(nome, script):
    log(f"Iniciando {nome} ({script})")
    script_path = BASE_DIR / script
    if not script_path.exists():
        log(f"[ERRO] Script nao encontrado: {script_path}")
        return False
    cmd = [sys.executable, str(script_path)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            env=CHILD_ENV,
            capture_output=True,
            text=True,
        )
    except Exception as e:
        log(f"[EXCECAO] Falha ao executar {script}: {e}")
        return False

    if proc.stdout:
        log(proc.stdout.strip())
    if proc.stderr:
        log(f"[STDERR] {proc.stderr.strip()}")

    if proc.returncode != 0:
        log(f"[ERRO] {nome} falhou com exit code {proc.returncode}")
        return False

    log(f"Concluido {nome}")
    return True


def main():
    ok = True
    for nome, script in ETAPAS:
        if not run_step(nome, script):
            ok = False
            break
    if ok:
        log("Pipeline finalizado com sucesso.")
    else:
        log("Pipeline encerrado com erro.")


if __name__ == "__main__":
    main()
