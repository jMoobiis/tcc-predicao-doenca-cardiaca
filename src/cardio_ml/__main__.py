"""Ponto de entrada de linha de comando.

    python -m cardio_ml run                      reproduz todos os resultados do zero
    python -m cardio_ml run --rapido             só o experimento principal (sem os complementares)
    python -m cardio_ml verificar REF.json       compara saida/resultados.json com uma execução de referência
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import carregar_config


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cardio_ml", description="Pipeline do TCC: predição de doença cardíaca")
    p.add_argument("--config", help="caminho do config.toml (padrão: raiz do projeto)")
    sub = p.add_subparsers(dest="comando", required=True)
    r = sub.add_parser("run", help="executa o pipeline completo e grava saida/")
    r.add_argument("--rapido", action="store_true", help="pula os experimentos complementares")
    v = sub.add_parser("verificar", help="confere valor a valor contra uma execução de referência")
    v.add_argument("referencia", type=Path)
    v.add_argument("--atual", type=Path, help="padrão: saida/resultados.json")
    a = p.parse_args(argv)
    cfg = carregar_config(a.config)

    if a.comando == "run":
        from .experimentos import executar
        executar(cfg, completo=not a.rapido)
        return 0

    from .relatorios import comparar
    atual = a.atual or cfg.pasta_saida / "resultados.json"
    n, dif = comparar(a.referencia, atual)
    print(f"Valores da referência conferidos: {n}; diferentes: {len(dif)}")
    for k, x, y in dif[:30]:
        print(f"  {k}: {x} x {y}")
    return 0 if not dif else 1


if __name__ == "__main__":
    sys.exit(main())
