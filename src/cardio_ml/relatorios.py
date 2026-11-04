"""Registro dos resultados: texto legível (resultados.txt), JSON estruturado e tabelas CSV."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path


class Relatorio:
    def __init__(self, pasta_saida: Path):
        self.pasta = Path(pasta_saida)
        self.linhas: list[str] = []
        self.resultados: dict = {}
        self.inicio = time.time()

    def log(self, msg="") -> None:
        print(msg, flush=True)
        self.linhas.append(str(msg))

    def titulo(self, texto: str) -> None:
        self.log("=" * 78)
        self.log(texto)
        self.log("=" * 78)

    def progresso(self, texto: str) -> None:
        print(f"[{time.time() - self.inicio:7.1f}s] {texto}", flush=True)

    def csv(self, nome: str, cabecalho: list[str], linhas: list[list]) -> None:
        (self.pasta / "tabelas").mkdir(parents=True, exist_ok=True)
        with open(self.pasta / "tabelas" / nome, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(cabecalho)
            w.writerows(linhas)

    def gravar(self) -> None:
        self.log(f"Tempo total: {time.time() - self.inicio:.0f} s")
        self.resultados["tempo_total_s"] = round(time.time() - self.inicio)
        self.pasta.mkdir(parents=True, exist_ok=True)
        (self.pasta / "resultados.txt").write_text("\n".join(self.linhas), encoding="utf-8")
        with open(self.pasta / "resultados.json", "w", encoding="utf-8") as f:
            json.dump(self.resultados, f, ensure_ascii=False, indent=1, default=str)
        print(f"\nResultados gravados em {self.pasta}")


def folhas(obj, caminho=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from folhas(v, f"{caminho}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from folhas(v, f"{caminho}[{i}]")
    else:
        yield caminho, obj


def comparar(referencia: Path, atual: Path, tolerancia: float = 1e-12) -> tuple[int, list]:
    """Compara cada valor da execução de referência com o da execução atual.

    Só as chaves presentes na referência são conferidas: análises novas não contam como
    diferença. Ambiente e tempo de execução são ignorados.
    """
    A = json.loads(Path(referencia).read_text(encoding="utf-8"))
    B = json.loads(Path(atual).read_text(encoding="utf-8"))
    for d in (A, B):
        d.pop("tempo_total_s", None)
        d.pop("ambiente", None)
    fa, fb = dict(folhas(A)), dict(folhas(B))
    diferentes = []
    for k in sorted(fa):
        x, y = fa[k], fb.get(k, "<ausente>")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if abs(x - y) > tolerancia:
                diferentes.append((k, x, y))
        elif x != y:
            diferentes.append((k, x, y))
    return len(fa), diferentes
