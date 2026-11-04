"""Leitura do arquivo de configuração (config.toml)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    semente: int
    pasta_dados: Path
    pasta_saida: Path
    url_base: str
    proporcao_teste: float
    limiar_descarte: float
    dobras_externas: int
    dobras_internas: int
    metrica_selecao: str
    n_bootstrap: int
    n_permutacoes: int
    limiar_padrao: float
    metas_revocacao: tuple[float, ...]


def carregar_config(caminho: str | Path | None = None, raiz: Path = RAIZ) -> Config:
    caminho = Path(caminho) if caminho else raiz / "config.toml"
    with open(caminho, "rb") as f:
        c = tomllib.load(f)
    g, a = c["geral"], c["avaliacao"]
    return Config(
        semente=int(g["semente"]),
        pasta_dados=(raiz / g["pasta_dados"]).resolve(),
        pasta_saida=(raiz / g["pasta_saida"]).resolve(),
        url_base=g["url_base"],
        proporcao_teste=float(c["divisao"]["proporcao_teste"]),
        limiar_descarte=float(c["preprocessamento"]["limiar_descarte"]),
        dobras_externas=int(c["validacao"]["dobras_externas"]),
        dobras_internas=int(c["validacao"]["dobras_internas"]),
        metrica_selecao=c["validacao"]["metrica_selecao"],
        n_bootstrap=int(a["n_bootstrap"]),
        n_permutacoes=int(a["n_permutacoes"]),
        limiar_padrao=float(a["limiar_padrao"]),
        metas_revocacao=tuple(float(m) for m in a["metas_revocacao"]),
    )
