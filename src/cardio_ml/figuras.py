"""Figuras da análise exploratória, dos modelos e o diagrama do pipeline."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import roc_curve

from .dados import NOMES_PT, NUMERICAS

ROTULO_CLASSE = {0: "Ausência", 1: "Presença"}


def salvar(fig, pasta: Path, nome: str) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(pasta / nome, dpi=300)
    plt.close(fig)


def exploratorias(dados, pasta: Path) -> None:
    sns.set_theme(style="whitegrid")
    rot = dados.assign(Classe=dados["alvo"].map(ROTULO_CLASSE))
    ordem = [ROTULO_CLASSE[0], ROTULO_CLASSE[1]]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=rot, x="age", hue="Classe", hue_order=ordem, multiple="dodge", bins=20, ax=ax)
    ax.set_xlabel("Idade (anos)"); ax.set_ylabel("Frequência")
    salvar(fig, pasta, "fig_idade.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=rot.assign(Sexo=rot["sex"].map({0: "Feminino", 1: "Masculino"})),
                  x="Sexo", hue="Classe", hue_order=ordem, ax=ax)
    ax.set_xlabel("Sexo"); ax.set_ylabel("Frequência")
    salvar(fig, pasta, "fig_sexo.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    nomes_cp = {1: "Angina típica", 2: "Angina atípica", 3: "Dor não anginosa", 4: "Assintomático"}
    sns.countplot(data=rot.assign(Dor=rot["cp"].map(nomes_cp)), x="Dor", hue="Classe",
                  order=list(nomes_cp.values()), hue_order=ordem, ax=ax)
    ax.set_xlabel("Tipo de dor torácica"); ax.set_ylabel("Frequência")
    salvar(fig, pasta, "fig_dor.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=rot, x="Classe", y="thalach", order=ordem, ax=ax)
    ax.set_xlabel("Classe"); ax.set_ylabel("Frequência cardíaca máxima (bpm)")
    salvar(fig, pasta, "fig_thalach.png")

    # rótulos em português (nome curto, com o nome original da base entre parênteses)
    corr = dados[NUMERICAS + ["alvo"]].corr()
    rotulos = [f"{NOMES_PT[c][1]} ({c})" if c != "alvo" else "Diagnóstico (alvo)" for c in corr.columns]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=rotulos, yticklabels=rotulos)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    salvar(fig, pasta, "fig_correlacao.png")


def modelos(probs, y_te, teste, melhor, pasta: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for nome, p in probs.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, label=f"{nome} (AUC = {teste[nome]['roc_auc']:.3f})".replace(".", ","))
    ax.plot([0, 1], [0, 1], "k--", label="Classificador aleatório")
    ax.set_xlabel("Taxa de falsos positivos (1 − especificidade)")
    ax.set_ylabel("Taxa de verdadeiros positivos (revocação)")
    ax.legend(loc="lower right")
    salvar(fig, pasta, "fig_roc.png")

    r = teste[melhor]
    cm = np.array([[r["vn"], r["fp"]], [r["fn"], r["vp"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=list(ROTULO_CLASSE.values()), yticklabels=list(ROTULO_CLASSE.values()))
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
    salvar(fig, pasta, "fig_matriz_confusao.png")


def diagrama_pipeline(pasta: Path) -> None:
    """Fluxo de dados do artefato: da carga das bases às tabelas e figuras do trabalho."""
    etapas = [
        ("dados.py", "Carga das 4 bases\nUCI (920 registros)"),
        ("dados.py", "Divisão estratificada\n80/20 (semente 42)"),
        ("preprocessamento.py", "Descarte por ausência\n(só no treino)\nimputação, one-hot,\npadronização"),
        ("treino.py / modelos.py", "Validação cruzada\naninhada 10×5\n+ busca em grade"),
        ("avaliacao.py / limiares.py", "Teste único: métricas,\nbootstrap, diferenças,\nlimiares, baselines"),
        ("relatorios.py / figuras.py", "resultados.json/.txt,\ntabelas CSV\ne figuras"),
    ]
    fig, ax = plt.subplots(figsize=(12, 2.9))
    ax.set_xlim(0, 12); ax.set_ylim(0, 2.9); ax.axis("off")
    larg, alt, passo = 1.75, 1.55, 2.0
    for i, (modulo, texto) in enumerate(etapas):
        x = 0.12 + i * passo
        ax.add_patch(FancyBboxPatch((x, 0.55), larg, alt, boxstyle="round,pad=0.03",
                                    fc="#eef3f8", ec="#2b4c6f", lw=1.2))
        ax.text(x + larg / 2, 0.55 + alt / 2, texto, ha="center", va="center", fontsize=8.5)
        ax.text(x + larg / 2, 0.30, modulo, ha="center", va="center", fontsize=7.5,
                family="monospace", color="#2b4c6f")
        if i < len(etapas) - 1:
            ax.add_patch(FancyArrowPatch((x + larg + 0.02, 0.55 + alt / 2),
                                         (x + passo - 0.02, 0.55 + alt / 2),
                                         arrowstyle="-|>", mutation_scale=12, color="#2b4c6f"))
    ax.text(6, 2.65, "python -m cardio_ml run   (configuração: config.toml)", ha="center",
            fontsize=9, family="monospace")
    salvar(fig, pasta, "fig_pipeline.png")
