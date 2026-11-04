"""Os cinco classificadores comparados e as grades de hiperparâmetros testadas."""
from __future__ import annotations

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import Config
from .preprocessamento import montar_preprocessador


def catalogo(semente: int) -> dict:
    """nome -> (estimador base, grade de busca). A ordem define a ordem das tabelas."""
    return {
        "Regressão Logística": (
            LogisticRegression(max_iter=1000, random_state=semente),
            {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__solver": ["liblinear", "lbfgs"]}),
        "Árvore de Decisão": (
            DecisionTreeClassifier(random_state=semente),
            {"clf__criterion": ["gini", "entropy"], "clf__max_depth": [3, 5, 7, 10, None],
             "clf__min_samples_split": [2, 5, 10], "clf__min_samples_leaf": [1, 2, 5]}),
        "Random Forest": (
            RandomForestClassifier(random_state=semente),
            {"clf__n_estimators": [100, 200, 500], "clf__max_depth": [5, 10, None],
             "clf__max_features": ["sqrt", "log2"], "clf__min_samples_leaf": [1, 2, 4]}),
        "SVM": (
            # a SVM não estima probabilidade: calibração sigmoide (Platt) por validação
            # cruzada interna; ensemble=False mantém um único SVC treinado no conjunto inteiro
            CalibratedClassifierCV(SVC(), method="sigmoid", ensemble=False),
            # gamma só existe no kernel RBF: duas grades evitam combinações repetidas
            [{"clf__estimator__kernel": ["linear"], "clf__estimator__C": [0.1, 1, 10, 100]},
             {"clf__estimator__kernel": ["rbf"], "clf__estimator__C": [0.1, 1, 10, 100],
              "clf__estimator__gamma": ["scale", "auto", 0.01, 0.1]}]),
        "k-NN": (
            KNeighborsClassifier(),
            # minkowski com p=2 é a própria distância euclidiana: retirada da grade
            {"clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
             "clf__weights": ["uniform", "distance"],
             "clf__metric": ["euclidean", "manhattan"]}),
    }


def busca(nome: str, usados, cfg: Config) -> GridSearchCV:
    estimador, grade = catalogo(cfg.semente)[nome]
    pipe = Pipeline([("prep", montar_preprocessador(usados)), ("clf", clone(estimador))])
    interna = StratifiedKFold(n_splits=cfg.dobras_internas, shuffle=True, random_state=cfg.semente)
    return GridSearchCV(pipe, grade, scoring=cfg.metrica_selecao, cv=interna, n_jobs=-1)


def limpar(params: dict) -> dict:
    return {k.replace("clf__", "").replace("estimator__", ""): v for k, v in params.items()}


def tamanho_grade(grade) -> int:
    grades = grade if isinstance(grade, list) else [grade]
    total = 0
    for g in grades:
        n = 1
        for v in g.values():
            n *= len(v)
        total += n
    return total


def descrever_hiperparametros(finais: dict, cfg: Config) -> dict:
    """Grade testada, configuração escolhida e TODOS os parâmetros efetivos do classificador."""
    simples = (int, float, str, bool, type(None))
    saida = {}
    for nome, (estimador, grade) in catalogo(cfg.semente).items():
        clf = finais[nome].best_estimator_.named_steps["clf"]
        efetivos = {k: v for k, v in clf.get_params(deep=True).items() if isinstance(v, simples)}
        saida[nome] = {
            "classe": type(clf).__name__,
            "grade": [limpar(g) for g in (grade if isinstance(grade, list) else [grade])],
            "n_combinacoes": tamanho_grade(grade),
            "escolhidos": limpar(finais[nome].best_params_),
            "parametros_efetivos": {k.replace("estimator__", "SVC."): v for k, v in efetivos.items()},
        }
    return saida
