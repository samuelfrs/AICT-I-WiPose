"""
Métricas da classificação de ações (tarefas c e e), reusadas pela rede neural.

Duas visões, sempre reportadas juntas:

- **por janela** — cada janela de quadros é uma amostra. Tem mais amostras, mas
  janelas da mesma sequência são correlacionadas.
- **por sequência** — as probabilidades das janelas de uma sequência são
  promediadas e a classe mais provável vira a predição da sequência. É a que
  responde "qual ação a pessoa executou".

O intervalo de confiança é **sempre por bootstrap de sequências**: reamostrar
janelas trataria como independentes amostras que não são, e estreitaria o IC.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def agregar_por_sequencia(
    probabilidades: np.ndarray, y: np.ndarray, grupos: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Média das probabilidades das janelas de cada sequência.

    Devolve ``(y_seq, pred_seq, ids_seq)``, um elemento por sequência, na ordem
    crescente de ``grupos``. Falha se uma sequência tiver rótulos misturados.
    """
    ids, inverso = np.unique(grupos, return_inverse=True)
    soma = np.zeros((len(ids), probabilidades.shape[1]))
    np.add.at(soma, inverso, probabilidades)
    contagem = np.bincount(inverso, minlength=len(ids))[:, None]
    media = soma / contagem

    y_seq = np.empty(len(ids), dtype=y.dtype)
    y_seq[inverso] = y
    if not np.array_equal(y_seq[inverso], y):
        raise ValueError("Há sequência com janelas de rótulos diferentes.")
    return y_seq, media.argmax(axis=1), ids


def ic_bootstrap(
    acertos: np.ndarray,
    grupos: Optional[np.ndarray] = None,
    n_reamostras: int = 2000,
    nivel: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """IC por bootstrap percentil da acurácia, reamostrando **sequências**.

    ``acertos``: 0/1 por amostra. ``grupos``: sequência de cada amostra; sem ele,
    cada amostra é a própria sequência. A acurácia de cada reamostra é a fração
    de acertos sobre todas as amostras das sequências sorteadas.
    """
    acertos = np.asarray(acertos, dtype=np.float64)
    if acertos.size == 0:
        raise ValueError("Sem amostras para o bootstrap.")
    if grupos is None:
        grupos = np.arange(len(acertos))
    _, inverso = np.unique(grupos, return_inverse=True)
    soma = np.bincount(inverso, weights=acertos)
    contagem = np.bincount(inverso).astype(np.float64)

    gerador = np.random.default_rng(seed)
    sorteio = gerador.integers(0, len(soma), size=(n_reamostras, len(soma)))
    acuracias = soma[sorteio].sum(axis=1) / contagem[sorteio].sum(axis=1)
    alfa = (1.0 - nivel) / 2.0
    baixo, alto = np.quantile(acuracias, [alfa, 1.0 - alfa])
    return float(baixo), float(alto)


def avaliar(
    probabilidades: np.ndarray,
    y: np.ndarray,
    grupos: np.ndarray,
    n_classes: int,
    seed: int = 0,
    n_reamostras: int = 2000,
) -> Dict:
    """Acurácia, F1 macro, IC 95% e matriz de confusão, por janela e por sequência."""
    pred = probabilidades.argmax(axis=1)
    acertos = (pred == y).astype(np.float64)
    y_seq, pred_seq, _ = agregar_por_sequencia(probabilidades, y, grupos)
    acertos_seq = (pred_seq == y_seq).astype(np.float64)
    rotulos = np.arange(n_classes)

    return {
        "por_janela": {
            "n": int(len(y)),
            "acuracia": float(acertos.mean()),
            "ic95": ic_bootstrap(acertos, grupos, n_reamostras, seed=seed),
            "f1_macro": float(f1_score(y, pred, labels=rotulos, average="macro",
                                       zero_division=0)),
            "confusao": confusion_matrix(y, pred, labels=rotulos).tolist(),
        },
        "por_sequencia": {
            "n": int(len(y_seq)),
            "acuracia": float(acertos_seq.mean()),
            "ic95": ic_bootstrap(acertos_seq, None, n_reamostras, seed=seed),
            "f1_macro": float(f1_score(y_seq, pred_seq, labels=rotulos,
                                       average="macro", zero_division=0)),
            "confusao": confusion_matrix(y_seq, pred_seq, labels=rotulos).tolist(),
        },
        "acaso": 1.0 / n_classes,
    }


def salvar_matriz_confusao(
    confusao: np.ndarray,
    rotulos: Sequence[str],
    caminho: Union[str, Path],
    titulo: str = "",
) -> Path:
    """Matriz de confusão **normalizada por linha** (recall por classe), em PNG.

    Escala sequencial de um só tom: a diagonal escura é o acerto; manchas fora
    dela mostram quais ações se confundem.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confusao = np.asarray(confusao, dtype=np.float64)
    totais = confusao.sum(axis=1, keepdims=True)
    normalizada = np.divide(confusao, totais, out=np.zeros_like(confusao),
                            where=totais > 0)

    n = len(rotulos)
    fig, ax = plt.subplots(figsize=(0.38 * n + 2.5, 0.38 * n + 1.8))
    imagem = ax.imshow(normalizada, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n), rotulos, rotation=90, fontsize=8)
    ax.set_yticks(range(n), rotulos, fontsize=8)
    ax.set_xlabel("ação predita")
    ax.set_ylabel("ação verdadeira")
    if titulo:
        ax.set_title(titulo, fontsize=10)
    # Rótulo só onde há massa relevante: número em toda célula vira ruído.
    for i in range(n):
        for j in range(n):
            valor = normalizada[i, j]
            if valor >= 0.10:
                ax.text(j, i, f"{100 * valor:.0f}", ha="center", va="center",
                        fontsize=6, color="white" if valor > 0.55 else "#333333")
    barra = fig.colorbar(imagem, ax=ax, fraction=0.04, pad=0.02)
    barra.set_label("fração da linha")
    fig.tight_layout()

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return caminho
