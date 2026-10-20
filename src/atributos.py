"""
Janelas de quadros e atributos estatísticos do CSI (entrada do baseline, tarefa c).

A unidade de classificação é uma **janela de quadros consecutivos** de uma
sequência (padrão: 30 quadros = 3 s, sem sobreposição -> 9 janelas por sequência,
os 27 quadros finais ficam de fora).

Para cada janela, os atributos são **média e desvio padrão** por antena ×
subportadora, calculados sobre todos os quadros × amostras temporais da janela,
para amplitude e para fase sanitizada::

    [amp_média (342) | amp_desvio (342) | fase_média (342) | fase_desvio (342)]

São 4 × 3 × 114 = 1.368 atributos. A média capta o perfil estático do canal; o
desvio, o quanto o corpo em movimento o perturba.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.mmfi import N_ANTENAS, N_QUADROS, N_SUBPORTADORAS

N_ATRIBUTOS = 4 * N_ANTENAS * N_SUBPORTADORAS  # 1.368


def janelas_de_sequencia(
    n_quadros: int = N_QUADROS, janela: int = 30, passo: int = None
) -> List[Tuple[int, int]]:
    """Intervalos ``[inicio, fim)`` das janelas que cabem inteiras na sequência.

    ``passo`` padrão = ``janela`` (sem sobreposição). Sobra no fim é descartada.
    """
    if not isinstance(janela, (int, np.integer)) or not 1 <= janela <= n_quadros:
        raise ValueError(
            f"janela inválida: {janela!r}. Faixa válida: 1 a {n_quadros} quadros."
        )
    if passo is None:
        passo = janela
    if not isinstance(passo, (int, np.integer)) or passo < 1:
        raise ValueError(f"passo inválido: {passo!r}. Deve ser ≥ 1.")
    return [(i, i + janela) for i in range(0, n_quadros - janela + 1, passo)]


def extrair_atributos(amp: np.ndarray, fase: np.ndarray) -> np.ndarray:
    """Atributos de uma ou mais janelas.

    Recebe ``(quadros, 3, 114, 10)`` ou ``(n_janelas, quadros, 3, 114, 10)`` e
    devolve ``(1368,)`` ou ``(n_janelas, 1368)``, em float32.
    """
    amp = np.asarray(amp)
    fase = np.asarray(fase)
    if amp.shape != fase.shape:
        raise ValueError(f"amp {amp.shape} e fase {fase.shape} com formas diferentes.")
    if amp.ndim not in (4, 5) or amp.shape[-3:-1] != (N_ANTENAS, N_SUBPORTADORAS):
        raise ValueError(
            f"Esperado (..., quadros, {N_ANTENAS}, {N_SUBPORTADORAS}, tempo), "
            f"obtido {amp.shape}."
        )
    unica = amp.ndim == 4
    if unica:
        amp, fase = amp[None], fase[None]

    # Eixos reduzidos: quadros (1) e amostras temporais (4). Acumula em float64
    # para o desvio não perder precisão somando 300 valores em float32.
    partes = []
    for sinal in (amp, fase):
        sinal = sinal.astype(np.float64)
        partes.append(sinal.mean(axis=(1, 4)).reshape(len(sinal), -1))
        partes.append(sinal.std(axis=(1, 4)).reshape(len(sinal), -1))
    atributos = np.concatenate(partes, axis=1).astype(np.float32)
    return atributos[0] if unica else atributos


def montar_conjunto(
    amp: np.ndarray,
    fase: np.ndarray,
    linhas: Dict[Tuple[str, str], int],
    sequencias: Sequence[Tuple[str, str]],
    acoes: Sequence[str],
    janela: int = 30,
    passo: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monta ``(X, y, grupos)`` a partir do cache, uma sequência por vez.

    - ``X``: ``(n_janelas_total, 1368)``
    - ``y``: índice da ação em ``acoes``
    - ``grupos``: índice da sequência em ``sequencias`` (para agregar e para o
      bootstrap — janelas da mesma sequência não são independentes)

    Lê do memmap só a sequência da vez (~4 MB por array), nunca o cache inteiro.
    """
    intervalos = janelas_de_sequencia(amp.shape[1], janela, passo)
    codigo = {a: i for i, a in enumerate(acoes)}
    n = len(sequencias) * len(intervalos)
    X = np.empty((n, N_ATRIBUTOS), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    grupos = np.empty(n, dtype=np.int64)

    k = 0
    for g, (sujeito, acao) in enumerate(sequencias):
        linha = linhas[(sujeito, acao)]
        a_seq = np.asarray(amp[linha])
        f_seq = np.asarray(fase[linha])
        blocos_a = np.stack([a_seq[i:f] for i, f in intervalos])
        blocos_f = np.stack([f_seq[i:f] for i, f in intervalos])
        m = len(intervalos)
        X[k:k + m] = extrair_atributos(blocos_a, blocos_f)
        y[k:k + m] = codigo[acao]
        grupos[k:k + m] = g
        k += m
    return X, y, grupos
