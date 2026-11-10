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

Normalização por sequência (ML-02, experimento)
-----------------------------------------------
Opcionalmente, cada sequência inteira (297 quadros) é normalizada **antes** do
janelamento, separadamente por antena × subportadora e separadamente para
amplitude e fase, com estatísticas calculadas sobre quadros × amostras daquela
sequência só (nunca sobre mais de uma). ``media`` subtrai a média; ``zscore``
subtrai a média e divide pelo desvio. A ideia é apagar a "assinatura" estática
que cada sujeito deixa no canal. O padrão continua ``nenhuma``.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.mmfi import N_ANTENAS, N_QUADROS, N_SUBPORTADORAS

N_ATRIBUTOS = 4 * N_ANTENAS * N_SUBPORTADORAS  # 1.368

MODOS_NORMALIZACAO = ("nenhuma", "media", "zscore")


def _validar_modo(modo: str) -> None:
    if modo not in MODOS_NORMALIZACAO:
        raise ValueError(
            f"normalização inválida: {modo!r}. "
            f"Modos válidos: {', '.join(MODOS_NORMALIZACAO)}."
        )


def normalizar_sequencia(x: np.ndarray, modo: str = "nenhuma") -> np.ndarray:
    """Normaliza **uma** sequência ``(quadros, 3, 114, 10)`` por antena × subportadora.

    Estatísticas sobre quadros (eixo 0) × amostras temporais (eixo 3) desta
    sequência:

    - ``nenhuma``: devolve a entrada sem alteração (o mesmo objeto);
    - ``media``: ``x - média``;
    - ``zscore``: ``(x - média) / desvio``. Onde o desvio é (numericamente)
      zero — sinal constante —, o resultado é 0, sem NaN nem inf.

    Chamar uma vez para a amplitude e outra para a fase. Só ``media`` e
    ``zscore`` conferem a forma e devolvem float32; ``nenhuma`` não confere
    nada e devolve a entrada como veio.
    """
    _validar_modo(modo)
    if modo == "nenhuma":
        return x
    x = np.asarray(x)
    if x.ndim != 4 or x.shape[1:3] != (N_ANTENAS, N_SUBPORTADORAS):
        raise ValueError(
            f"Esperado (quadros, {N_ANTENAS}, {N_SUBPORTADORAS}, tempo), "
            f"obtido {x.shape}."
        )
    x64 = x.astype(np.float64)
    media = x64.mean(axis=(0, 3), keepdims=True)
    centrado = x64 - media
    if modo == "media":
        return centrado.astype(np.float32)
    desvio = centrado.std(axis=(0, 3), keepdims=True)
    # Desvio "zero" relativo à escala do sinal: o arredondamento de um sinal
    # constante deixa resíduos da ordem de 1e-16 × |média|.
    constante = desvio <= 1e-9 * (np.abs(media) + 1.0)
    saida = np.divide(centrado, desvio, out=np.zeros_like(centrado), where=~constante)
    return saida.astype(np.float32)


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
    normalizacao: str = "nenhuma",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monta ``(X, y, grupos)`` a partir do cache, uma sequência por vez.

    - ``X``: ``(n_janelas_total, 1368)``
    - ``y``: índice da ação em ``acoes``
    - ``grupos``: índice da sequência em ``sequencias`` (para agregar e para o
      bootstrap — janelas da mesma sequência não são independentes)

    Lê do memmap só a sequência da vez (~4 MB por array), nunca o cache inteiro.
    ``normalizacao`` (ver :func:`normalizar_sequencia`) é aplicada à sequência
    inteira antes do janelamento, igual para treino e teste.
    """
    _validar_modo(normalizacao)
    intervalos = janelas_de_sequencia(amp.shape[1], janela, passo)
    codigo = {a: i for i, a in enumerate(acoes)}
    n = len(sequencias) * len(intervalos)
    X = np.empty((n, N_ATRIBUTOS), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    grupos = np.empty(n, dtype=np.int64)

    k = 0
    for g, (sujeito, acao) in enumerate(sequencias):
        linha = linhas[(sujeito, acao)]
        a_seq = normalizar_sequencia(np.asarray(amp[linha]), normalizacao)
        f_seq = normalizar_sequencia(np.asarray(fase[linha]), normalizacao)
        blocos_a = np.stack([a_seq[i:f] for i, f in intervalos])
        blocos_f = np.stack([f_seq[i:f] for i, f in intervalos])
        m = len(intervalos)
        X[k:k + m] = extrair_atributos(blocos_a, blocos_f)
        y[k:k + m] = codigo[acao]
        grupos[k:k + m] = g
        k += m
    return X, y, grupos
