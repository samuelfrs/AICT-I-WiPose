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

Conjuntos de atributos (ML-03)
------------------------------
``montar_conjunto`` aceita ``atributos``, uma lista de conjuntos de
:data:`CONJUNTOS_ATRIBUTOS` concatenados por janela na ordem pedida: ``base``
(os 1.368 acima) e ``antenas``, ``dop_amp`` e ``dop_cm`` (diferenças entre
antenas e espectros de Doppler, ver src/doppler.py), lidos de um cache próprio.
A normalização vale só para ``base``. O padrão, ``("base",)``, reproduz o
ML-01/02 bit a bit.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.mmfi import N_ANTENAS, N_QUADROS, N_SUBPORTADORAS

N_ATRIBUTOS = 4 * N_ANTENAS * N_SUBPORTADORAS  # 1.368

MODOS_NORMALIZACAO = ("nenhuma", "media", "zscore")

CONJUNTOS_ATRIBUTOS = ("base", "antenas", "dop_amp", "dop_cm")
ATRIBUTOS_PADRAO = ("base",)


def validar_atributos(atributos: Union[str, Sequence[str]]) -> Tuple[str, ...]:
    """Normaliza a lista de conjuntos (tupla ou ``"base,dop_amp"``) e valida.

    Mantém a ordem pedida. Lista vazia, conjunto desconhecido ou repetido:
    ``ValueError`` listando os conjuntos válidos.
    """
    if isinstance(atributos, str):
        atributos = [parte.strip() for parte in atributos.split(",")]
    atributos = tuple(atributos)
    validos = ", ".join(CONJUNTOS_ATRIBUTOS)
    desconhecidos = [a for a in atributos if a not in CONJUNTOS_ATRIBUTOS]
    if not atributos or desconhecidos:
        raise ValueError(
            f"conjunto de atributos inválido: {desconhecidos or 'lista vazia'}. "
            f"Conjuntos válidos: {validos}."
        )
    if len(set(atributos)) != len(atributos):
        raise ValueError(
            f"conjunto de atributos repetido em {list(atributos)}. "
            f"Conjuntos válidos: {validos}."
        )
    return atributos


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
    atributos: Union[str, Sequence[str]] = ATRIBUTOS_PADRAO,
    cache_atributos: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monta ``(X, y, grupos)`` a partir do cache, uma sequência por vez.

    - ``X``: ``(n_janelas_total, d)``, com os conjuntos de ``atributos``
      concatenados na ordem pedida (``d = 1368`` no padrão, só ``base``)
    - ``y``: índice da ação em ``acoes``
    - ``grupos``: índice da sequência em ``sequencias`` (para agregar e para o
      bootstrap — janelas da mesma sequência não são independentes)

    Lê do memmap só a sequência da vez (~4 MB por array), nunca o cache inteiro.
    ``normalizacao`` (ver :func:`normalizar_sequencia`) é aplicada à sequência
    inteira antes do janelamento, igual para treino e teste, e só ao ``base``.

    Os conjuntos além de ``base`` vêm de ``cache_atributos`` (o
    ``CacheAtributos`` de src/doppler.py), que precisa ter sido construído
    com a mesma janela e o mesmo passo.
    """
    _validar_modo(normalizacao)
    atributos = validar_atributos(atributos)
    if passo is None:
        passo = janela
    intervalos = janelas_de_sequencia(amp.shape[1], janela, passo)
    m = len(intervalos)

    dimensoes = []
    for nome in atributos:
        if nome == "base":
            dimensoes.append(N_ATRIBUTOS)
            continue
        if cache_atributos is None:
            raise ValueError(
                f"o conjunto {nome!r} vem do cache de atributos (src/doppler.py); "
                "passe cache_atributos."
            )
        if (cache_atributos.janela, cache_atributos.passo) != (janela, passo):
            raise ValueError(
                f"cache de atributos com janela/passo {cache_atributos.janela}/"
                f"{cache_atributos.passo}, pedido {janela}/{passo}."
            )
        dados = cache_atributos.conjuntos[nome]
        if dados.shape[1] != m:
            raise ValueError(
                f"cache de atributos com {dados.shape[1]} janelas por sequência, "
                f"esperado {m}."
            )
        dimensoes.append(dados.shape[2])
    colunas = np.concatenate([[0], np.cumsum(dimensoes)])

    codigo = {a: i for i, a in enumerate(acoes)}
    n = len(sequencias) * m
    X = np.empty((n, int(colunas[-1])), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    grupos = np.empty(n, dtype=np.int64)

    k = 0
    for g, (sujeito, acao) in enumerate(sequencias):
        for nome, c0, c1 in zip(atributos, colunas[:-1], colunas[1:]):
            if nome == "base":
                linha = linhas[(sujeito, acao)]
                a_seq = normalizar_sequencia(np.asarray(amp[linha]), normalizacao)
                f_seq = normalizar_sequencia(np.asarray(fase[linha]), normalizacao)
                blocos_a = np.stack([a_seq[i:f] for i, f in intervalos])
                blocos_f = np.stack([f_seq[i:f] for i, f in intervalos])
                X[k:k + m, c0:c1] = extrair_atributos(blocos_a, blocos_f)
            else:
                linha = cache_atributos.linhas[(sujeito, acao)]
                X[k:k + m, c0:c1] = cache_atributos.conjuntos[nome][linha]
        y[k:k + m] = codigo[acao]
        grupos[k:k + m] = g
        k += m
    return X, y, grupos
