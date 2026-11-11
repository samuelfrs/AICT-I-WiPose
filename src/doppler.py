"""
Atributos de Doppler e entre antenas, por janela (ML-03), e o cache deles em disco.

O ``base`` do ML-01 (média e desvio por antena × subportadora, ver
src/atributos.py) olha cada antena isolada e não vê **frequência** de
movimento. Os três conjuntos daqui completam isso:

- ``antenas`` (1.710): para cada par de antenas (0,1), (0,2), (1,2) e cada
  subportadora, média e desvio da **diferença de amplitude em dB** e a média
  circular da **diferença de fase crua**: média do cosseno, média do seno e
  comprimento resultante. (O ângulo médio seria descontínuo em ±π — ângulos
  quase iguais pareceriam opostos para os cortes das árvores.) A
  diferença entre antenas do mesmo receptor cancela o desvio de fase do
  oscilador, que é comum às três; por isso a fase crua basta aqui.
- ``dop_amp`` (3 × 51 = 153): espectro de Welch (``nperseg=100`` → 1 Hz por
  bin, 0 a 50 Hz) da **amplitude linear** sem a média da janela, promediado
  sobre as subportadoras, normalizado para somar 1 e em ``log(p + 1e-6)``. Um
  por antena.
- ``dop_cm`` (3 × 100 = 300): espectro de Welch **bilateral** (−50 a 49 Hz, na
  ordem da FFT: 0, 1, …, 49, −50, …, −1) da **multiplicação conjugada**
  ``H_p · conj(H_q)`` sem a média da janela, promediado sobre as subportadoras,
  normalizado e em log. Um por par de antenas. O sinal da frequência diz o
  sentido em que a fase relativa gira (aproximar × afastar).

Tudo sai da **série contínua** da sequência: os 297 quadros × 10 amostras viram
2.970 amostras a 100 Hz, e cada janela de :func:`janelas_de_sequencia`
(``[inicio, fim)`` em quadros) é o recorte ``[10·inicio, 10·fim)``. A
amplitude vem em dB com os −inf já tratados por ``carregar_csi_sequencia``; a
fase é a **crua** (o cache do ML-01 só guarda a sanitizada, então este cache
relê o dataset).

**Limitação:** o eixo de frequência supõe amostragem **uniforme a 100 Hz**
(10 amostras por quadro de 100 ms). O MM-Fi não traz o instante de cada
amostra; se o intervalo real variar, os bins de Doppler ficam borrados ou
deslocados.

Cada sequência leva ~0,35 s (leitura + cálculo) — calcular a cada execução
seria inviável. :func:`construir_cache_atributos` paga isso uma vez (~6 min
medidos para as 1.080 sequências) e grava, em ``destino``::

    antenas.npy  (1080, n_janelas, 1710) float32
    dop_amp.npy  (1080, n_janelas, 153)  float32
    dop_cm.npy   (1080, n_janelas, 300)  float32
    indice.json  versão, janela, passo e ordem das sequências
    concluido    marcador gravado POR ÚLTIMO

São ~80 MB com a janela padrão. Mudar a janela ou o passo reconstrói o cache.
"""

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.signal import welch

from src.atributos import janelas_de_sequencia
from src.mmfi import (
    N_ANTENAS,
    N_QUADROS,
    N_SUBPORTADORAS,
    N_TEMPO,
    caminho_dataset,
    carregar_csi_sequencia,
)
from src.protocolos import listar_sequencias

PARES = ((0, 1), (0, 2), (1, 2))
FS = 100.0      # amostras de CSI por segundo (10 por quadro de 100 ms)
NPERSEG = 100   # segmento de Welch: 1 s, resolução de 1 Hz

D_ANTENAS = 5 * len(PARES) * N_SUBPORTADORAS   # 1.710
D_DOP_AMP = N_ANTENAS * (NPERSEG // 2 + 1)     # 153
D_DOP_CM = len(PARES) * NPERSEG                # 300
DIMENSOES = {"antenas": D_ANTENAS, "dop_amp": D_DOP_AMP, "dop_cm": D_DOP_CM}
CONJUNTOS_DOPPLER = tuple(DIMENSOES)

# Janela mínima: o segmento de Welch tem de caber inteiro, senão o número de
# bins (e a dimensão dos atributos) mudaria com a janela.
JANELA_MINIMA = NPERSEG // N_TEMPO             # 10 quadros

MARCADOR = "concluido"
VERSAO_CACHE_ATRIBUTOS = 2   # 2: ``antenas`` com cos/sen médios em vez do ângulo


def frequencias_dop_amp() -> np.ndarray:
    """Frequência (Hz) de cada bin de ``dop_amp``, dentro de uma antena (51)."""
    return np.fft.rfftfreq(NPERSEG, d=1.0 / FS)


def frequencias_dop_cm() -> np.ndarray:
    """Frequência (Hz) de cada bin de ``dop_cm``, dentro de um par (100, ordem da FFT)."""
    return np.fft.fftfreq(NPERSEG, d=1.0 / FS)


def _validar_intervalos(intervalos: Sequence[Tuple[int, int]]) -> None:
    if not intervalos:
        raise ValueError("Nenhuma janela para extrair atributos.")
    curtas = [(i, f) for i, f in intervalos if f - i < JANELA_MINIMA]
    if curtas:
        raise ValueError(
            f"Os atributos de Doppler exigem janelas de ao menos {JANELA_MINIMA} "
            f"quadros ({NPERSEG} amostras, o segmento de Welch); "
            f"recebido {curtas[0][1] - curtas[0][0]}."
        )


def _validar_sequencia(x: np.ndarray, nome: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 4 or x.shape[1:] != (N_ANTENAS, N_SUBPORTADORAS, N_TEMPO):
        raise ValueError(
            f"{nome}: esperado (quadros, {N_ANTENAS}, {N_SUBPORTADORAS}, {N_TEMPO}), "
            f"obtido {x.shape}."
        )
    return x


def _recortes(x: np.ndarray, intervalos: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Série contínua ``(3, 114, quadros·10)`` recortada nas janelas.

    Devolve ``(n_janelas, 3, 114, amostras_por_janela)`` em float64.
    """
    serie = x.transpose(1, 2, 0, 3).reshape(N_ANTENAS, N_SUBPORTADORAS, -1)
    serie = serie.astype(np.float64)
    return np.stack([serie[:, :, i * N_TEMPO:f * N_TEMPO] for i, f in intervalos])


def _espectro_normalizado(potencia: np.ndarray) -> np.ndarray:
    """Normaliza para somar 1 no último eixo e passa para log (finito mesmo se zero)."""
    potencia = potencia / (potencia.sum(axis=-1, keepdims=True) + 1e-12)
    return np.log(potencia + 1e-6)


def atributos_antenas(amp: np.ndarray, fase: np.ndarray,
                      intervalos: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Diferenças entre pares de antenas, por subportadora: ``(n_janelas, 1710)``.

    ``amp`` em dB e ``fase`` **crua**, ambos ``(quadros, 3, 114, 10)``. Por
    par, na ordem de :data:`PARES`: ``[média(ΔdB), desvio(ΔdB),
    média cos(Δfase), média sen(Δfase), comprimento resultante(Δfase)]``,
    114 valores cada.
    """
    _validar_intervalos(intervalos)
    A = _recortes(_validar_sequencia(amp, "amp"), intervalos)
    P = _recortes(_validar_sequencia(fase, "fase"), intervalos)
    partes = []
    for p, q in PARES:
        d = A[:, p] - A[:, q]
        ph = P[:, p] - P[:, q]
        c, s = np.cos(ph).mean(axis=-1), np.sin(ph).mean(axis=-1)
        partes += [d.mean(axis=-1), d.std(axis=-1), c, s, np.hypot(c, s)]
    return np.concatenate(partes, axis=1).astype(np.float32)


def atributos_dop_amp(amp: np.ndarray,
                      intervalos: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Espectro da amplitude linear, por antena: ``(n_janelas, 153)``.

    Bins em :func:`frequencias_dop_amp`; antena 0 nas colunas 0–50, etc.
    """
    _validar_intervalos(intervalos)
    linear = 10.0 ** (_recortes(_validar_sequencia(amp, "amp"), intervalos) / 20.0)
    linear = linear - linear.mean(axis=-1, keepdims=True)
    _, potencia = welch(linear, fs=FS, nperseg=NPERSEG, axis=-1)  # (n, 3, 114, 51)
    potencia = _espectro_normalizado(potencia.mean(axis=2))       # (n, 3, 51)
    return potencia.reshape(len(intervalos), -1).astype(np.float32)


def atributos_dop_cm(amp: np.ndarray, fase: np.ndarray,
                     intervalos: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Espectro bilateral da multiplicação conjugada, por par: ``(n_janelas, 300)``.

    Bins em :func:`frequencias_dop_cm`; par (0,1) nas colunas 0–99, etc. Uma
    fase de ``p`` menos ``q`` girando a +f Hz dá pico no bin de +f Hz.
    """
    _validar_intervalos(intervalos)
    linear = 10.0 ** (_recortes(_validar_sequencia(amp, "amp"), intervalos) / 20.0)
    H = linear * np.exp(1j * _recortes(_validar_sequencia(fase, "fase"), intervalos))
    partes = []
    for p, q in PARES:
        z = H[:, p] * np.conj(H[:, q])
        z = z - z.mean(axis=-1, keepdims=True)
        _, potencia = welch(z, fs=FS, nperseg=NPERSEG, axis=-1,
                            return_onesided=False)                # (n, 114, 100)
        partes.append(_espectro_normalizado(potencia.mean(axis=1)))
    return np.concatenate(partes, axis=1).astype(np.float32)


def atributos_sequencia(amp: np.ndarray, fase: np.ndarray, janela: int = 30,
                        passo: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Os três conjuntos de uma sequência, nas janelas de :func:`janelas_de_sequencia`."""
    intervalos = janelas_de_sequencia(len(amp), janela, passo)
    return {
        "antenas": atributos_antenas(amp, fase, intervalos),
        "dop_amp": atributos_dop_amp(amp, intervalos),
        "dop_cm": atributos_dop_cm(amp, fase, intervalos),
    }


# ---------------------------------------------------------------------------
# Cache em disco
# ---------------------------------------------------------------------------

class CacheAtributos(NamedTuple):
    """Cache aberto: ``conjuntos[nome]`` é ``(n_sequencias, n_janelas, d)``."""
    conjuntos: Dict[str, np.ndarray]
    linhas: Dict[Tuple[str, str], int]
    janela: int
    passo: int


def cache_atributos_completo(destino: Union[str, Path]) -> bool:
    """O cache só vale se o marcador de conclusão existir."""
    destino = Path(destino)
    nomes = [f"{c}.npy" for c in CONJUNTOS_DOPPLER] + ["indice.json", MARCADOR]
    return all((destino / nome).is_file() for nome in nomes)


def _motivo_para_refazer(destino: Path, janela: int, passo: int,
                         sequencias: List[Tuple[str, str]]) -> Optional[str]:
    try:
        indice = json.loads((destino / "indice.json").read_text(encoding="utf-8"))
    except (ValueError, OSError) as erro:
        return f"indice.json ilegível ({erro})"
    if indice.get("versao") != VERSAO_CACHE_ATRIBUTOS:
        return f"versão {indice.get('versao')!r}, esperado {VERSAO_CACHE_ATRIBUTOS}"
    if (indice.get("janela"), indice.get("passo")) != (janela, passo):
        return (f"janela/passo {indice.get('janela')}/{indice.get('passo')}, "
                f"pedido {janela}/{passo}")
    if [tuple(s) for s in indice.get("sequencias", [])] != list(sequencias):
        return "lista de sequências diferente da esperada"
    return None


def _conferir_dataset(raiz: Path, sequencias: List[Tuple[str, str]]) -> None:
    from src.cache_csi import _conferir_dataset as conferir
    conferir(raiz, sequencias)


def construir_cache_atributos(
    destino: Union[str, Path],
    raiz: Optional[Union[str, Path]] = None,
    janela: int = 30,
    passo: Optional[int] = None,
    refazer: bool = False,
    sequencias: Optional[Sequence[Tuple[str, str]]] = None,
    carregar: Optional[Callable[..., Tuple[np.ndarray, np.ndarray]]] = None,
) -> Path:
    """Constrói o cache dos atributos de Doppler em ``destino`` (ou reaproveita).

    - Completo, mesma versão, janela, passo e sequências: reaproveitado.
    - Versão, janela, passo ou sequências diferentes, ou build interrompido
      (sem o marcador): avisa e reconstrói do zero.

    ``sequencias`` (padrão: as 1.080) e ``carregar`` (padrão:
    ``carregar_csi_sequencia``) existem para os testes. Devolve ``destino``.
    """
    destino = Path(destino)
    passo = janela if passo is None else passo
    intervalos = janelas_de_sequencia(N_QUADROS, janela, passo)
    _validar_intervalos(intervalos)
    sequencias = list(listar_sequencias() if sequencias is None else map(tuple, sequencias))
    carregar = carregar_csi_sequencia if carregar is None else carregar

    if cache_atributos_completo(destino) and not refazer:
        motivo = _motivo_para_refazer(destino, janela, passo, sequencias)
        if motivo is None:
            return destino
        print(f"AVISO: cache de atributos em {destino} desatualizado ({motivo}); "
              f"reconstruindo do zero.")
    elif destino.exists() and any(destino.iterdir()) and not refazer:
        print(f"AVISO: cache de atributos incompleto em {destino} (sem o marcador "
              f"'{MARCADOR}'); reconstruindo do zero.")

    raiz = caminho_dataset(raiz)
    _conferir_dataset(raiz, sequencias)

    # Apaga o marcador antes de tudo: um build que cair no meio não pode deixar
    # o cache antigo parecendo válido.
    (destino / MARCADOR).unlink(missing_ok=True)
    destino.mkdir(parents=True, exist_ok=True)

    arrays = {
        nome: np.lib.format.open_memmap(
            destino / f"{nome}.npy", mode="w+", dtype=np.float32,
            shape=(len(sequencias), len(intervalos), d))
        for nome, d in DIMENSOES.items()
    }
    inicio = time.time()
    for i, (sujeito, acao) in enumerate(sequencias):
        amp, fase = carregar(sujeito, acao, raiz)
        for nome, valores in atributos_sequencia(amp, fase, janela, passo).items():
            arrays[nome][i] = valores
        if (i + 1) % 54 == 0 or i + 1 == len(sequencias):
            print(f"    cache de atributos: {i + 1}/{len(sequencias)} sequências "
                  f"({time.time() - inicio:.0f} s)", flush=True)
    for valores in arrays.values():
        valores.flush()
    del arrays

    indice = {
        "versao": VERSAO_CACHE_ATRIBUTOS,
        "raiz": str(raiz),
        "janela": janela,
        "passo": passo,
        "n_janelas": len(intervalos),
        "dimensoes": DIMENSOES,
        "fs": FS,
        "nperseg": NPERSEG,
        "pares": [list(p) for p in PARES],
        "amplitude": "dB, -inf tratados (carregar_csi_sequencia)",
        "fase": "crua",
        "sequencias": [list(seq) for seq in sequencias],
    }
    (destino / "indice.json").write_text(
        json.dumps(indice, ensure_ascii=False, indent=1), encoding="utf-8")
    # Por último: só agora o cache passa a contar como completo.
    (destino / MARCADOR).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    return destino


def abrir_cache_atributos(destino: Union[str, Path]) -> CacheAtributos:
    """Abre o cache como memmaps somente-leitura (ver :class:`CacheAtributos`)."""
    destino = Path(destino)
    if not cache_atributos_completo(destino):
        raise FileNotFoundError(
            f"Cache de atributos incompleto ou ausente em {destino}. "
            "Rode construir_cache_atributos antes."
        )
    indice = json.loads((destino / "indice.json").read_text(encoding="utf-8"))
    n = len(indice["sequencias"])
    conjuntos = {}
    for nome, d in DIMENSOES.items():
        valores = np.load(destino / f"{nome}.npy", mmap_mode="r")
        if valores.shape != (n, indice["n_janelas"], d):
            raise ValueError(
                f"Cache de atributos em {destino} inconsistente: {nome}={valores.shape}, "
                f"esperado {(n, indice['n_janelas'], d)}. Reconstrua com refazer=True."
            )
        conjuntos[nome] = valores
    linhas = {(s, a): i for i, (s, a) in enumerate(indice["sequencias"])}
    return CacheAtributos(conjuntos, linhas, indice["janela"], indice["passo"])
