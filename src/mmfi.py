"""
Módulo de carregamento do dataset MM-Fi (NeurIPS 2023).

Substitui o Wi-Pose a partir de setembro/2026. O motivo da troca está em
docs/analise-mmfi.md: o Wi-Pose só distribuía amplitude e anotação 2D, o que
deixava as tarefas (b) e (d) sem objeto.

Estrutura do dataset
--------------------
    MMFi_Dataset/E01..E04/S01..S40/A01..A27/
        ground_truth.npy      pose 3D da sequência inteira: (297, 17, 3)
        wifi-csi/frameNNN.mat CSI de 100 ms: CSIamp e CSIphase, ambos (3, 114, 10)

⚠️ ``CSIamp`` está em **escala logarítmica (dB)**, não linear. A faixa observada é
de ~7 a ~51, e os ``-inf`` que aparecem em ~5% dos arquivos são o resultado de
``log10(0)``. Não aplique log de novo.

São 4 ambientes x 10 sujeitos x 27 ações = 1.080 sequências, 297 quadros cada.

Diferenças em relação ao Wi-Pose (importantes)
----------------------------------------------
- Os .mat são MATLAB v5: ``scipy.io.loadmat`` abre direto, sem h5py.
- ✅ **Existe fase** (``CSIphase``), crua e envolvida em ±π.
- A pose é **3D**, em **metros**, com 17 juntas no padrão **Human3.6M** (não COCO).
- ⚠️ O eixo **y cresce para baixo** — a cabeça tem y negativo.
- ⚠️ Cerca de 3% dos arquivos têm ``-inf`` em ``CSIamp``; tratamos aqui.
"""

import re
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.io as sio

# ---------------------------------------------------------------------------
# Constantes do dataset
# ---------------------------------------------------------------------------

N_ANTENAS = 3          # antenas receptoras (o transmissor tem 1)
N_SUBPORTADORAS = 114  # 40 MHz de banda em 5 GHz
N_TEMPO = 10           # amostras de CSI por quadro (100 ms a ~100 Hz)
N_JUNTAS = 17
N_QUADROS = 297        # todas as sequências têm exatamente este número

AMBIENTES = ("E01", "E02", "E03", "E04")

# As 27 ações. P1 = 14 atividades diárias, P2 = 13 exercícios de reabilitação.
ACOES: Dict[str, str] = {
    "A01": "alongar e relaxar",          "A02": "expansão do peito (horizontal)",
    "A03": "expansão do peito (vertical)", "A04": "torção (esquerda)",
    "A05": "torção (direita)",           "A06": "marcha estacionária",
    "A07": "extensão de membro (esq.)",  "A08": "extensão de membro (dir.)",
    "A09": "afundo (frente-esquerda)",   "A10": "afundo (frente-direita)",
    "A11": "extensão de membro (ambos)", "A12": "agachamento",
    "A13": "levantar mão esquerda",      "A14": "levantar mão direita",
    "A15": "afundo (esquerda)",          "A16": "afundo (direita)",
    "A17": "acenar mão esquerda",        "A18": "acenar mão direita",
    "A19": "pegar objeto",               "A20": "arremessar (p/ esquerda)",
    "A21": "arremessar (p/ direita)",    "A22": "chutar (p/ esquerda)",
    "A23": "chutar (p/ direita)",        "A24": "extensão do corpo (esq.)",
    "A25": "extensão do corpo (dir.)",   "A26": "salto",
    "A27": "curvar-se",
}

# Protocolos oficiais do artigo.
PROTOCOLO_1 = ("A02", "A03", "A04", "A05", "A13", "A14", "A17",
               "A18", "A19", "A20", "A21", "A22", "A23", "A27")
PROTOCOLO_2 = ("A01", "A06", "A07", "A08", "A09", "A10", "A11",
               "A12", "A15", "A16", "A24", "A25", "A26")
PROTOCOLO_3 = tuple(sorted(ACOES))

# Split cross-subject oficial (config.yaml do toolbox): 2 sujeitos por ambiente.
SUJEITOS_TESTE_S2 = ("S05", "S10", "S15", "S20", "S25", "S30", "S35", "S40")

# ---------------------------------------------------------------------------
# Esqueleto Human3.6M de 17 juntas
# ---------------------------------------------------------------------------
# ⚠️ O artigo do MM-Fi diz "17 keypoints" e induz a pensar em COCO. NÃO é COCO:
# verificado anatomicamente (cabeça->pés = 1,45 m com esta ordem; 0,32 m se COCO).
NOMES_JUNTAS = [
    "quadril",   "quadrilD",  "joelhoD",   "péD",
    "quadrilE",  "joelhoE",   "péE",       "coluna",
    "tórax",     "pescoço",   "cabeça",
    "ombroE",    "cotoveloE", "punhoE",
    "ombroD",    "cotoveloD", "punhoD",
]

CONEXOES_ESQUELETO = [
    (0, 1), (1, 2), (2, 3),              # perna direita
    (0, 4), (4, 5), (5, 6),              # perna esquerda
    (0, 7), (7, 8), (8, 9), (9, 10),     # coluna até a cabeça
    (8, 11), (11, 12), (12, 13),         # braço esquerdo
    (8, 14), (14, 15), (15, 16),         # braço direito
]


# ---------------------------------------------------------------------------
# Localização do dataset
# ---------------------------------------------------------------------------

def caminho_dataset(raiz: Optional[Union[str, Path]] = None) -> Path:
    """Diretório raiz do MM-Fi (o que contém E01..E04).

    Ordem de busca: argumento explícito -> variável de ambiente ``MMFI_DATASET``
    -> ``~/Downloads/datas/MMFi_Dataset``. Assim cada integrante da equipe aponta
    para o caminho da própria máquina sem editar código.
    """
    import os

    if raiz is not None:
        return Path(raiz).expanduser()
    if (bruto := os.environ.get("MMFI_DATASET")):
        return Path(bruto).expanduser()
    return Path.home() / "Downloads" / "datas" / "MMFi_Dataset"


_PADRAO_SUJEITO = re.compile(r"^S\d{2}$")


def ambiente_do_sujeito(sujeito: str) -> str:
    """Mapeia S01..S40 para o ambiente E01..E04 (10 sujeitos por ambiente).

    Cada sujeito aparece em exatamente um ambiente — por isso este mapeamento é
    determinístico, e não uma busca no disco.

    Valida o formato: aceitar ``"S1"`` ou ``"s10"`` levaria a um erro lá na frente
    culpando o caminho do dataset, que é o lugar errado para procurar o problema.
    """
    if not _PADRAO_SUJEITO.match(sujeito):
        raise ValueError(
            f"Sujeito com formato inválido: {sujeito!r}. Esperado 'S01' a 'S40'."
        )
    numero = int(sujeito[1:])
    if not 1 <= numero <= 40:
        raise ValueError(f"Sujeito fora do intervalo S01..S40: {sujeito!r}")
    return AMBIENTES[(numero - 1) // 10]


def caminho_sequencia(
    sujeito: str, acao: str, raiz: Optional[Union[str, Path]] = None
) -> Path:
    """Diretório de uma sequência, ex.: .../E01/S01/A01."""
    if acao not in ACOES:
        raise ValueError(f"Ação desconhecida: {acao!r}. Esperado A01..A27.")
    return caminho_dataset(raiz) / ambiente_do_sujeito(sujeito) / sujeito / acao


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def _tratar_nao_finitos(amplitude: np.ndarray) -> Tuple[np.ndarray, int]:
    """Substitui valores não-finitos pela média da respectiva fatia temporal.

    Cerca de 3% dos arquivos trazem ``-inf`` em ``CSIamp``. Sem tratamento, eles
    contaminam qualquer média, desvio ou treino como ``NaN``, silenciosamente.
    Esta é a mesma estratégia do carregador oficial do MM-Fi.

    Devolve ``(amplitude_corrigida, quantos_valores_foram_substituidos)``.
    """
    ruins = ~np.isfinite(amplitude)
    n_ruins = int(ruins.sum())
    if n_ruins == 0:
        return amplitude, 0

    amplitude = amplitude.copy()
    for t in range(amplitude.shape[2]):
        fatia = amplitude[:, :, t]
        mascara = ~np.isfinite(fatia)
        if mascara.any():
            bons = fatia[~mascara]
            # Se a fatia inteira for inválida, zera — melhor que propagar NaN.
            fatia[mascara] = bons.mean() if bons.size else 0.0
    return amplitude, n_ruins


def carregar_quadro(
    sujeito: str,
    acao: str,
    quadro: int,
    raiz: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Carrega um único quadro: CSI e a pose 3D correspondente.

    Parâmetros
    ----------
    sujeito : "S01" a "S40"
    acao    : "A01" a "A27"
    quadro  : 1 a 297 (numeração do dataset, começa em 1)

    Retorna
    -------
    amplitude : ndarray (3, 114, 10) float32
        Amplitude do CSI **em dB**, já com os ``-inf`` tratados.
        Eixos: (antena RX, subportadora, amostra temporal).
    fase : ndarray (3, 114, 10) float32
        Fase **crua**, em radianos, envolvida em ±π. Ver ``sanitizar_fase``.
    juntas : ndarray (17, 3) float32
        Pose 3D em **metros**, padrão Human3.6M. ⚠️ O eixo y cresce para BAIXO.
    meta : dict
        Metadados: sujeito, ação, nome da ação, ambiente, quadro, protocolo e
        quantos valores não-finitos foram corrigidos.
    """
    if not 1 <= quadro <= N_QUADROS:
        raise ValueError(f"Quadro fora do intervalo 1..{N_QUADROS}: {quadro}")

    pasta = caminho_sequencia(sujeito, acao, raiz)
    arquivo_csi = pasta / "wifi-csi" / f"frame{quadro:03d}.mat"
    arquivo_gt = pasta / "ground_truth.npy"

    if not arquivo_csi.exists():
        raise FileNotFoundError(
            f"CSI não encontrado: {arquivo_csi}\n"
            "Confira o caminho do dataset (variável de ambiente MMFI_DATASET)."
        )

    if not arquivo_gt.exists():
        raise FileNotFoundError(
            f"Ground truth não encontrado: {arquivo_gt}\n"
            "A sequência existe mas está incompleta — confira a extração."
        )

    try:
        dados = sio.loadmat(arquivo_csi)
        amplitude_bruta = np.asarray(dados["CSIamp"])
        fase = np.asarray(dados["CSIphase"])
    except Exception as erro:  # arquivo truncado, corrompido ou sem as variáveis
        raise ValueError(f"Falha ao ler {arquivo_csi.name}: {erro}") from erro

    forma_csi = (N_ANTENAS, N_SUBPORTADORAS, N_TEMPO)
    if amplitude_bruta.shape != forma_csi or fase.shape != forma_csi:
        raise ValueError(
            f"{arquivo_csi.name}: esperado CSI {forma_csi}, obtido "
            f"amp={amplitude_bruta.shape} fase={fase.shape}."
        )

    amplitude, n_corrigidos = _tratar_nao_finitos(amplitude_bruta)

    # O ground truth vem com a sequência inteira num arquivo só; o índice é
    # 0-based, enquanto os quadros do dataset são 1-based.
    poses = np.load(arquivo_gt)
    if poses.shape != (N_QUADROS, N_JUNTAS, 3):
        raise ValueError(
            f"{arquivo_gt.name}: esperado {(N_QUADROS, N_JUNTAS, 3)}, "
            f"obtido {poses.shape}."
        )
    juntas = poses[quadro - 1]

    meta = {
        "sujeito": sujeito,
        "acao": acao,
        "nome_acao": ACOES[acao],
        "ambiente": ambiente_do_sujeito(sujeito),
        "quadro": quadro,
        "protocolo": "P1 (diária)" if acao in PROTOCOLO_1 else "P2 (reabilitação)",
        "valores_corrigidos": n_corrigidos,
    }

    return (
        amplitude.astype(np.float32),
        fase.astype(np.float32),
        juntas.astype(np.float32),
        meta,
    )


def carregar_pose_sequencia(
    sujeito: str, acao: str, raiz: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """Pose 3D da sequência inteira: ``(297, 17, 3)``.

    Barato — é um único ``.npy`` de ~60 KB. Útil para animar o movimento sem
    precisar abrir os 297 arquivos de CSI.
    """
    return np.load(caminho_sequencia(sujeito, acao, raiz) / "ground_truth.npy")


# ---------------------------------------------------------------------------
# Pré-processamento de fase (tarefa d)
# ---------------------------------------------------------------------------

# Fronteira entre os dois blocos de 20 MHz que formam o canal de 40 MHz.
# Cada bloco tem referência de fase própria, e há uma descontinuidade aqui.
FRONTEIRA_20MHZ = 57


def sanitizar_fase(fase: np.ndarray, por_metade: bool = True) -> np.ndarray:
    """Remove o offset linear da fase ao longo das subportadoras.

    A fase medida vem contaminada por atraso de detecção de pacote, CFO e SFO,
    que juntos produzem uma **rampa linear** ao longo das subportadoras, além de
    estar envolvida em ±π. O procedimento clássico:

    1. ``unwrap`` ao longo do eixo das subportadoras;
    2. ajuste de reta por mínimos quadrados;
    3. subtração da reta ajustada.

    Parâmetros
    ----------
    por_metade : bool
        Se ``True`` (padrão), ajusta **uma reta para cada bloco de 20 MHz**, em
        vez de uma única reta sobre as 114 subportadoras. Ver a nota abaixo.

    ⚠️ **A descontinuidade no meio do espectro**

    O canal de 40 MHz é formado por **dois blocos de 20 MHz**, e cada um tem sua
    própria referência de fase. Medimos um salto médio de **1,667 rad na
    subportadora 56** — **32x o salto mediano** entre subportadoras vizinhas.

    Ajustar uma única reta sobre as 114 ignora essa quebra e deixa resíduo. Medido
    em 6 sequências de sujeitos e ambientes diferentes, ajustar por metade
    **reduz o desvio padrão do resíduo em 49% a 65%**.

    ⚠️ **Isto ainda é o ponto de partida da tarefa (d), não a resposta final.**
    Mesmo por metade, a rampa linear nem sempre domina. Vale comparar com a
    multiplicação conjugada entre antenas, que não supõe linearidade nenhuma.

    Recebe e devolve ``(3, 114, 10)``.
    """
    if fase.ndim != 3:
        raise ValueError(f"Esperado tensor (antenas, subportadoras, tempo), "
                         f"obtido ndim={fase.ndim}.")

    n_subportadoras = fase.shape[1]
    if por_metade and n_subportadoras == N_SUBPORTADORAS:
        blocos = [(0, FRONTEIRA_20MHZ), (FRONTEIRA_20MHZ, n_subportadoras)]
    else:
        if por_metade:
            warnings.warn(
                f"por_metade=True ignorado: esperava {N_SUBPORTADORAS} "
                f"subportadoras, recebeu {n_subportadoras}. Ajustando uma reta só.",
                stacklevel=2,
            )
        blocos = [(0, n_subportadoras)]

    saneada = np.empty_like(fase, dtype=np.float64)
    for antena in range(fase.shape[0]):
        for t in range(fase.shape[2]):
            for inicio, fim in blocos:
                # O unwrap é feito dentro de cada bloco: atravessar a fronteira
                # faria o unwrap "corrigir" uma descontinuidade que é real.
                traco = np.unwrap(fase[antena, inicio:fim, t])
                k = np.arange(fim - inicio, dtype=np.float64)
                coeficientes = np.polyfit(k, traco, 1)
                saneada[antena, inicio:fim, t] = traco - np.polyval(coeficientes, k)
    return saneada.astype(np.float32)
