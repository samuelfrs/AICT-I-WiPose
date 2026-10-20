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
de ~7 a ~51, e os ``-inf`` que aparecem em ~6% dos quadros são o resultado de
``log10(0)``. Não aplique log de novo.

São 4 ambientes x 10 sujeitos x 27 ações = 1.080 sequências, 297 quadros cada.

Diferenças em relação ao Wi-Pose (importantes)
----------------------------------------------
- Os .mat são MATLAB v5: ``scipy.io.loadmat`` abre direto, sem h5py.
- ✅ **Existe fase** (``CSIphase``), crua e envolvida em ±π.
- A pose é **3D**, em **metros**, com 17 juntas no padrão **Human3.6M** (não COCO).
- ⚠️ O eixo **y cresce para baixo** — a cabeça tem y negativo.
- ⚠️ Cerca de 6% dos quadros têm ``-inf`` em ``CSIamp``; tratamos aqui.
- O loader aceita **dois layouts**: a árvore original de ``.mat`` e a versão
  leve gerada por ``scripts/empacotar_mmfi.py`` (um ``.npz`` por sequência).
  Ver :func:`formato_disponivel` — a detecção é automática.
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


def caminho_pacote(
    sujeito: str, acao: str, raiz: Optional[Union[str, Path]] = None
) -> Path:
    """Arquivo da sequência no formato leve, ex.: .../E01/S01/A01.npz."""
    if acao not in ACOES:
        raise ValueError(f"Ação desconhecida: {acao!r}. Esperado A01..A27.")
    return caminho_dataset(raiz) / ambiente_do_sujeito(sujeito) / sujeito / f"{acao}.npz"


def formato_disponivel(
    sujeito: str, acao: str, raiz: Optional[Union[str, Path]] = None
) -> str:
    """Descobre em qual dos dois formatos a sequência está no disco.

    Devolve ``"pacote"`` (versão leve, um .npz por sequência), ``"original"``
    (árvore de .mat do MM-Fi) ou ``"ausente"``.

    O pacote tem prioridade: se os dois existirem, ler um arquivo só é bem mais
    rápido que abrir 297.

    ⚠️ A checagem exige ``is_file()``, e não só ``exists()``: um diretório
    chamado ``A01.npz`` faria toda leitura falhar em vez de cair para os ``.mat``.
    Um ``.npz`` ainda sendo gravado, ou truncado, também não deve sombrear uma
    árvore original íntegra ao lado.
    """
    pacote = caminho_pacote(sujeito, acao, raiz)
    tem_original = (caminho_sequencia(sujeito, acao, raiz) / "wifi-csi").is_dir()

    if pacote.is_file():
        if not tem_original:
            return "pacote"
        # Com os dois formatos no disco, só preferimos o pacote se ele abrir.
        try:
            with np.load(pacote) as dados:
                if {"csi_amp", "csi_fase", "pose"} <= set(dados.files):
                    return "pacote"
        except Exception:
            pass
        warnings.warn(
            f"{pacote.name} está ilegível ou incompleto; usando os .mat originais.",
            stacklevel=2,
        )
    return "original" if tem_original else "ausente"


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


def _validar_poses(poses: np.ndarray, nome: str) -> None:
    """Confere a forma da sequência inteira de pose.

    Sem isto, um ground truth truncado só apareceria como ``IndexError`` cru ao
    pedir um quadro alto — e nos scripts isso escapa como traceback em vez de
    mensagem útil.
    """
    esperado = (N_QUADROS, N_JUNTAS, 3)
    if poses.shape != esperado:
        raise ValueError(f"{nome}: pose com forma {poses.shape}, esperado {esperado}.")


def _ler_do_original(
    sujeito: str, acao: str, quadro: int, raiz: Optional[Union[str, Path]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lê um quadro da árvore original do MM-Fi (297 .mat por sequência)."""
    pasta = caminho_sequencia(sujeito, acao, raiz)
    arquivo_csi = pasta / "wifi-csi" / f"frame{quadro:03d}.mat"
    arquivo_gt = pasta / "ground_truth.npy"

    if not arquivo_csi.exists():
        raise FileNotFoundError(f"CSI não encontrado: {arquivo_csi}")
    if not arquivo_gt.exists():
        raise FileNotFoundError(
            f"Ground truth não encontrado: {arquivo_gt}\n"
            "A sequência existe mas está incompleta — confira a extração."
        )

    try:
        dados = sio.loadmat(arquivo_csi)
        amplitude = np.asarray(dados["CSIamp"])
        fase = np.asarray(dados["CSIphase"])
    except Exception as erro:  # arquivo truncado, corrompido ou sem as variáveis
        raise ValueError(f"Falha ao ler {arquivo_csi.name}: {erro}") from erro

    # O ground truth traz a sequência inteira; o índice é 0-based, enquanto os
    # quadros do dataset são 1-based.
    poses = np.load(arquivo_gt)
    _validar_poses(poses, arquivo_gt.name)
    return amplitude, fase, poses[quadro - 1]


def _ler_do_pacote(
    sujeito: str, acao: str, quadro: int, raiz: Optional[Union[str, Path]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lê um quadro da versão leve (um .npz por sequência).

    O ``.npz`` é um zip: indexar ``pacote["csi_amp"][quadro - 1]`` descomprime o
    array inteiro da sequência. Para varrer muitos quadros da mesma sequência,
    use :func:`carregar_csi_sequencia`, que paga isso uma vez só.
    """
    arquivo = caminho_pacote(sujeito, acao, raiz)
    try:
        with np.load(arquivo) as pacote:
            _validar_poses(pacote["pose"], arquivo.name)
            amplitude = pacote["csi_amp"][quadro - 1]
            fase = pacote["csi_fase"][quadro - 1]
            juntas = pacote["pose"][quadro - 1]
    except (ValueError, KeyError) as erro:
        raise ValueError(f"Falha ao ler {arquivo.name}: {erro}") from erro
    except Exception as erro:
        raise ValueError(f"Falha ao ler {arquivo.name}: {erro}") from erro
    return amplitude, fase, juntas


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

    formato = formato_disponivel(sujeito, acao, raiz)
    if formato == "ausente":
        raise FileNotFoundError(
            f"Sequência {sujeito}/{acao} não encontrada em {caminho_dataset(raiz)}\n"
            "Confira o caminho do dataset (variável de ambiente MMFI_DATASET)."
        )

    if formato == "pacote":
        amplitude_bruta, fase, juntas = _ler_do_pacote(sujeito, acao, quadro, raiz)
    else:
        amplitude_bruta, fase, juntas = _ler_do_original(sujeito, acao, quadro, raiz)

    forma_csi = (N_ANTENAS, N_SUBPORTADORAS, N_TEMPO)
    if amplitude_bruta.shape != forma_csi or fase.shape != forma_csi:
        raise ValueError(
            f"{sujeito}/{acao}/quadro {quadro}: esperado CSI {forma_csi}, obtido "
            f"amp={amplitude_bruta.shape} fase={fase.shape}."
        )
    if juntas.shape != (N_JUNTAS, 3):
        raise ValueError(
            f"{sujeito}/{acao}/quadro {quadro}: esperado pose {(N_JUNTAS, 3)}, "
            f"obtido {juntas.shape}."
        )

    amplitude, n_corrigidos = _tratar_nao_finitos(amplitude_bruta)

    meta = {
        "formato": formato,
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

    Barato nos dois formatos: no original é um ``.npy`` de ~60 KB; no pacote, o
    ``.npz`` descomprime só o array pedido.
    """
    if formato_disponivel(sujeito, acao, raiz) == "pacote":
        arquivo = caminho_pacote(sujeito, acao, raiz)
        try:
            with np.load(arquivo) as pacote:
                poses = pacote["pose"]
        except Exception as erro:
            raise ValueError(f"Falha ao ler {arquivo.name}: {erro}") from erro
    else:
        arquivo = caminho_sequencia(sujeito, acao, raiz) / "ground_truth.npy"
        poses = np.load(arquivo)

    _validar_poses(poses, arquivo.name)
    return poses


def carregar_csi_sequencia(
    sujeito: str,
    acao: str,
    raiz: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """CSI da sequência inteira: ``(297, 3, 114, 10)`` para amplitude e fase.

    Use isto em vez de chamar :func:`carregar_quadro` num laço: no formato leve,
    cada chamada descomprimiria o array inteiro de novo; no original, abriria os
    297 arquivos um a um.

    A amplitude já vem com os valores não-finitos tratados, quadro a quadro.
    """
    if formato_disponivel(sujeito, acao, raiz) == "pacote":
        arquivo = caminho_pacote(sujeito, acao, raiz)
        try:
            with np.load(arquivo) as pacote:
                amplitude = pacote["csi_amp"].astype(np.float32)
                fase = pacote["csi_fase"].astype(np.float32)
        except Exception as erro:
            raise ValueError(f"Falha ao ler {arquivo.name}: {erro}") from erro

        esperado = (N_QUADROS, N_ANTENAS, N_SUBPORTADORAS, N_TEMPO)
        if amplitude.shape != esperado or fase.shape != esperado:
            raise ValueError(
                f"{arquivo.name}: CSI com forma amp={amplitude.shape} "
                f"fase={fase.shape}, esperado {esperado}."
            )
    else:
        pasta = caminho_sequencia(sujeito, acao, raiz) / "wifi-csi"
        arquivos = sorted(pasta.glob("frame*.mat"))
        if len(arquivos) != N_QUADROS:
            # Um buraco na numeração desalinharia tudo contra pose[quadro-1],
            # porque a posição na lista deixa de ser o número do quadro.
            raise ValueError(
                f"{pasta}: esperado {N_QUADROS} arquivos, encontrei {len(arquivos)}."
            )
        forma_quadro = (N_ANTENAS, N_SUBPORTADORAS, N_TEMPO)
        amplitude = np.empty((N_QUADROS,) + forma_quadro, dtype=np.float32)
        fase = np.empty_like(amplitude)
        for i, arquivo in enumerate(arquivos):
            dados = sio.loadmat(arquivo)
            # A atribuição faria broadcast em silêncio se a forma divergisse —
            # um (3,114,1) degenerado seria replicado pelos 10 instantes.
            if dados["CSIamp"].shape != forma_quadro:
                raise ValueError(
                    f"{arquivo.name}: CSI com forma {dados['CSIamp'].shape}, "
                    f"esperado {forma_quadro}."
                )
            amplitude[i] = dados["CSIamp"]
            fase[i] = dados["CSIphase"]

    for i in range(len(amplitude)):
        amplitude[i], _ = _tratar_nao_finitos(amplitude[i])
    return amplitude, fase


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


def sanitizar_fase_lote(fase: np.ndarray, por_metade: bool = True) -> np.ndarray:
    """Versão vetorizada de :func:`sanitizar_fase` para ``(..., 3, 114, 10)``.

    Mesmo algoritmo — ``unwrap`` ao longo das subportadoras e subtração de uma
    reta de mínimos quadrados por bloco de 20 MHz —, mas sem laço em Python. O
    laço de :func:`sanitizar_fase` custa 60 ajustes por quadro (3 antenas ×
    2 blocos × 10 amostras); nos 320 mil
    quadros do dataset isso inviabiliza o cache.

    A reta sai da forma fechada dos mínimos quadrados, aplicada de uma vez a
    todos os traços do bloco (``k = 0..n-1``)::

        b = Σ(k - k̄)(y - ȳ) / Σ(k - k̄)²     a = ȳ - b·k̄

    Aceita qualquer número de eixos à esquerda, ex.: ``(297, 3, 114, 10)`` para
    uma sequência inteira. O eixo das subportadoras é sempre o penúltimo.
    """
    if fase.ndim < 3:
        raise ValueError(f"Esperado tensor (..., antenas, subportadoras, tempo), "
                         f"obtido ndim={fase.ndim}.")

    n_subportadoras = fase.shape[-2]
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

    fase = np.asarray(fase, dtype=np.float64)
    saneada = np.empty_like(fase)
    for inicio, fim in blocos:
        # Unwrap dentro do bloco, como em sanitizar_fase: a fronteira é real.
        traco = np.unwrap(fase[..., inicio:fim, :], axis=-2)
        k = np.arange(fim - inicio, dtype=np.float64)
        k_centrado = (k - k.mean())[:, None]          # (n, 1): broadcast no tempo
        media_y = traco.mean(axis=-2, keepdims=True)
        inclinacao = ((k_centrado * (traco - media_y)).sum(axis=-2, keepdims=True)
                      / (k_centrado ** 2).sum())
        intercepto = media_y - inclinacao * k.mean()
        saneada[..., inicio:fim, :] = traco - (intercepto + inclinacao * k[:, None])
    return saneada.astype(np.float32)
