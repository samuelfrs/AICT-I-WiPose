"""
Módulo de manipulação e carregamento dos dados do dataset Wi-Pose.
Compatível com arquivos MATLAB v7.3 (HDF5) e em conformidade com a estrutura real dos dados.
"""

from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union
import h5py
import numpy as np

# Mapeamento oficial das 12 ações presentes no dataset
ACOES = [
    "bend",
    "circle",
    "crouch",
    "jump",
    "pull",
    "push",
    "run",
    "sitdown",
    "standup",
    "throw",
    "walk",
    "wave",
]

NOME_PARA_ROTULO = {acao: idx for idx, acao in enumerate(ACOES)}
ROTULO_PARA_NOME = {idx: acao for idx, acao in enumerate(ACOES)}

# Ordem anatômica comum de 18 juntas do AlphaPose / COCO 18
NOMES_JUNTAS = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
    "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
    "LEye", "REar", "LEar"
]

CONEXOES_ESQUELETO = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (1, 5), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (0, 14), (14, 16), (0, 15), (15, 17)
]


def extrair_metadados_nome(arquivo_path: Union[str, Path]) -> Dict[str, Union[str, int]]:
    """
    Extrai ação, sujeito/ensaio e índice do frame a partir do nome do arquivo.
    Exemplo: 'bend_009-frame001.mat' -> acao='bend', ensaio='009', frame=1
    """
    nome = Path(arquivo_path).stem
    match = re.match(r"^([a-zA-Z]+)_(\d+)-frame(\d+)$", nome)
    if match:
        acao, ensaio, frame = match.groups()
        return {
            "acao": acao.lower(),
            "rotulo": NOME_PARA_ROTULO.get(acao.lower(), -1),
            "ensaio": ensaio,
            "frame": int(frame),
        }
    return {"acao": "desconhecida", "rotulo": -1, "ensaio": "000", "frame": -1}


def carregar_amostra_mat(
    arquivo_path: Union[str, Path]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Union[str, int]]]:
    """
    Carrega uma amostra .mat do Wi-Pose via HDF5.

    Retorna:
        csi: ndarray com forma (9, 30, 5) -> (9 canais Tx-Rx, 30 subportadoras, 5 pacotes temporais)
        juntas: ndarray com forma (18, 3) -> (18 juntas, [x, y, confianca])
        metadados: dict com as informações de rótulo, ação, ensaio e frame
    """
    path = Path(arquivo_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    metadados = extrair_metadados_nome(path)

    with h5py.File(path, "r") as f:
        # Raw CSI tem forma (3, 3, 30, 5) no h5py -> 3 Rx, 3 Tx, 30 subportadoras, 5 instantes
        raw_csi = np.array(f["CSI"], dtype=np.float32)
        # Redimensiona os 3x3 pares de antena em 9 canais: forma (9, 30, 5)
        csi = raw_csi.reshape(9, 30, 5)

        # Raw SkeletonPoints tem forma (1, 54) organizada como [18 y, 18 x, 18 confiança]
        raw_pts = np.array(f["SkeletonPoints"], dtype=np.float32).squeeze()
        y = raw_pts[0:18]
        x = raw_pts[18:36]
        c = raw_pts[36:54]
        juntas = np.stack([x, y, c], axis=-1)  # forma (18, 3)

    return csi, juntas, metadados
