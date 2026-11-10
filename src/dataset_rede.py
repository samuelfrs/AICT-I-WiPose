"""Dataset PyTorch para Classificação de Ações em Janelas Temporais de CSI (MM-Fi).

Implementação vetorizada com cache leve de 50 MB:
- Extrai características espectro-temporais (média e desvio padrão temporal de amplitude e fase sanitizada)
  em janelas deslizantes de 30 quadros (3 segundos = 300 pacotes OFDM).
- Suporta splits S1 (intra-sujeito agrupado por sequência) e S2 (Cross-Subject oficial).
- Suporta protocolos P1 (14 ações cotidianas), P2 (13 exercícios) e P3 (27 ações completas).
- Evita vazamento de dados: o scaler é ajustado apenas no conjunto de treino.
"""

from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

from src.mmfi import (
    ACOES,
    N_QUADROS,
    PROTOCOLO_1,
    PROTOCOLO_2,
    PROTOCOLO_3,
    caminho_dataset,
    caminho_pacote,
    sanitizar_fase_lote,
)
from src.protocolos import dividir_s1_agrupado, dividir_s2

CACHE_PADRAO = Path(__file__).resolve().parent.parent / "artifacts" / "cache_janelas_30.npz"


def construir_cache_janelas(
    caminho_cache: Path = CACHE_PADRAO,
    raiz: Optional[Union[str, Path]] = None,
    janela: int = 30,
    passo: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extrai características das janelas de todas as 1.080 sequências e salva em cache."""
    caminho_cache = Path(caminho_cache)
    caminho_cache.parent.mkdir(parents=True, exist_ok=True)

    acoes_ordenadas = sorted(list(ACOES.keys()))
    acao_to_idx = {a: i for i, a in enumerate(acoes_ordenadas)}
    todos_sujeitos = [f"S{i:02d}" for i in range(1, 41)]

    intervalos = [(i, i + janela) for i in range(0, N_QUADROS - janela + 1, passo)]

    todas_janelas_X = []
    todos_y = []
    todos_seq_indices = []
    seq_info = []

    seq_count = 0
    t0 = time.time()
    print(f"[CACHE] Construindo cache de janelas temporais em {caminho_cache}...")

    for s in todos_sujeitos:
        for a in acoes_ordenadas:
            pkg_path = caminho_pacote(s, a, raiz=raiz)
            if not pkg_path.is_file():
                continue

            with np.load(pkg_path) as pkg:
                amp = pkg["csi_amp"]   # (297, 3, 114, 10)
                fase = pkg["csi_fase"] # (297, 3, 114, 10)

                ruins = ~np.isfinite(amp)
                if ruins.any():
                    amp[ruins] = 0.0

                fase_san = sanitizar_fase_lote(fase)
                y_val = acao_to_idx[a]

                for ini, fim in intervalos:
                    sub_a = amp[ini:fim]
                    sub_f = fase_san[ini:fim]

                    m_a = sub_a.mean(axis=(0, 3)).reshape(-1)
                    s_a = sub_a.std(axis=(0, 3)).reshape(-1)
                    m_f = sub_f.mean(axis=(0, 3)).reshape(-1)
                    s_f = sub_f.std(axis=(0, 3)).reshape(-1)

                    feat = np.concatenate([m_a, s_a, m_f, s_f]).astype(np.float32)
                    todas_janelas_X.append(feat)
                    todos_y.append(y_val)
                    todos_seq_indices.append(seq_count)

                seq_info.append(f"{s}_{a}")
                seq_count += 1

    X = np.stack(todas_janelas_X)
    y = np.array(todos_y, dtype=np.int64)
    seq_indices = np.array(todos_seq_indices, dtype=np.int64)
    seq_info = np.array(seq_info)

    np.savez_compressed(
        caminho_cache,
        X=X,
        y=y,
        seq_indices=seq_indices,
        seq_info=seq_info,
    )
    print(f"[CACHE] Cache concluído em {time.time() - t0:.1f}s. Amostras: {len(X)}")
    return X, y, seq_indices, seq_info


class MMFiTemporalDataset(Dataset if torch else object):
    """Dataset PyTorch com características temporais de CSI para treino/teste."""

    def __init__(
        self,
        split: str = "S2",
        fase: str = "treino",  # "treino" ou "teste"
        protocolo: str = "P3",  # "P1", "P2" ou "P3"
        caminho_cache: Optional[Union[str, Path]] = None,
        refazer_cache: bool = False,
        seed: int = 0,
        media_scaler: Optional[np.ndarray] = None,
        desvio_scaler: Optional[np.ndarray] = None,
    ):
        if torch is None:
            raise RuntimeError("PyTorch não está instalado no ambiente.")
        super().__init__()
        self.split = split
        self.fase = fase
        self.protocolo = protocolo
        self.caminho_cache = Path(caminho_cache) if caminho_cache else CACHE_PADRAO

        if not self.caminho_cache.is_file() or refazer_cache:
            X_all, y_all, seq_indices_all, seq_info_all = construir_cache_janelas(
                caminho_cache=self.caminho_cache
            )
        else:
            dados = np.load(self.caminho_cache)
            X_all = dados["X"]
            y_all = dados["y"]
            seq_indices_all = dados["seq_indices"]
            seq_info_all = dados["seq_info"]

        # Mapeamento e filtragem de ações
        todas_acoes = sorted(list(ACOES.keys()))
        if protocolo == "P1":
            acoes_ativas = set(PROTOCOLO_1)
        elif protocolo == "P2":
            acoes_ativas = set(PROTOCOLO_2)
        else:
            acoes_ativas = set(PROTOCOLO_3)

        self.acoes_lista = sorted(list(acoes_ativas))
        self.num_classes = len(self.acoes_lista)
        acao_para_novo_idx = {a: i for i, a in enumerate(self.acoes_lista)}

        # Filtra sequências das ações do protocolo
        mask_proto = np.array(
            [seq_info_all[sid].split("_")[1] in acoes_ativas for sid in seq_indices_all]
        )
        X_proto = X_all[mask_proto]
        y_proto_raw = y_all[mask_proto]
        seq_idx_proto = seq_indices_all[mask_proto]

        y_proto = np.array([acao_para_novo_idx[todas_acoes[yi]] for yi in y_proto_raw])

        # Divisão Disjunta por Sequência conforme Protocolo
        if split == "S1":
            treino_seqs, teste_seqs = dividir_s1_agrupado(
                acoes=self.acoes_lista, seed=seed, proporcao_teste=0.25
            )
        else:
            treino_seqs, teste_seqs = dividir_s2(acoes=self.acoes_lista)

        seqs_alvo = set(f"{s}_{a}" for s, a in (treino_seqs if fase == "treino" else teste_seqs))
        mask_split = np.array([seq_info_all[sid] in seqs_alvo for sid in seq_idx_proto])

        X_sel = X_proto[mask_split]
        y_sel = y_proto[mask_split]
        seq_idx_sel = seq_idx_proto[mask_split]

        # Normalização com parâmetros do treino
        if fase == "treino":
            self.media = X_sel.mean(axis=0, keepdims=True)
            self.desvio = X_sel.std(axis=0, keepdims=True)
            self.desvio[self.desvio < 1e-6] = 1.0
        else:
            if media_scaler is None or desvio_scaler is None:
                raise ValueError("Para a fase de teste, forneça media_scaler e desvio_scaler do treino.")
            self.media = media_scaler
            self.desvio = desvio_scaler

        X_norm = (X_sel - self.media) / self.desvio

        self.X = torch.from_numpy(X_norm).float()
        self.y = torch.from_numpy(y_sel).long()
        self.seq_indices = seq_idx_sel
        self.seq_info = [seq_info_all[sid] for sid in seq_idx_sel]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        return {
            "csi": self.X[idx],
            "rotulo_acao": self.y[idx],
            "seq_idx": self.seq_indices[idx],
        }


# Alias para compatibilidade com código legado
MMFiMultiTaskDataset = MMFiTemporalDataset
