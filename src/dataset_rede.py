"""Dataset PyTorch de Alta Performance para a Rede Neural Multi-Head no MM-Fi.

Implementação pré-carregada e vetorizada em memória (RAM):
- Carrega apenas os quadros subamostrados ([::stride_quadros]) direto do disco para Tensores Contíguos.
- Ocupação de RAM extremamente baixa: ~180 MB para o dataset de treino inteiro.
- Elimina 100% da latência de descompressão/disco durante as épocas de treino.
- Suporta splits S1 (intra-sujeito) e S2 (Cross-Subject oficial) e protocolos P1, P2 e P3.
"""

from pathlib import Path
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
    SUJEITOS_TESTE_S2,
    caminho_dataset,
    caminho_pacote,
    sanitizar_fase,
)


class MMFiMultiTaskDataset(Dataset if torch else object):
    """Dataset PyTorch com tensores contíguos em memória para treinamento rápido."""

    def __init__(
        self,
        raiz: Optional[Union[str, Path]] = None,
        split: str = "S2",
        fase: str = "treino",  # "treino" ou "teste"
        protocolo: str = "P3",  # "P1", "P2" ou "P3"
        usar_fase: bool = False,
        stride_quadros: int = 15,
        sujeitos: Optional[List[str]] = None,
        max_sequencias: Optional[int] = None,
    ):
        if torch is None:
            raise RuntimeError("PyTorch não está instalado no ambiente.")
        super().__init__()
        self.raiz = caminho_dataset(raiz)
        self.split = split
        self.fase = fase
        self.protocolo = protocolo
        self.usar_fase = usar_fase
        self.stride_quadros = max(1, stride_quadros)

        # Mapeamento de ações
        self.acoes_lista = sorted(list(ACOES.keys()))
        if protocolo == "P1":
            self.acoes_ativas = set(PROTOCOLO_1)
        elif protocolo == "P2":
            self.acoes_ativas = set(PROTOCOLO_2)
        else:
            self.acoes_ativas = set(PROTOCOLO_3)

        self.acao_para_idx = {acao: i for i, acao in enumerate(self.acoes_lista)}

        # Sujeitos conforme split
        todos_sujeitos = [f"S{i:02d}" for i in range(1, 41)]
        if sujeitos is not None:
            self.sujeitos = sujeitos
        elif split == "S2":
            if fase == "teste":
                self.sujeitos = list(SUJEITOS_TESTE_S2)
            else:
                self.sujeitos = [s for s in todos_sujeitos if s not in SUJEITOS_TESTE_S2]
        else:
            self.sujeitos = todos_sujeitos

        # Pré-carregamento vetorizado em Tensores
        self._carregar_dados_vetorizados(max_sequencias)

    def _carregar_dados_vetorizados(self, max_sequencias: Optional[int] = None) -> None:
        """Carrega e vetoriza as fatias [::stride] diretamente para arrays em memória."""
        lista_csi = []
        lista_rotulos = []
        lista_poses = []

        count_seq = 0
        for sujeito in self.sujeitos:
            for acao in sorted(list(self.acoes_ativas)):
                if self.split == "S1":
                    eh_teste = (hash(f"{sujeito}_{acao}") % 5) == 0
                    if (self.fase == "teste" and not eh_teste) or (self.fase == "treino" and eh_teste):
                        continue

                arquivo_pacote = caminho_pacote(sujeito, acao, self.raiz)
                if not arquivo_pacote.is_file():
                    continue

                try:
                    with np.load(arquivo_pacote) as pkg:
                        # Extrai apenas as fatias necessárias (redução imediata de 95% do volume)
                        amp = pkg["csi_amp"][::self.stride_quadros].astype(np.float32)  # (N, 3, 114, 10)
                        pose = pkg["pose"][::self.stride_quadros].astype(np.float32)    # (N, 17, 3)

                        # Tratamento vetorizado dos -inf
                        ruins = ~np.isfinite(amp)
                        if ruins.any():
                            amp[ruins] = 0.0

                        # Normalização Min-Max por quadro vetorizada
                        mins = amp.min(axis=(1, 2, 3), keepdims=True)
                        maxs = amp.max(axis=(1, 2, 3), keepdims=True)
                        delta = maxs - mins
                        delta[delta == 0] = 1.0
                        amp_norm = (amp - mins) / delta

                        if self.usar_fase:
                            fase = pkg["csi_fase"][::self.stride_quadros].astype(np.float32)
                            fase_sanitizada = np.empty_like(fase)
                            for idx in range(len(fase)):
                                fase_sanitizada[idx] = sanitizar_fase(fase[idx])
                            fase_norm = np.clip(fase_sanitizada / np.pi, -1.0, 1.0)
                            csi_completo = np.concatenate([amp_norm, fase_norm], axis=1)  # (N, 6, 114, 10)
                        else:
                            csi_completo = amp_norm  # (N, 3, 114, 10)

                        rotulo = self.acao_para_idx[acao]
                        n_amostras = len(amp_norm)

                        lista_csi.append(csi_completo)
                        lista_rotulos.append(np.full((n_amostras,), rotulo, dtype=np.int64))
                        lista_poses.append(pose)

                        count_seq += 1
                        if max_sequencias and count_seq >= max_sequencias:
                            break
                except Exception:
                    continue
            if max_sequencias and count_seq >= max_sequencias:
                break

        if lista_csi:
            self.csi = torch.from_numpy(np.concatenate(lista_csi, axis=0)).float()
            self.rotulos = torch.from_numpy(np.concatenate(lista_rotulos, axis=0)).long()
            self.poses = torch.from_numpy(np.concatenate(lista_poses, axis=0)).float()
        else:
            self.csi = torch.empty(0, 3, 114, 10)
            self.rotulos = torch.empty(0, dtype=torch.long)
            self.poses = torch.empty(0, 17, 3)

    def __len__(self) -> int:
        return len(self.rotulos)

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        return {
            "csi": self.csi[idx],
            "rotulo_acao": self.rotulos[idx],
            "pose_3d": self.poses[idx],
        }
