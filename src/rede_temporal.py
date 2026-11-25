"""
Rede temporal própria para a classificação de ações (tarefa c, ML-04).

Entrada
-------
Uma **janela de 30 quadros** (3 s) de uma sequência. Cada quadro vira uma
matriz ``(60, 114)``: os canais são {amplitude, fase sanitizada} × 3 antenas ×
10 amostras temporais do quadro, e o eixo restante são as 114 subportadoras.
Antes do janelamento, cada sequência tem subtraída a **sua própria** média por
canal × subportadora (apaga o perfil estático que cada sujeito deixa no canal,
como no ML-02). A média usa a sequência **inteira** (297 quadros), então
mesmo a métrica "por janela" carrega um pouco de contexto do resto da
sequência — não é uma janela de 3 s isolada. Depois, cada canal é padronizado com média e desvio do
**treino** (:func:`estatisticas_canal`, só sobre as sequências de treino).
Os dados ficam em float16 na RAM (~4,4 GB para as 1.080 sequências).

Modelo (:class:`RedeTemporal`, ~286 mil parâmetros)
--------------------------------------------------
- codificador por quadro: ``Dropout(0.1)`` → 3 × [``Conv1d(k=5)`` →
  ``BatchNorm1d`` → ``GELU`` → ``MaxPool1d(2)``] com 60→64→64→128 canais sobre
  as subportadoras → média sobre as subportadoras;
- GRU bidirecional (128 por direção) sobre os 30 quadros, média temporal;
- ``Dropout(0.4)`` → ``Linear(256, 27)``.

Treino (:func:`treinar`)
------------------------
3 janelas aleatórias por sequência de treino por época, lote 32, ``AdamW``
(lr 1e-3, weight decay 1e-2) com ``OneCycleLR`` (máx. 2e-3), entropia cruzada
com label smoothing 0,1, aumento por escala ``1 + 0,1·N(0,1)`` por canal e
ruído ``0,05·N(0,1)``. A cada época o modelo é medido na **validação**
(sujeitos ``SUJEITOS_VALIDACAO_S2``, fora do treino); a época escolhida é a de
maior acurácia por sequência na validação (empate: maior por janela; depois a
mais cedo), e é o estado dessa época que vai para o teste — avaliado uma vez.

Avaliação (:func:`prever`): as 9 janelas sem sobreposição de cada sequência; as
probabilidades são promediadas por sequência em ``src/avaliacao.avaliar``.
"""

import copy
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.atributos import janelas_de_sequencia
from src.avaliacao import agregar_por_sequencia
from src.mmfi import N_ANTENAS, N_SUBPORTADORAS, N_TEMPO
from src.protocolos import (
    SUJEITOS_VALIDACAO_S2,
    conferir_validacao,
    dividir_s2,
    separar_validacao,
)

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - o resto do projeto roda sem torch
    torch = None
    nn = None

JANELA = 30
N_CANAIS = 2 * N_ANTENAS * N_TEMPO  # 60

HIPERPARAMETROS = {
    "epocas": 15,
    "janela": JANELA,
    "janelas_por_sequencia": 3,
    "lote": 32,
    "otimizador": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "agendador": "OneCycleLR",
    "lr_max": 2e-3,
    "label_smoothing": 0.1,
    "aumento_escala": 0.1,
    "aumento_ruido": 0.05,
    "dropout_entrada": 0.1,
    "dropout_saida": 0.4,
}

Sequencia = Tuple[str, str]


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def dividir_rede(
    acoes: Sequence[str],
    sujeitos_validacao: Sequence[str] = SUJEITOS_VALIDACAO_S2,
) -> Tuple[List[Sequencia], List[Sequencia], List[Sequencia]]:
    """``(treino, validacao, teste)`` do S2: a validação sai do treino.

    Validação vazia, com sujeito do teste ou fora do treino: ``ValueError``
    (``conferir_validacao``). Os três lados são disjuntos por sujeito.
    """
    treino_s2, teste = dividir_s2(acoes)
    conferir_validacao(treino_s2, teste, sujeitos_validacao)
    treino, validacao = separar_validacao(treino_s2, sujeitos_validacao)
    if not treino or not validacao:
        raise ValueError("Treino ou validação vazios depois da separação.")
    sujeitos = [{s for s, _ in lado} for lado in (treino, validacao, teste)]
    if (sujeitos[0] & sujeitos[1]) or (sujeitos[0] & sujeitos[2]) or (sujeitos[1] & sujeitos[2]):
        raise ValueError("Vazamento: sujeito em mais de um lado (treino/validação/teste).")
    return treino, validacao, teste


def canais_da_sequencia(amp_seq: np.ndarray, fase_seq: np.ndarray) -> np.ndarray:
    """``(quadros, 3, 114, 10)`` × 2 → ``(quadros, 60, 114)`` em float32.

    Canal ``c = sinal·30 + antena·10 + amostra`` (sinal 0 = amplitude, 1 = fase).
    Subtrai a média **desta sequência** por canal × subportadora.
    """
    amp_seq = np.asarray(amp_seq)
    fase_seq = np.asarray(fase_seq)
    if amp_seq.shape != fase_seq.shape or amp_seq.ndim != 4 or \
            amp_seq.shape[1:] != (N_ANTENAS, N_SUBPORTADORAS, N_TEMPO):
        raise ValueError(
            f"Esperado (quadros, {N_ANTENAS}, {N_SUBPORTADORAS}, {N_TEMPO}) para amp e "
            f"fase, obtido {amp_seq.shape} e {fase_seq.shape}."
        )
    quadros = amp_seq.shape[0]
    partes = [np.asarray(s, dtype=np.float32).transpose(0, 1, 3, 2)
              .reshape(quadros, N_ANTENAS * N_TEMPO, N_SUBPORTADORAS)
              for s in (amp_seq, fase_seq)]
    x = np.concatenate(partes, axis=1)
    x -= x.mean(axis=0, dtype=np.float64, keepdims=True).astype(np.float32)
    return x


@dataclass
class DadosRede:
    """Sequências prontas para a rede: ``x`` ``(n, quadros, 60, 114)`` float16."""
    x: np.ndarray
    y: np.ndarray
    sequencias: List[Sequencia]


def carregar_dados(
    amp: np.ndarray,
    fase: np.ndarray,
    linhas: Dict[Sequencia, int],
    sequencias: Sequence[Sequencia],
    acoes: Sequence[str],
    log: Optional[Callable[[str], None]] = None,
) -> DadosRede:
    """Lê as ``sequencias`` do cache (src/cache_csi.py) uma a uma.

    ``y`` é o índice da ação em ``acoes``. Nada é calculado entre sequências:
    a padronização (que usa o treino) é aplicada depois, por lote.
    """
    sequencias = [tuple(s) for s in sequencias]
    codigo = {a: i for i, a in enumerate(acoes)}
    quadros = amp.shape[1]
    x = np.empty((len(sequencias), quadros, N_CANAIS, N_SUBPORTADORAS), dtype=np.float16)
    y = np.empty(len(sequencias), dtype=np.int64)
    inicio = time.time()
    for i, (sujeito, acao) in enumerate(sequencias):
        linha = linhas[(sujeito, acao)]
        x[i] = canais_da_sequencia(amp[linha], fase[linha])
        y[i] = codigo[acao]
        if log is not None and ((i + 1) % 108 == 0 or i + 1 == len(sequencias)):
            log(f"    {i + 1}/{len(sequencias)} sequências ({time.time() - inicio:.0f} s)")
    return DadosRede(x=x, y=y, sequencias=sequencias)


def estatisticas_canal(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Média e desvio por canal sobre sequências × quadros × subportadoras.

    Chamar **só com o treino**. Acumula em float64, uma sequência por vez.
    Desvio nulo vira 1 (canal constante fica em 0 depois de centrado).
    """
    soma = np.zeros(x.shape[2])
    soma_q = np.zeros(x.shape[2])
    for seq in x:
        s = seq.astype(np.float64)
        soma += s.sum(axis=(0, 2))
        soma_q += (s * s).sum(axis=(0, 2))
    n = x.shape[0] * x.shape[1] * x.shape[3]
    media = soma / n
    desvio = np.sqrt(np.maximum(soma_q / n - media ** 2, 0.0))
    desvio[desvio < 1e-6] = 1.0
    return media.astype(np.float32), desvio.astype(np.float32)


def montar_lote(dados: DadosRede, indices: np.ndarray, inicios: np.ndarray,
                janela: int, media: np.ndarray, desvio: np.ndarray) -> "torch.Tensor":
    """Janelas ``[inicio, inicio + janela)`` das sequências ``indices``, padronizadas.

    Devolve ``(B, janela, 60, 114)`` float32.
    """
    quadros = np.asarray(inicios)[:, None] + np.arange(janela)[None, :]
    x = dados.x[np.asarray(indices)[:, None], quadros].astype(np.float32)
    x -= media[None, None, :, None]
    x /= desvio[None, None, :, None]
    return torch.from_numpy(x)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

class RedeTemporal(nn.Module if nn is not None else object):
    """Codificador convolucional por quadro + GRU bidirecional sobre a janela."""

    def __init__(self, n_canais: int = N_CANAIS, n_classes: int = 27, oculto: int = 128,
                 dropout_entrada: float = 0.1, dropout_saida: float = 0.4):
        super().__init__()
        camadas: List[nn.Module] = [nn.Dropout(dropout_entrada)]
        for entrada, saida in zip((n_canais, 64, 64), (64, 64, 128)):
            camadas += [nn.Conv1d(entrada, saida, kernel_size=5, padding=2),
                        nn.BatchNorm1d(saida), nn.GELU(), nn.MaxPool1d(2)]
        camadas.append(nn.AdaptiveAvgPool1d(1))
        self.codificador = nn.Sequential(*camadas)
        self.gru = nn.GRU(128, oculto, batch_first=True, bidirectional=True)
        self.cabeca = nn.Sequential(nn.Dropout(dropout_saida), nn.Linear(2 * oculto, n_classes))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """``(B, T, canais, subportadoras)`` → logits ``(B, n_classes)``."""
        b, t, c, s = x.shape
        z = self.codificador(x.reshape(b * t, c, s)).reshape(b, t, -1)
        z, _ = self.gru(z)
        return self.cabeca(z.mean(dim=1))


def contar_parametros(modelo: "nn.Module") -> int:
    return int(sum(p.numel() for p in modelo.parameters()))


# ---------------------------------------------------------------------------
# Treino e avaliação
# ---------------------------------------------------------------------------

def prever(modelo: "nn.Module", dados: DadosRede, media: np.ndarray, desvio: np.ndarray,
           janela: int = JANELA, lote: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Probabilidades das janelas **sem sobreposição** de cada sequência.

    Devolve ``(proba, y, grupos)`` por janela, com ``grupos`` = índice da
    sequência em ``dados`` (entrada de ``src/avaliacao.avaliar``).
    """
    inicios = np.array([i for i, _ in janelas_de_sequencia(dados.x.shape[1], janela)])
    m = len(inicios)
    seqs = np.repeat(np.arange(len(dados.y)), m)
    pos = np.tile(inicios, len(dados.y))
    modelo.eval()
    saidas = []
    with torch.no_grad():
        for k in range(0, len(seqs), lote):
            x = montar_lote(dados, seqs[k:k + lote], pos[k:k + lote], janela, media, desvio)
            saidas.append(torch.softmax(modelo(x), dim=1).numpy())
    proba = np.concatenate(saidas).astype(np.float64)
    return proba, dados.y[seqs], seqs


def medir_validacao(proba: np.ndarray, y: np.ndarray, grupos: np.ndarray) -> Dict[str, float]:
    """Acurácia por janela, por sequência e log-loss por janela (para o histórico)."""
    y_seq, pred_seq, _ = agregar_por_sequencia(proba, y, grupos)
    p_certa = np.clip(proba[np.arange(len(y)), y], 1e-12, None)
    return {
        "acuracia_janela": float((proba.argmax(axis=1) == y).mean()),
        "acuracia_sequencia": float((pred_seq == y_seq).mean()),
        "perda": float(-np.log(p_certa).mean()),
    }


def melhor_epoca(historico: Sequence[Dict]) -> int:
    """Época (1-based) de maior acurácia por sequência na validação.

    Empate: maior acurácia por janela; depois, a época mais cedo.
    """
    melhor = max(range(len(historico)),
                 key=lambda i: (historico[i]["validacao"]["acuracia_sequencia"],
                                historico[i]["validacao"]["acuracia_janela"], -i))
    return melhor + 1


def treinar(
    treino: DadosRede,
    validacao: DadosRede,
    media: np.ndarray,
    desvio: np.ndarray,
    seed: int,
    n_classes: int,
    epocas: int = HIPERPARAMETROS["epocas"],
    janela: int = JANELA,
    lote: int = HIPERPARAMETROS["lote"],
    janelas_por_sequencia: int = HIPERPARAMETROS["janelas_por_sequencia"],
    log: Callable[[str], None] = print,
) -> Tuple["nn.Module", List[Dict], int]:
    """Treina com ``seed`` e devolve ``(modelo_da_epoca_escolhida, historico, epoca)``.

    Só ``treino`` entra no gradiente; ``validacao`` só escolhe a época.
    """
    if epocas < 1:
        raise ValueError(f"epocas deve ser ≥ 1, recebido {epocas}.")
    hp = HIPERPARAMETROS
    torch.manual_seed(seed)
    gerador_np = np.random.default_rng(seed)
    gerador_torch = torch.Generator().manual_seed(seed)

    modelo = RedeTemporal(n_classes=n_classes, dropout_entrada=hp["dropout_entrada"],
                          dropout_saida=hp["dropout_saida"])
    n_janelas = len(treino.y) * janelas_por_sequencia
    passos = (n_janelas + lote - 1) // lote
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=hp["lr"],
                                   weight_decay=hp["weight_decay"])
    agendador = torch.optim.lr_scheduler.OneCycleLR(otimizador, max_lr=hp["lr_max"],
                                                    total_steps=epocas * passos)
    criterio = nn.CrossEntropyLoss(label_smoothing=hp["label_smoothing"])
    ultimo_inicio = treino.x.shape[1] - janela

    historico: List[Dict] = []
    estados: List[Dict] = []          # estado ao fim de cada época (~1 MB cada)
    for epoca in range(1, epocas + 1):
        inicio = time.time()
        modelo.train()
        seqs = np.repeat(np.arange(len(treino.y)), janelas_por_sequencia)
        inicios = gerador_np.integers(0, ultimo_inicio + 1, size=len(seqs))
        ordem = gerador_np.permutation(len(seqs))
        seqs, inicios = seqs[ordem], inicios[ordem]
        perda_total = 0.0
        for k in range(0, len(seqs), lote):
            x = montar_lote(treino, seqs[k:k + lote], inicios[k:k + lote], janela, media, desvio)
            escala = 1.0 + hp["aumento_escala"] * torch.randn(
                (x.shape[0], 1, x.shape[2], 1), generator=gerador_torch)
            x = x * escala + hp["aumento_ruido"] * torch.randn(x.shape, generator=gerador_torch)
            y = torch.from_numpy(treino.y[seqs[k:k + lote]])
            otimizador.zero_grad()
            perda = criterio(modelo(x), y)
            perda.backward()
            otimizador.step()
            agendador.step()
            perda_total += float(perda.item()) * len(y)

        metricas = medir_validacao(*prever(modelo, validacao, media, desvio, janela))
        historico.append({"epoca": epoca, "perda_treino": perda_total / len(seqs),
                          "validacao": metricas, "tempo_s": round(time.time() - inicio, 1)})
        estados.append(copy.deepcopy(modelo.state_dict()))
        log(f"    época {epoca:2d}/{epocas}  perda treino {historico[-1]['perda_treino']:.3f}  "
            f"validação: janela {100 * metricas['acuracia_janela']:5.1f}%  "
            f"sequência {100 * metricas['acuracia_sequencia']:5.1f}%  "
            f"perda {metricas['perda']:.3f}  ({historico[-1]['tempo_s']:.0f} s)")

    # Um só lugar decide a época: a mesma que é reportada é a restaurada.
    escolhida = melhor_epoca(historico)
    modelo.load_state_dict(estados[escolhida - 1])
    modelo.eval()
    return modelo, historico, escolhida
