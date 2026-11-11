"""
Protocolos de divisão treino/teste do MM-Fi, sem vazamento (tarefa c).

A unidade de divisão é sempre a **sequência** ``(sujeito, ação)``. Dividir por
quadro ou por janela colocaria quadros vizinhos — separados por 100 ms e quase
idênticos — dos dois lados do split, e a acurácia de teste mediria memória, não
generalização.

Protocolos
----------
- **S1 agrupado** — sorteia, para cada ação, uma fração das 40 sequências para o
  teste. Os sujeitos de teste também aparecem no treino (com outras ações), então
  o S1 mede o cenário "pessoa conhecida".
- **S2 (oficial)** — cross-subject do toolbox do MM-Fi: os 8 sujeitos de
  ``SUJEITOS_TESTE_S2`` (2 por ambiente) só aparecem no teste.
- **Validação do S2** (ML-03) — para escolher atributos, hiperparâmetros ou
  época sem olhar o teste: ``SUJEITOS_VALIDACAO_S2`` (1 por ambiente, nenhum
  do teste) saem do treino do S2 e viram a validação.

⚠️ Cada sujeito está em um único ambiente (docs/analise-mmfi.md §2), então o S2
também retém parte de cada ambiente — não é um eixo independente do S3.

Todas as funções devolvem listas ordenadas de ``(sujeito, acao)``, reusadas pelo
baseline e pela rede neural.
"""

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from src.mmfi import ACOES, PROTOCOLO_3, SUJEITOS_TESTE_S2

SUJEITOS = tuple(f"S{i:02d}" for i in range(1, 41))

# Um por ambiente (S01–S10 = E01, …), nenhum de SUJEITOS_TESTE_S2.
SUJEITOS_VALIDACAO_S2 = ("S04", "S14", "S24", "S34")

Sequencia = Tuple[str, str]
Divisao = Tuple[List[Sequencia], List[Sequencia]]


def _validar_acoes(acoes: Iterable[str]) -> Tuple[str, ...]:
    acoes = tuple(sorted(set(acoes)))
    if not acoes:
        raise ValueError("Lista de ações vazia.")
    desconhecidas = [a for a in acoes if a not in ACOES]
    if desconhecidas:
        raise ValueError(f"Ações desconhecidas: {desconhecidas}. Esperado A01..A27.")
    return acoes


def listar_sequencias(acoes: Sequence[str] = PROTOCOLO_3) -> List[Sequencia]:
    """Todas as sequências ``(sujeito, acao)`` das ações pedidas (40 por ação)."""
    acoes = _validar_acoes(acoes)
    return [(s, a) for s in SUJEITOS for a in acoes]


def dividir_s2(acoes: Sequence[str] = PROTOCOLO_3) -> Divisao:
    """Split cross-subject oficial: teste = ``SUJEITOS_TESTE_S2``.

    Devolve ``(treino, teste)``. Nenhum sujeito aparece dos dois lados.
    """
    todas = listar_sequencias(acoes)
    treino = [seq for seq in todas if seq[0] not in SUJEITOS_TESTE_S2]
    teste = [seq for seq in todas if seq[0] in SUJEITOS_TESTE_S2]
    return treino, teste


def separar_validacao(
    treino: Sequence[Sequencia],
    sujeitos_validacao: Sequence[str] = SUJEITOS_VALIDACAO_S2,
) -> Divisao:
    """Separa ``treino`` em ``(restante, validacao)`` pelos sujeitos de validação.

    Mantém a ordem de ``treino`` dos dois lados.
    """
    validos = set(sujeitos_validacao)
    validacao = [seq for seq in treino if seq[0] in validos]
    restante = [seq for seq in treino if seq[0] not in validos]
    return restante, validacao


def dividir_validacao_s2(acoes: Sequence[str] = PROTOCOLO_3) -> Divisao:
    """Treino do S2 dividido em ``(treino_sem_validacao, validacao)``.

    A validação são as sequências de ``SUJEITOS_VALIDACAO_S2``; o teste do S2
    não entra em nenhum dos dois lados.
    """
    treino, _ = dividir_s2(acoes)
    return separar_validacao(treino, SUJEITOS_VALIDACAO_S2)


def dividir_s1_agrupado(
    acoes: Sequence[str] = PROTOCOLO_3,
    seed: int = 0,
    proporcao_teste: float = 0.25,
) -> Divisao:
    """Split aleatório **por sequência**, estratificado por ação.

    Para cada ação, sorteia ``round(40 * proporcao_teste)`` sujeitos para o
    teste (10 com o padrão de 0,25). O gerador de cada ação é semeado com
    ``(seed, número da ação)``: assim o sorteio de A05 é o mesmo usando P1 ou P3,
    e a mesma seed reproduz exatamente o mesmo split.

    Devolve ``(treino, teste)``.
    """
    if not 0.0 < proporcao_teste < 1.0:
        raise ValueError(
            f"proporcao_teste deve estar em (0, 1), recebido {proporcao_teste}."
        )
    acoes = _validar_acoes(acoes)
    n_teste = int(round(len(SUJEITOS) * proporcao_teste))
    if not 1 <= n_teste < len(SUJEITOS):
        raise ValueError(
            f"proporcao_teste={proporcao_teste} deixa {n_teste} sequências de "
            f"teste por ação; é preciso ao menos 1 no teste e 1 no treino."
        )

    treino: List[Sequencia] = []
    teste: List[Sequencia] = []
    for acao in acoes:
        gerador = np.random.default_rng([int(seed), int(acao[1:])])
        sorteados = set(gerador.choice(len(SUJEITOS), size=n_teste, replace=False))
        for i, sujeito in enumerate(SUJEITOS):
            (teste if i in sorteados else treino).append((sujeito, acao))
    return sorted(treino), sorted(teste)
