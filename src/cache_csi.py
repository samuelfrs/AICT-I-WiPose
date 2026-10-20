"""
Cache em disco do CSI do MM-Fi: amplitude e fase sanitizada, lidas por memmap.

Ler os 1.080 arquivos do dataset e sanitizar a fase a cada experimento leva
minutos. O cache paga isso uma vez e grava, em ``destino``:

    amp.npy      (1080, 297, 3, 114, 10) float32 — amplitude em dB, -inf tratados
    fase.npy     (1080, 297, 3, 114, 10) float32 — fase sanitizada (sanitizar_fase_lote)
    indice.json  ordem das sequências: linha i <-> (sujeito, acao)
    concluido    marcador gravado POR ÚLTIMO; sem ele o cache é tratado como ausente

São ~4,4 GB por array. :func:`abrir_cache` devolve memmaps somente-leitura: o
sistema operacional traz para a RAM só as sequências tocadas, então o cache nunca
é carregado inteiro (a máquina de referência tem 15 GB).
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.mmfi import (
    N_ANTENAS,
    N_QUADROS,
    N_SUBPORTADORAS,
    N_TEMPO,
    caminho_dataset,
    carregar_csi_sequencia,
    formato_disponivel,
    sanitizar_fase_lote,
)
from src.protocolos import listar_sequencias

MARCADOR = "concluido"
VERSAO_CACHE = 1


def cache_completo(destino: Union[str, Path]) -> bool:
    """O cache só vale se o marcador de conclusão existir."""
    destino = Path(destino)
    return all((destino / nome).is_file()
               for nome in ("amp.npy", "fase.npy", "indice.json", MARCADOR))


def _conferir_dataset(raiz: Path, sequencias: List[Tuple[str, str]]) -> None:
    """Falha cedo, antes de criar 9 GB de arquivo, se faltar alguma sequência."""
    if not raiz.is_dir():
        raise FileNotFoundError(
            f"Dataset MM-Fi não encontrado em {raiz}.\n"
            "Aponte a variável de ambiente MMFI_DATASET para a raiz (a que contém E01..E04)."
        )
    faltando = [f"{s}/{a}" for s, a in sequencias
                if formato_disponivel(s, a, raiz) == "ausente"]
    if faltando:
        raise FileNotFoundError(
            f"{len(faltando)} de {len(sequencias)} sequências ausentes em {raiz} "
            f"(ex.: {', '.join(faltando[:3])}).\n"
            "Confira a variável de ambiente MMFI_DATASET."
        )


def construir_cache(
    destino: Union[str, Path],
    raiz: Optional[Union[str, Path]] = None,
    refazer: bool = False,
) -> Path:
    """Constrói o cache em ``destino`` (ou reaproveita um completo).

    - Cache completo e ``refazer=False``: não relê o dataset.
    - Cache incompleto (build interrompido, sem marcador): avisa e reconstrói.
    - Dataset ausente: ``FileNotFoundError`` citando ``MMFI_DATASET``.

    Devolve o diretório do cache.
    """
    destino = Path(destino)
    sequencias = listar_sequencias()
    if cache_completo(destino) and not refazer:
        # A raiz NÃO é comparada: mover a pasta do dataset não invalida o cache.
        motivo = None
        try:
            indice = json.loads((destino / "indice.json").read_text(encoding="utf-8"))
            if indice.get("versao") != VERSAO_CACHE:
                motivo = (f"versão {indice.get('versao')!r}, esperado {VERSAO_CACHE}")
            elif [tuple(s) for s in indice.get("sequencias", [])] != list(sequencias):
                motivo = "lista de sequências diferente da esperada"
        except (ValueError, OSError) as erro:
            motivo = f"indice.json ilegível ({erro})"
        if motivo is None:
            return destino
        print(f"AVISO: cache em {destino} desatualizado ({motivo}); "
              f"reconstruindo do zero.")
    elif destino.exists() and any(destino.iterdir()) and not refazer:
        print(f"AVISO: cache incompleto em {destino} (sem o marcador "
              f"'{MARCADOR}'); reconstruindo do zero.")

    raiz = caminho_dataset(raiz)
    _conferir_dataset(raiz, sequencias)

    # Apaga o marcador antes de tudo: se este build cair no meio, o cache
    # antigo não pode continuar parecendo válido.
    (destino / MARCADOR).unlink(missing_ok=True)
    destino.mkdir(parents=True, exist_ok=True)

    forma = (len(sequencias), N_QUADROS, N_ANTENAS, N_SUBPORTADORAS, N_TEMPO)
    amp = np.lib.format.open_memmap(destino / "amp.npy", mode="w+",
                                    dtype=np.float32, shape=forma)
    fase = np.lib.format.open_memmap(destino / "fase.npy", mode="w+",
                                     dtype=np.float32, shape=forma)

    inicio = time.time()
    for i, (sujeito, acao) in enumerate(sequencias):
        amplitude, fase_crua = carregar_csi_sequencia(sujeito, acao, raiz)
        amp[i] = amplitude
        fase[i] = sanitizar_fase_lote(fase_crua)
        if (i + 1) % 54 == 0 or i + 1 == len(sequencias):
            decorrido = time.time() - inicio
            print(f"    cache: {i + 1}/{len(sequencias)} sequências "
                  f"({decorrido:.0f} s)", flush=True)

    amp.flush()
    fase.flush()
    del amp, fase

    indice = {
        "versao": VERSAO_CACHE,
        "raiz": str(raiz),
        "forma": list(forma),
        "fase": "sanitizada (sanitizar_fase_lote, reta por bloco de 20 MHz)",
        "amplitude": "dB, valores não-finitos substituídos pela média da fatia",
        "sequencias": [list(seq) for seq in sequencias],
    }
    (destino / "indice.json").write_text(
        json.dumps(indice, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    # Por último: só agora o cache passa a contar como completo.
    (destino / MARCADOR).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    return destino


def abrir_cache(
    destino: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[str, str], int]]:
    """Abre o cache como memmaps somente-leitura.

    Devolve ``(amp, fase, linhas)``, em que ``linhas[(sujeito, acao)]`` é o
    índice da sequência no primeiro eixo dos dois arrays.
    """
    destino = Path(destino)
    if not cache_completo(destino):
        raise FileNotFoundError(
            f"Cache incompleto ou ausente em {destino}. Rode construir_cache antes."
        )
    indice = json.loads((destino / "indice.json").read_text(encoding="utf-8"))
    amp = np.load(destino / "amp.npy", mmap_mode="r")
    fase = np.load(destino / "fase.npy", mmap_mode="r")

    forma = tuple(indice["forma"])
    if amp.shape != forma or fase.shape != forma:
        raise ValueError(
            f"Cache em {destino} inconsistente: amp={amp.shape}, fase={fase.shape}, "
            f"índice diz {forma}. Reconstrua com refazer=True."
        )
    linhas = {(s, a): i for i, (s, a) in enumerate(indice["sequencias"])}
    return amp, fase, linhas

