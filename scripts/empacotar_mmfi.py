"""
Gera uma versão leve do MM-Fi, só com as modalidades que o projeto usa.

O dataset completo tem ~100 GB, quase todos em modalidades que não vamos tocar
(só a profundidade são ~73 GB). Este script mantém WiFi CSI, mmWave e a pose 3D,
e reempacota cada sequência num único arquivo comprimido.

Redução medida: ~100 GB → ~6,3 GB (float32) ou ~3,4 GB (float16).

Vem de três coisas:
  1. Descartar depth, lidar, rgb, infra1 e infra2.
  2. Juntar os 297 arquivos de cada sequência num só — some o custo por arquivo
     e a compressão passa a ver a sequência inteira de uma vez.
  3. Comprimir, e opcionalmente reduzir a precisão.

Estrutura gerada:

    MMFi_leve/
    ├── LEIAME.md
    └── E01/S01/A01.npz     ← uma sequência inteira num arquivo

``src/mmfi.py`` lê os dois formatos, então quem tiver o dataset original e quem
tiver a versão leve rodam exatamente o mesmo código.

Uso:
    # teste rápido: só o sujeito S01
    .venv/Scripts/python.exe scripts/empacotar_mmfi.py --destino D:/MMFi_leve --sujeitos S01

    # tudo (demora; dá para interromper e retomar)
    .venv/Scripts/python.exe scripts/empacotar_mmfi.py --destino D:/MMFi_leve
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mmfi import (  # noqa: E402
    ACOES,
    AMBIENTES,
    N_QUADROS,
    ambiente_do_sujeito,
    caminho_dataset,
)

PRECISOES = {"float16": np.float16, "float32": np.float32, "float64": np.float64}


def _ler_wifi(pasta: Path) -> tuple:
    """Lê os 297 .mat de uma sequência e empilha amplitude e fase."""
    arquivos = sorted(pasta.glob("frame*.mat"))
    if len(arquivos) != N_QUADROS:
        raise ValueError(f"{pasta}: esperado {N_QUADROS} arquivos, achei {len(arquivos)}")

    amplitude = np.empty((N_QUADROS, 3, 114, 10), dtype=np.float64)
    fase = np.empty_like(amplitude)
    for i, arquivo in enumerate(arquivos):
        dados = sio.loadmat(arquivo)
        amplitude[i] = dados["CSIamp"]
        fase[i] = dados["CSIphase"]
    return amplitude, fase


def _ler_mmwave(pasta: Path) -> tuple:
    """Lê as nuvens de pontos do mmWave.

    Cada quadro tem um número diferente de pontos (de 8 a ~34), então não dá para
    empilhar num array retangular. Concatenamos tudo e guardamos os índices de
    corte — o quadro ``i`` vai de ``cortes[i]`` a ``cortes[i+1]``.
    """
    arquivos = sorted(pasta.glob("frame*.bin"))
    nuvens = [
        np.frombuffer(a.read_bytes(), dtype=np.float64).reshape(-1, 5)
        for a in arquivos
    ]
    cortes = np.cumsum([0] + [len(n) for n in nuvens]).astype(np.int32)
    return np.concatenate(nuvens) if nuvens else np.empty((0, 5)), cortes


def empacotar_sequencia(
    origem: Path, destino: Path, tipo: np.dtype, com_mmwave: bool
) -> int:
    """Empacota uma sequência num .npz. Devolve o tamanho em bytes."""
    conteudo = {}

    amplitude, fase = _ler_wifi(origem / "wifi-csi")
    # A amplitude tem -inf em ~5% dos arquivos (é log10(0)). Preservamos como
    # está: o tratamento é responsabilidade do loader, não do empacotador — assim
    # a versão leve continua fiel ao original.
    conteudo["csi_amp"] = amplitude.astype(tipo)
    conteudo["csi_fase"] = fase.astype(tipo)

    conteudo["pose"] = np.load(origem / "ground_truth.npy").astype(np.float32)

    if com_mmwave and (origem / "mmwave").is_dir():
        pontos, cortes = _ler_mmwave(origem / "mmwave")
        conteudo["mmwave_pontos"] = pontos.astype(np.float32)
        conteudo["mmwave_cortes"] = cortes

    # Escrita atômica: grava num temporário e só então renomeia. Sem isso, uma
    # interrupção no meio deixa um .npz que ABRE SEM ERRO mas com chaves
    # faltando — e o loader carrega esse arquivo em silêncio, sem o mmWave.
    #
    # Cuidado: savez_compressed acrescenta ".npz" se o nome não terminar nisso,
    # então o temporário precisa terminar em .npz.
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(".tmp.npz")
    try:
        np.savez_compressed(temporario, **conteudo)
        os.replace(temporario, destino)   # atômico no Windows e no Linux
    except BaseException:
        temporario.unlink(missing_ok=True)
        raise
    return destino.stat().st_size


LEIAME = """# MM-Fi — versão leve

Subconjunto do [MM-Fi](https://github.com/ybhbingo/MMFi_dataset) com apenas as
modalidades usadas neste projeto: **WiFi CSI**, **mmWave** e a **pose 3D**.

Gerado por `scripts/empacotar_mmfi.py`. Precisão dos dados de CSI: **{precisao}**.

## Estrutura

    E01/S01/A01.npz     ← uma sequência inteira (297 quadros) por arquivo

Cada `.npz` contém:

| Chave | Forma | O que é |
| --- | --- | --- |
| `csi_amp` | (297, 3, 114, 10) | Amplitude do CSI, **em dB** |
| `csi_fase` | (297, 3, 114, 10) | Fase do CSI, crua, em radianos (±π) |
| `pose` | (297, 17, 3) | Pose 3D em metros, esqueleto Human3.6M |
| `mmwave_pontos` | (N, 5) | Nuvens de pontos concatenadas{mmwave_nota} |
| `mmwave_cortes` | (298,) | O quadro `i` vai de `cortes[i]` a `cortes[i+1]` |

## Como usar

O código do projeto lê este formato automaticamente. Basta apontar o caminho:

```cmd
set MMFI_DATASET=C:\\caminho\\para\\MMFi_leve
```

```
.venv/Scripts/python.exe scripts/inspecionar_mmfi.py --sujeito S18 --acao A22
```

## Avisos

- `csi_amp` está em **dB**, não em escala linear. Os `-inf` (em ~5% dos quadros)
  são `log10(0)` e foram **preservados como no original** — o loader os trata.
- O esqueleto é **Human3.6M** de 17 juntas, não COCO, e o eixo **y cresce para
  baixo**.
{nota_precisao}
"""

NOTA_FLOAT16 = """- O CSI foi salvo em `float16` para reduzir o tamanho. Erro máximo medido:
  **0,0156 dB** na amplitude e **0,001 rad** na fase — bem abaixo do ruído de
  medição do rádio, mas converta para `float32` antes de contas longas.
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--destino", required=True,
                        help="Pasta onde gravar a versão leve")
    parser.add_argument("--origem", default=None,
                        help="Raiz do MM-Fi original (padrão: MMFI_DATASET)")
    parser.add_argument("--precisao", default="float32", choices=sorted(PRECISOES),
                        help="Precisão do CSI (a pose é sempre float32, que é o "
                             "dtype original). float16 reduz pela metade, com erro "
                             "de 0,0156 dB (padrão: float32)")
    parser.add_argument("--sem-mmwave", action="store_true",
                        help="Empacotar só WiFi e pose")
    parser.add_argument("--sujeitos", nargs="+", default=None,
                        help="Só estes sujeitos, ex.: --sujeitos S01 S02")
    parser.add_argument("--refazer", action="store_true",
                        help="Regravar sequências já empacotadas")
    args = parser.parse_args()

    origem = caminho_dataset(args.origem)
    destino = Path(args.destino).expanduser()
    tipo = PRECISOES[args.precisao]
    com_mmwave = not args.sem_mmwave

    if not origem.is_dir():
        print(f"ERRO: MM-Fi não encontrado em {origem}")
        return 1

    sujeitos = args.sujeitos or [
        p.name
        for ambiente in AMBIENTES
        if (origem / ambiente).is_dir()
        for p in sorted((origem / ambiente).iterdir()) if p.is_dir()
    ]
    tarefas = [(s, a) for s in sujeitos for a in sorted(ACOES)]

    # O LEIAME vai ANTES do laço: se o processo morrer no meio, o pacote parcial
    # ainda carrega o documento que explica escala dB, -inf, Human3.6M e y-para-baixo.
    destino.mkdir(parents=True, exist_ok=True)
    nota = NOTA_FLOAT16 if args.precisao == "float16" else ""
    (destino / "LEIAME.md").write_text(
        LEIAME.format(
            precisao=args.precisao,
            nota_precisao=nota,
            mmwave_nota="" if com_mmwave else " *(não incluído)*",
        ),
        encoding="utf-8",
    )

    print(f"Origem  : {origem}")
    print(f"Destino : {destino}")
    print(f"Precisão: {args.precisao}   mmWave: {'sim' if com_mmwave else 'não'}")
    print(f"Sequências a processar: {len(tarefas)}\n")

    inicio = time.time()
    bytes_gravados = feitas = puladas = falhas = 0

    for indice, (sujeito, acao) in enumerate(tarefas, start=1):
        try:
            ambiente = ambiente_do_sujeito(sujeito)
        except ValueError as erro:
            # Um nome de sujeito inválido não pode abortar um run de horas.
            print(f"  [{indice}/{len(tarefas)}] {sujeito}: {erro}")
            falhas += 1
            continue

        pasta_origem = origem / ambiente / sujeito / acao
        arquivo = destino / ambiente / sujeito / f"{acao}.npz"

        if arquivo.exists() and not args.refazer:
            bytes_gravados += arquivo.stat().st_size
            puladas += 1
            continue
        if not pasta_origem.is_dir():
            print(f"  [{indice}/{len(tarefas)}] {sujeito}/{acao}: ausente na origem")
            falhas += 1
            continue

        try:
            bytes_gravados += empacotar_sequencia(
                pasta_origem, arquivo, tipo, com_mmwave
            )
            feitas += 1
        except KeyboardInterrupt:
            print(f"\n  Interrompido em {sujeito}/{acao}. "
                  f"O que já foi gravado continua válido — rode de novo para retomar.")
            break
        except Exception as erro:
            print(f"  [{indice}/{len(tarefas)}] {sujeito}/{acao}: FALHOU — {erro}")
            falhas += 1
            continue

        if feitas % 20 == 0 or indice == len(tarefas):
            decorrido = time.time() - inicio
            restantes = len(tarefas) - indice
            eta = decorrido / max(feitas, 1) * restantes / 60
            print(f"  [{indice}/{len(tarefas)}] {bytes_gravados / 1024**3:.2f} GB "
                  f"gravados · {decorrido / 60:.0f} min · faltam ~{eta:.0f} min")

    print(f"\n{'=' * 60}")
    print(f"Empacotadas : {feitas}")
    if puladas:
        print(f"Já existiam : {puladas} (use --refazer para regravar)")
    if falhas:
        print(f"Falhas      : {falhas}")
    print(f"Tamanho     : {bytes_gravados / 1024**3:.2f} GB")
    print(f"Tempo       : {(time.time() - inicio) / 60:.0f} min")

    print(f"\n⚠️  ANTES de apagar o original ou mandar o pacote para alguém:")
    print(f"    python scripts/verificar_pacote_mmfi.py --pacote {destino} --contra-original")
    print(f"\nPara usar:  set MMFI_DATASET={destino}")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
