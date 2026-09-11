"""
Script de linha de comando para inspecionar amostras do MM-Fi e gerar gráficos
do CSI (amplitude e fase) e da pose 3D. Atende à Tarefa (b) da disciplina.

Uso:
    .venv/Scripts/python.exe scripts/inspecionar_mmfi.py
    .venv/Scripts/python.exe scripts/inspecionar_mmfi.py --sujeito S12 --acao A12 --quadro 150
    .venv/Scripts/python.exe scripts/inspecionar_mmfi.py --movimento

Se o dataset não estiver em ~/Downloads/datas/MMFi_Dataset, aponte o caminho:
    set MMFI_DATASET=D:\\caminho\\para\\MMFi_Dataset       (cmd)
    export MMFI_DATASET=/d/caminho/para/MMFi_Dataset       (Git Bash)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# O console do Windows usa cp1252 por padrão e engasga com acentuação.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Adiciona raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mmfi import (  # noqa: E402
    ACOES,
    FRONTEIRA_20MHZ,
    N_QUADROS,
    caminho_dataset,
    carregar_quadro,
    sanitizar_fase,
)
from src.visualizacao_mmfi import (  # noqa: E402
    plotar_amostra_completa,
    plotar_movimento,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspecionar amostra do MM-Fi")
    parser.add_argument("--sujeito", default="S01", help="S01 a S40")
    parser.add_argument("--acao", default="A01", choices=sorted(ACOES), help="A01 a A27")
    parser.add_argument("--quadro", type=int, default=1, help=f"1 a {N_QUADROS}")
    parser.add_argument("--raiz", default=None, help="Caminho do MMFi_Dataset")
    parser.add_argument("--salvar", default="artifacts/figuras/mmfi_amostra.png")
    parser.add_argument("--mostrar", action="store_true",
                        help="Abrir a janela interativa do gráfico")
    parser.add_argument("--movimento", action="store_true",
                        help="Gerar também o painel de esqueletos ao longo da sequência")
    args = parser.parse_args()

    raiz = caminho_dataset(args.raiz)
    print(f"--> Dataset em: {raiz}")
    if not raiz.is_dir():
        print(f"ERRO: diretório não encontrado.\n"
              f"Baixe o MM-Fi ou aponte a variável de ambiente MMFI_DATASET.")
        return 1

    print(f"--> Carregando {args.sujeito} / {args.acao} / quadro {args.quadro}")
    try:
        amplitude, fase, juntas, meta = carregar_quadro(
            args.sujeito, args.acao, args.quadro, args.raiz
        )
    except (FileNotFoundError, ValueError) as erro:
        print(f"ERRO: {erro}")
        return 1

    print("\n=== Metadados da amostra ===")
    for chave, valor in meta.items():
        print(f"  {chave}: {valor}")

    print("\n=== Estrutura das variáveis ===")
    print(f"  CSIamp   : {amplitude.shape} (3 antenas RX x 114 subportadoras x 10 tempos)"
          f" | {amplitude.dtype} | escala dB")
    print(f"             faixa [{amplitude.min():.2f}, {amplitude.max():.2f}]")
    print(f"  CSIphase : {fase.shape} | {fase.dtype}")
    print(f"             faixa [{fase.min():.3f}, {fase.max():.3f}] rad "
          f"(±π = ±{np.pi:.3f} → fase crua, envolvida)")
    print(f"  Esqueleto: {juntas.shape} (17 juntas x [x, y, z]) | {juntas.dtype}")
    print(f"             x [{juntas[:,0].min():+.2f}, {juntas[:,0].max():+.2f}] m")
    print(f"             y [{juntas[:,1].min():+.2f}, {juntas[:,1].max():+.2f}] m  "
          f"(⚠️ cresce para BAIXO)")
    print(f"             z [{juntas[:,2].min():+.2f}, {juntas[:,2].max():+.2f}] m  "
          f"(distância até a câmera)")

    altura = np.linalg.norm(juntas[10] - juntas[[3, 6]].mean(axis=0))
    print(f"\n  Vão cabeça→pés: {altura:.2f} m (confere que a unidade é metro)")

    # Efeito da sanitização de fase — o número que a tarefa (d) vai perseguir.
    #
    # As contas abaixo são feitas POR BLOCO de 20 MHz e POR TRAÇO, para bater com
    # o que sanitizar_fase realmente faz. Ajustar uma reta sobre as 114 de uma vez
    # dá inclinação diferente (às vezes com sinal trocado), e medir o desvio do
    # tensor inteiro de uma vez soma os offsets DC dos 30 traços, inflando o ganho.
    print("\n=== Efeito da sanitização de fase (tarefa d) ===")
    blocos = [(0, FRONTEIRA_20MHZ), (FRONTEIRA_20MHZ, fase.shape[1])]
    for antena in range(fase.shape[0]):
        partes = []
        for inicio, fim in blocos:
            traco = np.unwrap(fase[antena, inicio:fim, 0])
            inclinacao = np.polyfit(np.arange(fim - inicio), traco, 1)[0]
            partes.append(f"{inclinacao:+.4f}")
        print(f"  antena {antena + 1}: inclinação por bloco de 20 MHz = "
              f"{partes[0]} e {partes[1]} rad/subportadora")

    saneada = sanitizar_fase(fase)
    antes = np.mean([np.unwrap(fase[a, i:f, t]).std()
                     for a in range(fase.shape[0])
                     for t in range(fase.shape[2])
                     for i, f in blocos])
    depois = np.mean([saneada[a, i:f, t].std()
                      for a in range(saneada.shape[0])
                      for t in range(saneada.shape[2])
                      for i, f in blocos])
    print(f"\n  desvio padrão médio por traço  antes: {antes:.3f} rad")
    print(f"                                depois: {depois:.3f} rad")
    print(f"  redução: {100 * (antes - depois) / antes:.1f}%")

    print("\n--> Gerando visualizações...")
    try:
        plotar_amostra_completa(
            args.sujeito, args.acao, args.quadro, args.raiz,
            salvar_em=args.salvar, mostrar=args.mostrar,
        )
        if args.movimento:
            destino = str(Path(args.salvar).with_name(
                f"mmfi_movimento_{args.sujeito}_{args.acao}.png"))
            plotar_movimento(args.sujeito, args.acao, raiz=args.raiz,
                             salvar_em=destino, mostrar=args.mostrar)
    except (FileNotFoundError, ValueError) as erro:
        print(f"ERRO ao gerar figuras: {erro}")
        return 1

    print("Concluído!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
