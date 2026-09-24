"""
Verifica a integridade de uma versão leve do MM-Fi gerada por empacotar_mmfi.py.

**Rode isto antes de apagar o dataset original ou de mandar o pacote para alguém.**

Um `.npz` é um zip. Se a gravação for interrompida no meio, o arquivo pode ficar
legível mas com chaves faltando — e aí uma sequência sem `mmwave_pontos`, ou sem
a pose, carrega sem erro nenhum e o problema só aparece semanas depois. O mesmo
vale para download truncado.

Por padrão, lê cada array por inteiro — o que força o CRC do zip e detecta
corrupção silenciosa de bytes, além de chaves faltando e formas erradas.

Com ``--contra-original``, compara os valores com o dataset original. É a única
checagem que prova fidelidade, e a que justifica apagar o original.

Uso:
    .venv/Scripts/python.exe scripts/verificar_pacote_mmfi.py --pacote D:/MMFi_leve
    .venv/Scripts/python.exe scripts/verificar_pacote_mmfi.py --pacote D:/MMFi_leve --contra-original
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mmfi import (  # noqa: E402
    ACOES,
    AMBIENTES,
    N_ANTENAS,
    N_JUNTAS,
    N_QUADROS,
    N_SUBPORTADORAS,
    N_TEMPO,
    ambiente_do_sujeito,
    caminho_dataset,
)

FORMAS = {
    "csi_amp": (N_QUADROS, N_ANTENAS, N_SUBPORTADORAS, N_TEMPO),
    "csi_fase": (N_QUADROS, N_ANTENAS, N_SUBPORTADORAS, N_TEMPO),
    "pose": (N_QUADROS, N_JUNTAS, 3),
}


def verificar_arquivo(caminho: Path, espera_mmwave: bool) -> list:
    """Devolve a lista de problemas encontrados. Vazia = arquivo íntegro.

    Lê cada array por inteiro de propósito: é a leitura que valida o CRC do zip
    e pega corrupção de bytes que a checagem de cabeçalho deixaria passar.

    ``espera_mmwave`` vem da varredura do pacote como um todo — ver
    :func:`detectar_mmwave`. Sem isso, uma sequência cuja gravação foi
    interrompida logo após a pose passaria por "pacote sem mmWave", que é uma
    configuração legítima.
    """
    problemas = []
    try:
        with np.load(caminho) as pacote:
            chaves = set(pacote.files)

            for chave, forma in FORMAS.items():
                if chave not in chaves:
                    problemas.append(f"chave ausente: {chave}")
                    continue
                if pacote[chave].shape != forma:
                    problemas.append(
                        f"{chave}: forma {pacote[chave].shape}, esperado {forma}"
                    )

            tem_pontos = "mmwave_pontos" in chaves
            tem_cortes = "mmwave_cortes" in chaves
            if espera_mmwave and not (tem_pontos and tem_cortes):
                problemas.append(
                    "mmwave ausente, mas o resto do pacote tem — "
                    "provável gravação interrompida"
                )
            elif tem_pontos != tem_cortes:
                problemas.append("mmwave incompleto: só uma das duas chaves")
            elif tem_pontos:
                pontos, cortes = pacote["mmwave_pontos"], pacote["mmwave_cortes"]
                if pontos.ndim != 2 or pontos.shape[1] != 5:
                    problemas.append(f"mmwave_pontos: forma {pontos.shape}, esperado (N, 5)")
                if len(cortes) != N_QUADROS + 1:
                    problemas.append(
                        f"mmwave_cortes: {len(cortes)} valores, esperado {N_QUADROS + 1}"
                    )
                elif int(cortes[-1]) != len(pontos):
                    problemas.append(
                        f"mmwave_cortes: último corte {int(cortes[-1])} != "
                        f"{len(pontos)} pontos"
                    )
                elif np.any(np.diff(cortes) < 0):
                    problemas.append("mmwave_cortes: não é crescente")

            for chave in chaves:
                arr = pacote[chave]  # a leitura em si valida o CRC do zip
                # -inf em csi_amp é esperado (é log10(0)); NaN não é, em lugar nenhum.
                if arr.dtype.kind == "f" and np.isnan(arr).any():
                    problemas.append(f"{chave}: contém NaN")

    except zipfile.BadZipFile:
        problemas.append("não é um zip válido (gravação ou download truncado)")
    except Exception as erro:
        problemas.append(f"falha ao abrir: {type(erro).__name__}: {erro}")

    return problemas


def detectar_mmwave(pacote: Path) -> bool:
    """O pacote foi gerado COM mmWave?

    Decide pela maioria de uma amostra: se a maior parte das sequências tem as
    chaves de mmWave, então as que não têm estão defeituosas. Assim distinguimos
    "gerado com --sem-mmwave" (legítimo) de "gravação interrompida" (defeito).

    A amostra sai dos arquivos que **existem no disco**, espalhados uniformemente.
    Amostrar a lista de sequências esperadas daria errado em pacote incompleto:
    o passo fixo poderia cair só em arquivos ausentes — ou, pior, só no único
    que está quebrado.
    """
    arquivos = sorted(pacote.rglob("*.npz"))
    if not arquivos:
        return False

    amostra = (
        arquivos
        if len(arquivos) <= 30
        else [arquivos[i] for i in np.linspace(0, len(arquivos) - 1, 30, dtype=int)]
    )

    com = total = 0
    for arquivo in amostra:
        try:
            with np.load(arquivo) as dados:
                com += "mmwave_pontos" in dados.files
                total += 1
        except Exception:
            continue  # arquivo quebrado não vota; o verificador o pega depois
    return total > 0 and com > total / 2


def comparar_com_original(
    caminho: Path, sujeito: str, acao: str, origem: Path, quadros: int
) -> list:
    """Compara alguns quadros contra os .mat originais."""
    import scipy.io as sio

    problemas = []
    pasta = origem / ambiente_do_sujeito(sujeito) / sujeito / acao
    if not (pasta / "wifi-csi").is_dir():
        return [f"original ausente para comparação: {pasta}"]

    indices = np.linspace(1, N_QUADROS, quadros, dtype=int)
    with np.load(caminho) as pacote:
        amp, fase, pose = pacote["csi_amp"], pacote["csi_fase"], pacote["pose"]
        poses_orig = np.load(pasta / "ground_truth.npy")

        for quadro in indices:
            dados = sio.loadmat(pasta / "wifi-csi" / f"frame{quadro:03d}.mat")
            # Tolerância de float32: o empacotador converte de float64.
            for nome, empacotado, original in (
                ("amplitude", amp[quadro - 1], dados["CSIamp"]),
                ("fase", fase[quadro - 1], dados["CSIphase"]),
            ):
                validos = np.isfinite(original)
                if not np.allclose(
                    np.asarray(empacotado)[validos], original[validos],
                    rtol=1e-5, atol=1e-5,
                ):
                    problemas.append(f"quadro {quadro}: {nome} diverge do original")
                # A máscara de -inf tem que ser idêntica, não só os valores.
                if not np.array_equal(np.isfinite(np.asarray(empacotado)), validos):
                    problemas.append(f"quadro {quadro}: máscara de -inf diverge")

            if not np.allclose(pose[quadro - 1], poses_orig[quadro - 1]):
                problemas.append(f"quadro {quadro}: pose diverge do original")

    return problemas


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pacote", required=True, help="Pasta da versão leve")
    parser.add_argument("--contra-original", action="store_true",
                        help="Comparar valores com o dataset original")
    parser.add_argument("--origem", default=None,
                        help="Raiz do MM-Fi original (padrão: MMFI_DATASET)")
    parser.add_argument("--quadros-por-sequencia", type=int, default=5,
                        help="Quantos quadros comparar por sequência (padrão: 5)")
    args = parser.parse_args()

    pacote = Path(args.pacote).expanduser()
    if not pacote.is_dir():
        print(f"ERRO: pacote não encontrado em {pacote}")
        return 1

    esperadas = [
        (ambiente, f"S{n:02d}", acao)
        for i, ambiente in enumerate(AMBIENTES)
        for n in range(i * 10 + 1, i * 10 + 11)
        for acao in sorted(ACOES)
    ]

    origem = caminho_dataset(args.origem) if args.contra_original else None
    if args.contra_original and not origem.is_dir():
        print(f"ERRO: original não encontrado em {origem}")
        return 1

    espera_mmwave = detectar_mmwave(pacote)

    print(f"Pacote : {pacote}")
    print(f"mmWave : {'esperado em todas as sequências' if espera_mmwave else 'não incluído neste pacote'}")
    if args.contra_original:
        print(f"Comparando com o original em: {origem}")
    print(f"Esperadas: {len(esperadas)} sequências\n")

    inicio = time.time()
    ok = ausentes = quebradas = 0
    defeitos = []
    total_bytes = 0

    for indice, (ambiente, sujeito, acao) in enumerate(esperadas, start=1):
        arquivo = pacote / ambiente / sujeito / f"{acao}.npz"
        rotulo = f"{ambiente}/{sujeito}/{acao}"

        if not arquivo.is_file():
            ausentes += 1
            defeitos.append((rotulo, ["ausente"]))
            continue

        total_bytes += arquivo.stat().st_size
        problemas = verificar_arquivo(arquivo, espera_mmwave)
        if not problemas and args.contra_original:
            problemas = comparar_com_original(
                arquivo, sujeito, acao, origem, args.quadros_por_sequencia
            )

        if problemas:
            quebradas += 1
            defeitos.append((rotulo, problemas))
        else:
            ok += 1

        if indice % 100 == 0 or indice == len(esperadas):
            decorrido = time.time() - inicio
            print(f"  [{indice}/{len(esperadas)}] ok={ok} problemas={quebradas + ausentes} "
                  f"· {decorrido / 60:.1f} min")

    print(f"\n{'=' * 64}")
    print(f"Íntegras  : {ok} de {len(esperadas)}")
    print(f"Ausentes  : {ausentes}")
    print(f"Com defeito: {quebradas}")
    print(f"Tamanho   : {total_bytes / 1024**3:.2f} GB")
    print(f"Tempo     : {(time.time() - inicio) / 60:.1f} min")

    if defeitos:
        print(f"\n{'-' * 64}\nPROBLEMAS (até 30):")
        for rotulo, problemas in defeitos[:30]:
            print(f"  {rotulo}: {'; '.join(problemas[:3])}")
        if len(defeitos) > 30:
            print(f"  ... e mais {len(defeitos) - 30}")
        print("\nRegrave só as afetadas com:")
        print("  empacotar_mmfi.py --destino <pacote> --sujeitos <S..> --refazer")
        print("\n⚠️  NÃO apague o dataset original enquanto houver problema aqui.")
        return 1

    print("\n✅ Pacote íntegro.")
    if not args.contra_original:
        print("   Para provar fidelidade antes de apagar o original, rode também")
        print("   com --contra-original.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
