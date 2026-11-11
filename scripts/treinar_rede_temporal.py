"""
Treina e avalia a rede temporal própria (src/rede_temporal.py) no S2 (ML-04).

1. Divide o S2 em treino, validação (``SUJEITOS_VALIDACAO_S2`` = S04, S14, S24,
   S34, retirados do treino) e teste (os 8 sujeitos oficiais). A rede treina
   só com as **28** pessoas restantes — não há retreino com as 32 depois da
   escolha da época (os RF e as redes neurais v1 e v2 treinam com as 32).
2. Treina só no treino; a cada época mede a validação. A época de maior
   acurácia por sequência na validação é a escolhida, e é gravada no JSON
   **antes** de o teste ser tocado.
3. O modelo dessa época é avaliado **uma vez** no teste S2 com
   ``src/avaliacao.avaliar`` (por janela e por sequência, IC 95% por bootstrap
   de sequências).

Uso:
    .venv/Scripts/python.exe scripts/treinar_rede_temporal.py --seed 0
    .venv/Scripts/python.exe scripts/treinar_rede_temporal.py --seed 0 1 2

Com várias seeds, os dados são carregados uma vez só (~4,4 GB em RAM, ~45 s).
Uma época leva ~40 s com a CPU livre (~10–12 min por seed); dividindo a CPU com
outro treino, foram medidos 43–45 min por seed.

Saídas, por seed (``<sufixo>`` = ``_j<janela>_e<epocas>`` só fora do padrão 30/15):
    artifacts/resultados/rede_temporal_s2_seed<seed><sufixo>.json
    artifacts/figuras/confusao_rede_temporal_s2_seed<seed><sufixo>.png   (por sequência)

Reusa o cache do CSI do ML-01 (artifacts/cache/), construindo-o se faltar.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROJETO))

from src.atributos import janelas_de_sequencia  # noqa: E402
from src.avaliacao import avaliar, salvar_matriz_confusao  # noqa: E402
from src.mmfi import N_QUADROS, PROTOCOLO_3  # noqa: E402
from src.protocolos import SUJEITOS_VALIDACAO_S2  # noqa: E402
from src.rede_temporal import (  # noqa: E402
    HIPERPARAMETROS,
    N_CANAIS,
    DadosRede,
    RedeTemporal,
    carregar_dados,
    contar_parametros,
    dividir_rede,
    estatisticas_canal,
    prever,
    treinar,
)

ESTADO_PARCIAL = "época escolhida; teste ainda não avaliado"
ESTADO_COMPLETO = "completo"

Log = Callable[[str], None]


def nome_base(seed: int, janela: int = HIPERPARAMETROS["janela"],
              epocas: int = HIPERPARAMETROS["epocas"]) -> str:
    """``rede_temporal_s2_seed<seed>``, com ``_j<janela>_e<epocas>`` fora do padrão.

    Assim uma execução com outra configuração não sobrescreve (nem entra na
    comparação, que só lê o padrão) as do padrão.
    """
    nome = f"rede_temporal_s2_seed{seed}"
    if (janela, epocas) != (HIPERPARAMETROS["janela"], HIPERPARAMETROS["epocas"]):
        nome += f"_j{janela}_e{epocas}"
    return nome


def descrever_unidade(janela: int, n_quadros: int = N_QUADROS) -> str:
    m = len(janelas_de_sequencia(n_quadros, janela))
    return (f"janela de {janela / 10:g} s ({janela} quadros); "
            f"por sequência = média das {m} janelas")


def montar_cabecalho(acoes: Sequence[str], janela: int, epocas: int,
                     n_sequencias: Dict[str, int], versoes: Optional[Dict] = None) -> Dict:
    return {
        "modelo": "rede_temporal",
        "descricao": "Conv1d por quadro (subportadoras) + GRU bidirecional sobre a janela",
        "protocolo": "s2",
        "acoes": list(acoes),
        "unidade": descrever_unidade(janela),
        "hiperparametros": {**HIPERPARAMETROS, "epocas": epocas, "janela": janela},
        "n_parametros": contar_parametros(RedeTemporal(n_canais=N_CANAIS, n_classes=len(acoes))),
        "sujeitos_validacao": list(SUJEITOS_VALIDACAO_S2),
        "n_sequencias": dict(n_sequencias),
        "treino_sem_retreino": "treina só nos sujeitos fora da validação (28 no S2); "
                               "não retreina com os 32 depois de escolher a época",
        "criterio_epoca": "maior acurácia por sequência na validação (empate: maior por "
                          "janela; depois a época mais cedo); o teste não participa",
        "acaso": 1.0 / len(acoes),
        "versoes": dict(versoes or {}),
    }


def preparar_dados(amp, fase, linhas, acoes: Sequence[str],
                   divisao: Tuple[Sequence, Sequence, Sequence],
                   log: Optional[Log] = None) -> Tuple[Dict[str, DadosRede], np.ndarray, np.ndarray]:
    """Carrega treino, validação e teste e calcula a padronização **só do treino**.

    ``divisao`` = ``(treino, validacao, teste)`` (ver ``dividir_rede``).
    """
    dados = {}
    for nome, seqs in zip(("treino", "validação", "teste"), divisao):
        if log is not None:
            log(f"--> Carregando {nome}")
        dados[nome] = carregar_dados(amp, fase, linhas, seqs, acoes, log=log)
    media, desvio = estatisticas_canal(dados["treino"].x)
    return dados, media, desvio


def rodar_seed(dados: Dict[str, DadosRede], media: np.ndarray, desvio: np.ndarray,
               seed: int, cabecalho: Dict, acoes: Sequence[str],
               pasta_resultados: Path, pasta_figuras: Path,
               epocas: int = HIPERPARAMETROS["epocas"], janela: int = HIPERPARAMETROS["janela"],
               lote: int = HIPERPARAMETROS["lote"],
               gravar: Optional[Callable[[Path, Dict], None]] = None,
               log: Log = print) -> Dict:
    """Treina uma seed, grava a época escolhida, avalia o teste **uma vez** e grava.

    Devolve o dicionário final (o mesmo gravado em JSON).
    """
    gravar = gravar or _gravar
    t0 = time.time()
    modelo, historico, epoca = treinar(
        dados["treino"], dados["validação"], media, desvio, seed, len(acoes),
        epocas=epocas, janela=janela, lote=lote, log=log)
    base = nome_base(seed, janela, epocas)
    saida = Path(pasta_resultados) / f"{base}.json"
    parcial = {**cabecalho, "seed": seed, "historico": historico, "epoca_escolhida": epoca,
               "validacao_epoca_escolhida": historico[epoca - 1]["validacao"],
               "escolha_registrada_em": time.strftime("%Y-%m-%d %H:%M:%S")}
    gravar(saida, {**parcial, "estado": ESTADO_PARCIAL})
    log(f"--> Época escolhida pela validação: {epoca} (gravada em {saida} antes do teste)")

    proba, y, grupos = prever(modelo, dados["teste"], media, desvio, janela)
    metricas = avaliar(proba, y, grupos, len(acoes), seed=seed)
    j, q = metricas["por_janela"], metricas["por_sequencia"]
    log(f"--> Teste S2: janela {100 * j['acuracia']:5.1f}% "
        f"IC95% [{100 * j['ic95'][0]:4.1f}, {100 * j['ic95'][1]:4.1f}]   "
        f"sequência {100 * q['acuracia']:5.1f}% "
        f"IC95% [{100 * q['ic95'][0]:4.1f}, {100 * q['ic95'][1]:4.1f}]   "
        f"(acaso {100 / len(acoes):.1f}%)")

    figura = Path(pasta_figuras) / f"confusao_{base}.png"
    salvar_matriz_confusao(
        q["confusao"], acoes, figura,
        titulo=f"Rede temporal, S2, seed {seed}, por sequência "
               f"(acurácia {100 * q['acuracia']:.1f}%)")
    final = {**parcial, "teste": metricas,
             "figura_confusao": str(figura.relative_to(RAIZ_PROJETO)
                                    if figura.is_relative_to(RAIZ_PROJETO) else figura),
             "estado": ESTADO_COMPLETO, "tempo_s": round(time.time() - t0, 1),
             "data": time.strftime("%Y-%m-%d %H:%M:%S")}
    gravar(saida, final)
    log(f"--> Resultado: {saida}\n--> Figura: {figura}")
    return final


def _gravar(caminho: Path, dados: Dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rede temporal própria no S2 (ML-04)")
    parser.add_argument("--seed", type=int, nargs="+", default=[0],
                        help="Uma ou mais seeds (ex.: --seed 0 1 2)")
    parser.add_argument("--epocas", type=int, default=HIPERPARAMETROS["epocas"])
    parser.add_argument("--janela", type=int, default=HIPERPARAMETROS["janela"],
                        help="Quadros por janela (30 = 3 s)")
    parser.add_argument("--raiz", default=None, help="Caminho do MM-Fi (padrão: MMFI_DATASET)")
    parser.add_argument("--cache", default=str(RAIZ_PROJETO / "artifacts" / "cache"))
    parser.add_argument("--resultados", default=str(RAIZ_PROJETO / "artifacts" / "resultados"))
    parser.add_argument("--figuras", default=str(RAIZ_PROJETO / "artifacts" / "figuras"))
    args = parser.parse_args()
    if args.epocas < 1:
        print("ERRO: --epocas deve ser ≥ 1.")
        return 1
    if not 1 <= args.janela <= N_QUADROS:
        print(f"ERRO: --janela deve estar entre 1 e {N_QUADROS}.")
        return 1

    import torch
    from src.cache_csi import abrir_cache, construir_cache

    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        print(f"--> Cache do CSI em: {args.cache}")
        construir_cache(args.cache, args.raiz)
        amp, fase, linhas = abrir_cache(args.cache)
    except (FileNotFoundError, ValueError) as erro:
        print(f"ERRO: {erro}")
        return 1

    acoes = tuple(sorted(PROTOCOLO_3))
    divisao = dividir_rede(acoes)
    print(f"--> S2 / P3: {len(divisao[0])} sequências de treino, {len(divisao[1])} de validação "
          f"({', '.join(SUJEITOS_VALIDACAO_S2)}), {len(divisao[2])} de teste")

    inicio = time.time()
    dados, media, desvio = preparar_dados(amp, fase, linhas, acoes, divisao, log=print)
    print(f"--> Dados prontos em {time.time() - inicio:.0f} s "
          f"({sum(d.x.nbytes for d in dados.values()) / 1e9:.1f} GB em float16)", flush=True)

    import scipy
    import sklearn
    cabecalho = montar_cabecalho(
        acoes, args.janela, args.epocas,
        {"treino": len(divisao[0]), "validacao": len(divisao[1]), "teste": len(divisao[2])},
        {"torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
         "scipy": scipy.__version__, "python": platform.python_version(),
         "threads_torch": torch.get_num_threads()})

    def log(texto: str) -> None:
        print(texto, flush=True)

    for seed in args.seed:
        log(f"\n=== Seed {seed}: treino ({args.epocas} épocas) ===")
        rodar_seed(dados, media, desvio, seed, cabecalho, acoes, Path(args.resultados),
                   Path(args.figuras), epocas=args.epocas, janela=args.janela, log=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
