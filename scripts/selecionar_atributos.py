"""
Escolhe a combinação de conjuntos de atributos por **validação** e só então
avalia o teste S2 (ML-03).

Um protótipo mostrou que somar atributos de Doppler ajuda a Random Forest, mas a
combinação foi escolhida olhando o teste — o número estava inflado pela
seleção. Aqui a escolha é honesta:

1. **Validação** — os sujeitos ``SUJEITOS_VALIDACAO_S2`` (S04, S14, S24, S34:
   um por ambiente, nenhum do teste) saem do treino do S2. Cada candidata é
   treinada no restante e medida neles, com 3 seeds. Ganha a maior **média de
   acurácia por sequência** (empate: maior média por janela; depois, a
   primeira da lista, que é a mais simples).
2. A escolha é gravada no JSON **antes** de qualquer contato com o teste.
3. **Teste** — a escolhida e o ``base`` (referência) são treinados no treino
   S2 completo com 5 seeds e avaliados uma vez no teste S2: média ± desvio
   entre seeds, e o IC 95% por bootstrap de sequências de cada seed.
4. **Comparação pareada** — as duas são medidas nas mesmas 216 sequências de
   teste, então a diferença (escolhida − base) é avaliada por bootstrap
   **pareado** de sequências: por seed e na média das seeds, com IC 95% e a
   fração de reamostras em que a diferença é ≤ 0.

Só Random Forest, só S2 com as 27 ações (P3). Os 5 candidatos vieram de um
protótipo que comparou combinações **no teste**: escolher pela validação reduz
esse viés, mas não o elimina.

Uso:
    .venv/Scripts/python.exe scripts/selecionar_atributos.py

Saída: artifacts/resultados/selecao_atributos.json e tabelas no terminal.
Reusa o cache do ML-01 (artifacts/cache/) e o de atributos
(artifacts/cache_atributos/), construindo o que faltar.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROJETO))
sys.path.insert(0, str(RAIZ_PROJETO / "scripts"))

from src.atributos import janelas_de_sequencia, montar_conjunto, validar_atributos  # noqa: E402
from src.avaliacao import agregar_por_sequencia, avaliar  # noqa: E402
from src.mmfi import N_QUADROS, PROTOCOLO_3, SUJEITOS_TESTE_S2, ambiente_do_sujeito  # noqa: E402
from src.protocolos import SUJEITOS_VALIDACAO_S2, dividir_s2, separar_validacao  # noqa: E402
from treinar_baseline import criar_modelo  # noqa: E402

CANDIDATOS: Tuple[Tuple[str, ...], ...] = (
    ("base",),
    ("base", "antenas"),
    ("base", "dop_amp", "dop_cm"),
    ("base", "antenas", "dop_amp", "dop_cm"),
    ("dop_amp", "dop_cm"),
)
REFERENCIA = ("base",)
SEEDS_VALIDACAO = 3
SEEDS_TESTE = 5

Blocos = Dict[str, np.ndarray]


def nome_combinacao(combinacao: Sequence[str]) -> str:
    return ",".join(combinacao)


def juntar(blocos: Blocos, combinacao: Sequence[str],
           linhas: Optional[np.ndarray] = None) -> np.ndarray:
    """Concatena os conjuntos na ordem da combinação (opcionalmente só ``linhas``).

    Igual a ``montar_conjunto(..., atributos=combinacao)``, desde que cada
    bloco tenha sido montado com as mesmas sequências na mesma ordem.
    """
    partes = [blocos[nome] if linhas is None else blocos[nome][linhas]
              for nome in combinacao]
    return np.concatenate(partes, axis=1)


def treinar_e_medir(X_tr, y_tr, X_te, y_te, g_te, n_classes: int, seed: int,
                    acertos: Optional[Dict] = None) -> Dict:
    """RF do baseline (mesmos hiperparâmetros) e as métricas por janela e por sequência.

    Se ``acertos`` for um dict, recebe os acertos 0/1 por janela (na ordem de
    ``X_te``) e por sequência (na ordem crescente de ``g_te``), para a
    comparação pareada.
    """
    modelo, _ = criar_modelo("rf", seed)
    modelo.fit(X_tr, y_tr)
    proba = np.zeros((len(X_te), n_classes))
    proba[:, modelo.classes_] = modelo.predict_proba(X_te)
    metricas = avaliar(proba, y_te, g_te, n_classes, seed=seed)
    if acertos is not None:
        y_seq, pred_seq, _ = agregar_por_sequencia(proba, y_te, g_te)
        acertos["por_janela"] = (proba.argmax(axis=1) == y_te).astype(np.float64)
        acertos["por_sequencia"] = (pred_seq == y_seq).astype(np.float64)
    return {
        "seed": seed,
        **{visao: {chave: metricas[visao][chave]
                   for chave in ("n", "acuracia", "ic95", "f1_macro")}
           for visao in ("por_janela", "por_sequencia")},
    }


def bootstrap_pareado(acertos_a: np.ndarray, acertos_b: np.ndarray,
                      grupos: Optional[np.ndarray] = None, n_reamostras: int = 2000,
                      seed: int = 0) -> Dict:
    """Diferença de acurácia ``a − b`` medidas nas **mesmas** amostras.

    Reamostra sequências (``grupos``; sem ele, cada amostra é uma sequência) e
    aplica o mesmo sorteio a ``a`` e ``b``. Devolve a diferença observada, o
    IC 95% percentil e a fração de reamostras com diferença ≤ 0.
    """
    acertos_a = np.asarray(acertos_a, dtype=np.float64)
    acertos_b = np.asarray(acertos_b, dtype=np.float64)
    if acertos_a.shape != acertos_b.shape or acertos_a.size == 0:
        raise ValueError("Acertos pareados com formas diferentes ou vazios.")
    if grupos is None:
        grupos = np.arange(len(acertos_a))
    _, inverso = np.unique(grupos, return_inverse=True)
    soma_a = np.bincount(inverso, weights=acertos_a)
    soma_b = np.bincount(inverso, weights=acertos_b)
    contagem = np.bincount(inverso).astype(np.float64)
    sorteio = np.random.default_rng(seed).integers(0, len(contagem),
                                                   size=(n_reamostras, len(contagem)))
    total = contagem[sorteio].sum(axis=1)
    diferencas = (soma_a[sorteio].sum(axis=1) - soma_b[sorteio].sum(axis=1)) / total
    baixo, alto = np.quantile(diferencas, [0.025, 0.975])
    return {
        "diferenca": float((acertos_a.sum() - acertos_b.sum()) / len(acertos_a)),
        "ic95": [float(baixo), float(alto)],
        "fracao_menor_igual_zero": float(np.mean(diferencas <= 0)),
    }


def comparar_pareado(acertos_a: List[Dict], acertos_b: List[Dict], g_te: np.ndarray,
                     seeds: Sequence[int]) -> Dict:
    """Bootstrap pareado ``a − b`` por seed e na média das seeds.

    Por janela, a reamostragem é por sequência (``g_te``). Na média das seeds,
    cada janela/sequência entra com a fração de seeds em que acertou.
    """
    g_te = np.asarray(g_te)
    por_seed = []
    for seed, a, b in zip(seeds, acertos_a, acertos_b):
        por_seed.append({
            "seed": seed,
            "por_janela": bootstrap_pareado(a["por_janela"], b["por_janela"], g_te, seed=seed),
            "por_sequencia": bootstrap_pareado(a["por_sequencia"], b["por_sequencia"], seed=seed),
        })
    media = {
        visao: bootstrap_pareado(np.mean([a[visao] for a in acertos_a], axis=0),
                                 np.mean([b[visao] for b in acertos_b], axis=0),
                                 g_te if visao == "por_janela" else None, seed=0)
        for visao in ("por_janela", "por_sequencia")
    }
    return {"por_seed": por_seed, "media_das_seeds": media}


def resumir(execucoes: List[Dict]) -> Dict:
    """Média e desvio (amostral, ddof=1) da acurácia entre as seeds."""
    resumo = {}
    for visao in ("por_janela", "por_sequencia"):
        valores = np.array([e[visao]["acuracia"] for e in execucoes])
        resumo[visao] = {
            "media": float(valores.mean()),
            "desvio": float(valores.std(ddof=1)) if len(valores) > 1 else 0.0,
        }
    return resumo


def escolher(validacao: List[Dict]) -> Tuple[str, ...]:
    """Maior média por sequência; empate -> maior média por janela -> ordem da lista."""
    melhor = max(
        enumerate(validacao),
        key=lambda par: (par[1]["resumo"]["por_sequencia"]["media"],
                         par[1]["resumo"]["por_janela"]["media"],
                         -par[0]),
    )[1]
    return tuple(melhor["atributos"])


def conferir_validacao(seqs_treino, seqs_teste, sujeitos_validacao) -> None:
    """Validação disjunta do teste, dentro do treino e com um sujeito por ambiente."""
    sujeitos_teste = {s for s, _ in seqs_teste}
    sujeitos_treino = {s for s, _ in seqs_treino}
    validacao = set(sujeitos_validacao)
    if not validacao:
        raise ValueError("Sem sujeitos de validação.")
    no_teste = validacao & (sujeitos_teste | set(SUJEITOS_TESTE_S2))
    if no_teste:
        raise ValueError(f"Sujeitos de validação no teste: {sorted(no_teste)}.")
    if not validacao <= sujeitos_treino:
        raise ValueError(f"Sujeitos de validação fora do treino: {sorted(validacao - sujeitos_treino)}.")
    if sujeitos_treino & sujeitos_teste:
        raise ValueError("Vazamento: sujeito no treino e no teste.")


def selecionar(
    blocos_treino: Blocos, y_tr: np.ndarray, g_tr: np.ndarray, seqs_treino,
    blocos_teste: Blocos, y_te: np.ndarray, g_te: np.ndarray, seqs_teste,
    n_classes: int,
    candidatos: Sequence[Sequence[str]] = CANDIDATOS,
    seeds_validacao: Sequence[int] = range(SEEDS_VALIDACAO),
    seeds_teste: Sequence[int] = range(SEEDS_TESTE),
    sujeitos_validacao: Sequence[str] = SUJEITOS_VALIDACAO_S2,
    referencia: Sequence[str] = REFERENCIA,
    ao_escolher: Optional[Callable[[Dict], None]] = None,
    log: Callable[[str], None] = print,
) -> Dict:
    """Validação -> escolha (``ao_escolher`` é chamado aqui) -> teste.

    ``blocos_*[nome]`` são as janelas de ``seqs_*`` (``g_*`` indexa ``seqs_*``)
    no conjunto ``nome``. A validação só usa ``blocos_treino``.
    """
    conferir_validacao(seqs_treino, seqs_teste, sujeitos_validacao)
    candidatos = [validar_atributos(c) for c in candidatos]
    referencia = validar_atributos(referencia)

    _, seqs_validacao = separar_validacao(seqs_treino, sujeitos_validacao)
    seqs_validacao = set(seqs_validacao)
    seq_validacao = np.array([tuple(seq) in seqs_validacao for seq in seqs_treino])
    mascara = seq_validacao[np.asarray(g_tr)]
    i_fit, i_val = np.flatnonzero(~mascara), np.flatnonzero(mascara)

    validacao = []
    for combinacao in candidatos:
        X_fit, X_val = juntar(blocos_treino, combinacao, i_fit), juntar(blocos_treino, combinacao, i_val)
        execucoes = [treinar_e_medir(X_fit, y_tr[i_fit], X_val, y_tr[i_val], g_tr[i_val],
                                     n_classes, seed) for seed in seeds_validacao]
        item = {"atributos": list(combinacao), "n_atributos": int(X_fit.shape[1]),
                "execucoes": execucoes, "resumo": resumir(execucoes)}
        validacao.append(item)
        r = item["resumo"]
        log(f"    validação {nome_combinacao(combinacao):<28} d={X_fit.shape[1]:5d}  "
            f"janela {100 * r['por_janela']['media']:5.1f} ± {100 * r['por_janela']['desvio']:3.1f}%  "
            f"sequência {100 * r['por_sequencia']['media']:5.1f} ± "
            f"{100 * r['por_sequencia']['desvio']:3.1f}%")

    escolhida = escolher(validacao)
    resultado = {
        "sujeitos_validacao": list(sujeitos_validacao),
        "n_sequencias": {"treino_validacao": int(len(seqs_treino) - seq_validacao.sum()),
                         "validacao": int(seq_validacao.sum()),
                         "treino": len(seqs_treino), "teste": len(seqs_teste)},
        "criterio": "maior média de acurácia por sequência na validação "
                    "(empate: maior média por janela; depois a ordem dos candidatos)",
        "seeds_validacao": list(seeds_validacao),
        "seeds_teste": list(seeds_teste),
        "validacao": validacao,
        "escolhida": list(escolhida),
        "escolha_registrada_em": time.strftime("%Y-%m-%d %H:%M:%S"),
        "teste": None,
    }
    if ao_escolher is not None:
        ao_escolher(resultado)

    teste, acertos = {}, {}
    for combinacao in dict.fromkeys([escolhida, referencia]):
        X_tr, X_te = juntar(blocos_treino, combinacao), juntar(blocos_teste, combinacao)
        acertos[combinacao] = [{} for _ in seeds_teste]
        execucoes = [treinar_e_medir(X_tr, y_tr, X_te, y_te, g_te, n_classes, seed, a)
                     for seed, a in zip(seeds_teste, acertos[combinacao])]
        teste[nome_combinacao(combinacao)] = {
            "atributos": list(combinacao), "n_atributos": int(X_tr.shape[1]),
            "execucoes": execucoes, "resumo": resumir(execucoes)}
    resultado["teste"] = teste
    resultado["comparacao_pareada"] = None if escolhida == referencia else {
        "diferenca": f"{nome_combinacao(escolhida)} − {nome_combinacao(referencia)}",
        **comparar_pareado(acertos[escolhida], acertos[referencia], g_te, list(seeds_teste)),
    }
    return resultado


def _gravar(caminho: Path, dados: Dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")


def _imprimir_teste(resultado: Dict) -> None:
    for nome, item in resultado["teste"].items():
        papel = []
        if item["atributos"] == resultado["escolhida"]:
            papel.append("escolhida")
        if item["atributos"] == list(REFERENCIA):
            papel.append("referência")
        print(f"\n  {nome} ({' e '.join(papel)}, d={item['n_atributos']})")
        for e in item["execucoes"]:
            j, q = e["por_janela"], e["por_sequencia"]
            print(f"    seed {e['seed']}: janela {100 * j['acuracia']:5.1f}% "
                  f"IC95% [{100 * j['ic95'][0]:4.1f}, {100 * j['ic95'][1]:4.1f}]   "
                  f"sequência {100 * q['acuracia']:5.1f}% "
                  f"IC95% [{100 * q['ic95'][0]:4.1f}, {100 * q['ic95'][1]:4.1f}]")
        r = item["resumo"]
        print(f"    média ± desvio: janela {100 * r['por_janela']['media']:5.1f} ± "
              f"{100 * r['por_janela']['desvio']:3.1f}%   sequência "
              f"{100 * r['por_sequencia']['media']:5.1f} ± {100 * r['por_sequencia']['desvio']:3.1f}%")

    comparacao = resultado.get("comparacao_pareada")
    if comparacao is None:
        return
    print(f"\n  Comparação pareada ({comparacao['diferenca']}), bootstrap de sequências de teste:")

    def linha(rotulo, item):
        partes = []
        for visao, nome in (("por_janela", "janela"), ("por_sequencia", "sequência")):
            d = item[visao]
            partes.append(f"{nome} {100 * d['diferenca']:+5.1f} pp "
                          f"IC95% [{100 * d['ic95'][0]:+5.1f}, {100 * d['ic95'][1]:+5.1f}] "
                          f"P(≤0) {d['fracao_menor_igual_zero']:.3f}")
        print(f"    {rotulo:<15} " + "   ".join(partes))

    for item in comparacao["por_seed"]:
        linha(f"seed {item['seed']}:", item)
    linha("média das seeds:", comparacao["media_das_seeds"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Seleção de atributos por validação (ML-03)")
    parser.add_argument("--janela", type=int, default=30)
    parser.add_argument("--passo", type=int, default=None)
    parser.add_argument("--seeds-validacao", type=int, default=SEEDS_VALIDACAO)
    parser.add_argument("--seeds-teste", type=int, default=SEEDS_TESTE)
    parser.add_argument("--raiz", default=None, help="Caminho do MM-Fi (padrão: MMFI_DATASET)")
    parser.add_argument("--cache", default=str(RAIZ_PROJETO / "artifacts" / "cache"))
    parser.add_argument("--cache-atributos",
                        default=str(RAIZ_PROJETO / "artifacts" / "cache_atributos"))
    parser.add_argument("--saida", default=str(RAIZ_PROJETO / "artifacts" / "resultados"
                                               / "selecao_atributos.json"))
    args = parser.parse_args()
    if args.seeds_validacao < 1 or args.seeds_teste < 1:
        print("ERRO: é preciso ao menos 1 seed de validação e 1 de teste.")
        return 1
    try:
        janelas_de_sequencia(N_QUADROS, args.janela, args.passo)
    except ValueError as erro:
        print(f"ERRO: {erro}")
        return 1
    passo = args.passo or args.janela

    from src.cache_csi import abrir_cache, construir_cache
    from src.doppler import abrir_cache_atributos, construir_cache_atributos

    inicio = time.time()
    try:
        print(f"--> Cache do CSI em: {args.cache}")
        construir_cache(args.cache, args.raiz)
        print(f"--> Cache de atributos em: {args.cache_atributos}")
        construir_cache_atributos(args.cache_atributos, args.raiz, args.janela, passo)
        amp, fase, linhas = abrir_cache(args.cache)
        cache_atributos = abrir_cache_atributos(args.cache_atributos)
    except (FileNotFoundError, ValueError) as erro:
        print(f"ERRO: {erro}")
        return 1

    acoes = tuple(sorted(PROTOCOLO_3))
    treino, teste = dividir_s2(acoes)
    print(f"--> S2 / P3: {len(treino)} sequências de treino, {len(teste)} de teste; "
          f"validação = {', '.join(SUJEITOS_VALIDACAO_S2)} "
          f"({', '.join(ambiente_do_sujeito(s) for s in SUJEITOS_VALIDACAO_S2)}), "
          f"retirados do treino")

    conjuntos = list(dict.fromkeys(nome for c in CANDIDATOS + (REFERENCIA,) for nome in c))
    print(f"--> Montando atributos ({', '.join(conjuntos)}; janela={args.janela}, passo={passo})")
    blocos_tr, blocos_te = {}, {}
    for nome in conjuntos:
        blocos_tr[nome], y_tr, g_tr = montar_conjunto(
            amp, fase, linhas, treino, acoes, args.janela, passo,
            atributos=(nome,), cache_atributos=cache_atributos)
        blocos_te[nome], y_te, g_te = montar_conjunto(
            amp, fase, linhas, teste, acoes, args.janela, passo,
            atributos=(nome,), cache_atributos=cache_atributos)

    import scipy
    import sklearn
    cabecalho = {
        "protocolo": "s2",
        "modelo": "rf",
        "hiperparametros": criar_modelo("rf", 0)[1] | {"random_state": "seed"},
        "acoes": list(acoes),
        "janela": args.janela,
        "passo": passo,
        "candidatos": [list(c) for c in CANDIDATOS],
        "referencia": list(REFERENCIA),
        "versoes": {"sklearn": sklearn.__version__, "scipy": scipy.__version__,
                    "numpy": np.__version__, "python": platform.python_version()},
    }
    saida = Path(args.saida)

    def ao_escolher(parcial: Dict) -> None:
        _gravar(saida, {**cabecalho, **parcial, "estado": "escolha feita; teste ainda não avaliado"})
        print(f"\n--> Escolhida pela validação: {nome_combinacao(parcial['escolhida'])} "
              f"(gravada em {saida} antes de avaliar o teste)")

    print(f"--> Validação: RF, {args.seeds_validacao} seeds por candidata")
    resultado = selecionar(
        blocos_tr, y_tr, g_tr, treino, blocos_te, y_te, g_te, teste, len(acoes),
        seeds_validacao=range(args.seeds_validacao), seeds_teste=range(args.seeds_teste),
        ao_escolher=ao_escolher)

    print(f"\n=== Teste S2 (RF treinado no treino S2 completo, {args.seeds_teste} seeds; "
          f"acaso {100 / len(acoes):.1f}%) ===")
    _imprimir_teste(resultado)
    final = {**cabecalho, **resultado, "estado": "completo",
             "acaso": 1.0 / len(acoes),
             "tempo_s": round(time.time() - inicio, 1),
             "data": time.strftime("%Y-%m-%d %H:%M:%S")}
    _gravar(saida, final)
    print(f"\n--> Resultado: {saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
