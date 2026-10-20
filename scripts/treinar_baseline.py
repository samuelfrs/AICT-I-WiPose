"""
Baseline de classificação das ações do MM-Fi (Tarefa c) e matriz de confusão (e).

Random Forest ou SVM sobre média e desvio por antena × subportadora em janelas
de quadros (ver src/atributos.py). Divisão sempre por sequência, sem vazamento
(ver src/protocolos.py). Reporta acurácia por janela e por sequência, com IC 95%
por bootstrap de sequências, contra a linha do acaso.

Uso:
    .venv/Scripts/python.exe scripts/treinar_baseline.py --protocolo s2 --modelo rf
    .venv/Scripts/python.exe scripts/treinar_baseline.py --protocolo s1 --modelo svm --seed 1

Na primeira execução o cache do CSI é construído em artifacts/cache/ (~9 GB em
disco, alguns minutos). As seguintes o reaproveitam.

Saídas, com <sufixo> = <protocolo>_<modelo>[_<acoes>]_j<janela>_p<passo>_seed<seed>:
    artifacts/resultados/baseline_<sufixo>.json
    artifacts/figuras/confusao_<sufixo>.png       matriz por janela
    artifacts/figuras/confusao_<sufixo>_seq.png   matriz por sequência
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

# O console do Windows usa cp1252 por padrão e engasga com acentuação.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROJETO))

from src.atributos import janelas_de_sequencia, montar_conjunto  # noqa: E402
from src.avaliacao import avaliar, salvar_matriz_confusao  # noqa: E402
from src.cache_csi import abrir_cache, construir_cache  # noqa: E402
from src.mmfi import (  # noqa: E402
    N_QUADROS,
    PROTOCOLO_1,
    PROTOCOLO_2,
    PROTOCOLO_3,
)
from src.protocolos import dividir_s1_agrupado, dividir_s2  # noqa: E402

CONJUNTOS_DE_ACOES = {"p1": PROTOCOLO_1, "p2": PROTOCOLO_2, "p3": PROTOCOLO_3}


def criar_modelo(nome: str, seed: int):
    """Classificador; no SVM, precedido de padronização.

    O SVM com kernel RBF depende da escala dos atributos, então vai num Pipeline
    com StandardScaler — o Pipeline garante que o scaler só veja o treino. A
    Random Forest é invariante a escala e dispensa a padronização.

    Devolve ``(modelo, hiperparametros)``: os hiperparâmetros fixados aqui,
    lidos de ``get_params()`` para registro nos resultados.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if nome == "rf":
        classificador = RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=seed
        )
        definidos = ("n_estimators", "n_jobs", "random_state")
        modelo = classificador
    else:
        classificador = SVC(kernel="rbf", C=10.0, gamma="scale",
                            probability=True, random_state=seed)
        definidos = ("kernel", "C", "gamma", "probability", "random_state")
        modelo = make_pipeline(StandardScaler(), classificador)
    parametros = classificador.get_params()
    return modelo, {chave: parametros[chave] for chave in definidos}


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline de classificação de ações (MM-Fi)")
    parser.add_argument("--protocolo", choices=("s1", "s2"), default="s2",
                        help="s1 = split agrupado por sequência; s2 = cross-subject oficial")
    parser.add_argument("--modelo", choices=("rf", "svm"), default="rf")
    parser.add_argument("--acoes", choices=sorted(CONJUNTOS_DE_ACOES), default="p3",
                        help="p1 = diárias, p2 = reabilitação, p3 = as 27 (padrão)")
    parser.add_argument("--janela", type=int, default=30, help=f"quadros por janela (1 a {N_QUADROS})")
    parser.add_argument("--passo", type=int, default=None, help="padrão: igual à janela (sem sobreposição)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--raiz", default=None, help="Caminho do MM-Fi (padrão: MMFI_DATASET)")
    parser.add_argument("--cache", default=str(RAIZ_PROJETO / "artifacts" / "cache"))
    parser.add_argument("--refazer", action="store_true", help="Reconstrói o cache do zero")
    args = parser.parse_args()

    try:
        janelas_de_sequencia(N_QUADROS, args.janela, args.passo)
    except ValueError as erro:
        print(f"ERRO: {erro}")
        return 1

    inicio = time.time()
    print(f"--> Cache do CSI em: {args.cache}")
    try:
        construir_cache(args.cache, args.raiz, refazer=args.refazer)
    except FileNotFoundError as erro:
        print(f"ERRO: {erro}")
        return 1
    amp, fase, linhas = abrir_cache(args.cache)

    acoes = tuple(sorted(CONJUNTOS_DE_ACOES[args.acoes]))
    if args.protocolo == "s1":
        treino, teste = dividir_s1_agrupado(acoes, seed=args.seed)
    else:
        treino, teste = dividir_s2(acoes)
    assert not set(treino) & set(teste), "vazamento: sequência em treino e teste"
    print(f"--> Protocolo {args.protocolo.upper()} / {args.acoes.upper()} "
          f"({len(acoes)} ações): {len(treino)} sequências de treino, "
          f"{len(teste)} de teste")

    print(f"--> Extraindo atributos (janela={args.janela}, passo={args.passo or args.janela})")
    X_tr, y_tr, _ = montar_conjunto(amp, fase, linhas, treino, acoes, args.janela, args.passo)
    X_te, y_te, g_te = montar_conjunto(amp, fase, linhas, teste, acoes, args.janela, args.passo)
    print(f"    treino {X_tr.shape}, teste {X_te.shape}")

    print(f"--> Treinando {args.modelo.upper()} (seed={args.seed})")
    t0 = time.time()
    modelo, hiperparametros = criar_modelo(args.modelo, args.seed)
    modelo.fit(X_tr, y_tr)
    tempo_treino = time.time() - t0

    # predict_proba só tem colunas das classes vistas no treino; realinha às 27.
    proba_parcial = modelo.predict_proba(X_te)
    proba = np.zeros((len(X_te), len(acoes)))
    proba[:, modelo.classes_] = proba_parcial

    metricas = avaliar(proba, y_te, g_te, len(acoes), seed=args.seed)
    tempo_total = time.time() - inicio

    acaso = metricas["acaso"]
    print("\n=== Resultado ===")
    for visao, nome in (("por_janela", "janela"), ("por_sequencia", "sequência")):
        m = metricas[visao]
        baixo, alto = m["ic95"]
        print(f"  acurácia por {nome:<9}: {100 * m['acuracia']:5.1f}%  "
              f"IC95% [{100 * baixo:5.1f}, {100 * alto:5.1f}]  "
              f"F1 macro {m['f1_macro']:.3f}  (n={m['n']})")
    print(f"  acaso (1/{len(acoes)})        : {100 * acaso:5.1f}%")
    print(f"  tempo: treino {tempo_treino:.0f} s, total {tempo_total:.0f} s")

    import sklearn

    passo = args.passo or args.janela
    # Um sufixo só para JSON e figuras: execuções com outra janela, passo ou
    # seed não podem sobrescrever umas às outras.
    sufixo = (f"{args.protocolo}_{args.modelo}"
              + ("" if args.acoes == "p3" else f"_{args.acoes}")
              + f"_j{args.janela}_p{passo}_seed{args.seed}")
    resultado = {
        "protocolo": args.protocolo,
        "modelo": args.modelo,
        "acoes": list(acoes),
        "janela": args.janela,
        "passo": passo,
        "seed": args.seed,
        "hiperparametros": hiperparametros,
        "versoes": {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
        "n_sequencias": {"treino": len(treino), "teste": len(teste)},
        "n_janelas": {"treino": int(len(X_tr)), "teste": int(len(X_te))},
        "sequencias_teste": [list(s) for s in teste],
        "metricas": metricas,
        "tempo_s": {"treino": round(tempo_treino, 1), "total": round(tempo_total, 1)},
        "data": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    pasta_resultados = RAIZ_PROJETO / "artifacts" / "resultados"
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    arquivo_json = pasta_resultados / f"baseline_{sufixo}.json"
    arquivo_json.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n--> Métricas: {arquivo_json}")
    pasta_figuras = RAIZ_PROJETO / "artifacts" / "figuras"
    for visao, nome, final in (("por_janela", "por janela", ""),
                               ("por_sequencia", "por sequência", "_seq")):
        m = metricas[visao]
        figura = salvar_matriz_confusao(
            np.array(m["confusao"]), acoes,
            pasta_figuras / f"confusao_{sufixo}{final}.png",
            titulo=(f"{args.modelo.upper()} — protocolo {args.protocolo.upper()} — {nome} "
                    f"(acurácia {100 * m['acuracia']:.1f}%, acaso {100 * acaso:.1f}%)"),
        )
        print(f"--> Matriz de confusão ({nome}): {figura}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
