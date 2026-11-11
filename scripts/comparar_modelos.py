"""
Tabela de comparação entre os modelos da tarefa (c) no teste S2 (ML-04).

Lê os JSON já gravados em artifacts/resultados/ e monta uma linha por
abordagem, com média ± desvio (amostral) entre as execuções:

==================  ===============================================  =====================
modelo              fonte                                            unidade
==================  ===============================================  =====================
RF base             selecao_atributos.json → teste["base"]           janela de 3 s
RF + Doppler        selecao_atributos.json → teste[escolhida]        janela de 3 s
Rede temporal       rede_temporal_s2_seed<seed>.json                 janela de 3 s
Rede neural v1           rede_neural_v1_exec<i>.json                           quadro de 100 ms
Rede neural v2           rede_neural_v2_exec<i>.json                           janela de 3 s
==================  ===============================================  =====================

As execuções são descobertas pelo nome (qualquer ``<seed>``/``<i>``); arquivos
da rede temporal com sufixo de configuração (``_j<janela>_e<epocas>``) ficam
de fora — só o padrão entra na tabela.

"Por sequência" é sempre a média das probabilidades das janelas da sequência.
A rede neural v1 classifica quadros isolados e não mede por sequência: a coluna
fica vazia em vez de receber um número de outra unidade.

O "±" é só a variação entre execuções (ruído do treino). Não é um teste de
diferença entre modelos: os ICs por execução (só nos JSON do RF e da rede
temporal) reamostram sequências e ignoram que elas se agrupam por sujeito (8
no teste), então são otimistas; o único teste pareado é RF + Doppler × RF base
(ML-03).

Resultado ausente, ilegível ou fora do formato: a linha sai como "não
executado" (ou sem aquela execução), com aviso, sem traceback.

Uso:
    .venv/Scripts/python.exe scripts/comparar_modelos.py

Saída: tabela Markdown no terminal e artifacts/resultados/comparacao_modelos.json
(com a mesma tabela em ``tabela_markdown``).
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
N_CLASSES = 27

# Unidades conferidas nos logs das execuções: a v1 tem 3.240 amostras de teste
# = 216 sequências × 15 quadros (stride 20); a v2, 1.944 janelas = 216 × 9.
UNIDADE_JANELA = "janela de 3 s (30 quadros, 9 por sequência)"
UNIDADE_QUADRO_V1 = "quadro de 100 ms (1 a cada 20, 15 por sequência)"

EXECUCOES_PLANEJADAS = 3
# Execuções esperadas por linha. A v1 ficou com 1 por decisão do usuário
# (tempo); faltar uma execução esperada gera aviso, as cortadas não.
EXECUCOES_ESPERADAS = {"rede_temporal": 3, "rede_neural_v1": 1, "rede_neural_v2": 3}

SEM_OLHAR_REDE_NEURAL = ("parcial — reporta a última época (nº fixado antes), mas não há "
                    "validação: o script mostra o teste a cada época e o nº de "
                    "épocas/hiperparâmetros foi definido pelo autor")

LEITURA = (
    "± = desvio entre execuções (só o ruído do treino), não um teste de diferença entre modelos",
    "IC 95% por execução só existe nos JSON do RF e da rede temporal (os da rede neural v1/v2 não "
    "gravam IC); ele reamostra sequências e ignora que elas se agrupam por sujeito "
    "(8 no teste), então é otimista",
    "o único teste pareado é RF + Doppler × RF base (ML-03); entre os demais, a ordem é "
    "só o maior número observado, não uma classificação provada",
    "a rede temporal treina com 28 sujeitos (4 ficam para a validação, sem retreino); "
    "os RF e as redes neurais v1 e v2 treinam com os 32 — parte da diferença pode ser "
    "quantidade de dados",
)

Avisar = Callable[[str], None]


class FormatoInesperado(Exception):
    pass


def _ler(caminho: Path, avisar: Avisar, obrigatorio: bool = True) -> Optional[Dict]:
    if not caminho.is_file():
        if obrigatorio:
            avisar(f"resultado ausente: {caminho.name}")
        return None
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, ValueError) as erro:
        avisar(f"resultado ilegível: {caminho.name} ({erro})")
        return None
    if not isinstance(dados, dict):
        avisar(f"resultado fora do formato: {caminho.name} (esperado um objeto JSON, "
               f"veio {type(dados).__name__})")
        return None
    return dados


def _pegar(dados, *chaves):
    """``dados[c1][c2]...``; ausente ou de tipo errado -> ``FormatoInesperado``."""
    atual = dados
    for chave in chaves:
        try:
            atual = atual[chave]
        except (KeyError, IndexError, TypeError):
            raise FormatoInesperado("/".join(str(c) for c in chaves)) from None
    return atual


def _numero(valor) -> float:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        raise FormatoInesperado(f"valor não numérico: {valor!r}")
    return float(valor)


def descobrir(pasta: Path, padrao: str) -> List[Tuple[int, Path]]:
    """Arquivos de ``pasta`` cujo nome casa ``padrao`` (um grupo numérico), ordenados."""
    regex = re.compile(padrao)
    achados = []
    for caminho in pasta.glob("*.json") if pasta.is_dir() else []:
        casou = regex.fullmatch(caminho.name)
        if casou:
            achados.append((int(casou.group(1)), caminho))
    return sorted(achados)


def resumir(valores: Sequence[float]) -> Optional[Dict]:
    """Média e desvio amostral (ddof=1); com uma execução, desvio ``None``."""
    if not valores:
        return None
    arr = np.asarray(valores, dtype=np.float64)
    return {"media": float(arr.mean()),
            "desvio": float(arr.std(ddof=1)) if len(arr) > 1 else None,
            "n": int(len(arr)),
            "valores": [float(v) for v in arr]}


def conferir_consistencia(modelo: str, assinaturas: Sequence[Tuple[str, Dict]],
                          avisar: Avisar) -> None:
    """Avisa se execuções da mesma linha diferem em commit/comando/configuração."""
    if len(assinaturas) < 2:
        return
    referencia_nome, referencia = assinaturas[0]
    for nome, assinatura in assinaturas[1:]:
        diferentes = sorted(k for k in set(referencia) | set(assinatura)
                            if referencia.get(k) != assinatura.get(k))
        if diferentes:
            avisar(f"{modelo}: {nome} difere de {referencia_nome} em {', '.join(diferentes)}")


def _linha(modelo: str, unidade: str, janela: List[float], sequencia: List[float],
           n_execucoes: int, escolha: str, sem_olhar_teste: str,
           seed_controlada: Optional[bool], fonte: str, detalhes: Optional[Dict] = None,
           notas: Sequence[str] = (), mede_sequencia: bool = True) -> Dict:
    return {
        "modelo": modelo,
        "unidade": unidade,
        "executado": n_execucoes > 0,
        "n_execucoes": n_execucoes,
        "n_por_janela": len(janela),
        "n_por_sequencia": len(sequencia) if mede_sequencia else None,
        "mede_sequencia": mede_sequencia,
        "por_janela": resumir(janela),
        "por_sequencia": resumir(sequencia),
        "como_e_escolhido": escolha,
        "escolhido_sem_olhar_o_teste": sem_olhar_teste,
        "seed_controlada": seed_controlada,
        "fonte": fonte,
        "notas": list(notas),
        **({"detalhes": detalhes} if detalhes else {}),
    }


def _conferir_quantidade(chave: str, encontradas: int, avisar: Avisar, fonte: str) -> None:
    esperadas = EXECUCOES_ESPERADAS[chave]
    if encontradas < esperadas:
        avisar(f"{fonte}: {encontradas} execução(ões) válidas, esperadas {esperadas}")


def linhas_rf(pasta: Path, avisar: Avisar) -> List[Dict]:
    """RF base e RF + Doppler, do ML-03 (5 seeds no teste S2)."""
    fonte = "selecao_atributos.json"
    dados = _ler(pasta / fonte, avisar)
    teste = (dados or {}).get("teste")
    if dados is not None and not isinstance(teste, dict):
        avisar(f"{fonte} sem a parte de teste")
        teste = {}
    teste = teste or {}
    escolhida = (dados or {}).get("escolhida")
    if isinstance(escolhida, str):
        escolhida = [p.strip() for p in escolhida.split(",") if p.strip()]
    if dados is not None and not (isinstance(escolhida, list) and escolhida):
        avisar(f"{fonte} sem 'escolhida': a linha RF + Doppler fica sem dados")
        escolhida = None
    chave_escolhida = ",".join(escolhida) if escolhida else None

    linhas = []
    for nome, chave, escolha, sem_olhar in (
        ("RF base (ML-01)", "base",
         "hiperparâmetros fixos do ML-01; sem épocas", "sim"),
        ("RF + Doppler (ML-03)", chave_escolhida,
         f"conjunto de atributos ({chave_escolhida or '?'}) escolhido na validação "
         "(S04/S14/S24/S34)",
         "sim, pela validação (as 5 candidatas vieram de um protótipo que olhou o teste)"),
    ):
        janela, sequencia, seeds = [], [], []
        execucoes = teste.get(chave) if chave else None
        if dados is not None and teste and chave and not isinstance(execucoes, dict):
            avisar(f"{fonte} sem execuções de teste para {chave!r}")
        for k, e in enumerate((execucoes or {}).get("execucoes") or []
                              if isinstance(execucoes, dict) else []):
            try:
                j = _numero(_pegar(e, "por_janela", "acuracia"))
                q = _numero(_pegar(e, "por_sequencia", "acuracia"))
            except FormatoInesperado as erro:
                avisar(f"{fonte}: execução {k} de {chave!r} fora do formato ({erro})")
                continue
            janela.append(j)
            sequencia.append(q)
            seeds.append(e.get("seed") if isinstance(e, dict) else None)
        linhas.append(_linha(
            nome, UNIDADE_JANELA, janela, sequencia, len(janela), escolha, sem_olhar, True,
            fonte, {"seeds": seeds, "atributos": chave} if janela else None))
    return linhas


def linha_rede_temporal(pasta: Path, avisar: Avisar) -> Dict:
    janela, sequencia, epocas, usados, assinaturas = [], [], [], [], []
    for seed, caminho in descobrir(pasta, r"rede_temporal_s2_seed(\d+)\.json"):
        dados = _ler(caminho, avisar)
        if dados is None:
            continue
        if dados.get("estado") != "completo":
            avisar(f"{caminho.name} incompleto (estado {dados.get('estado')!r})")
            continue
        try:
            j = _numero(_pegar(dados, "teste", "por_janela", "acuracia"))
            q = _numero(_pegar(dados, "teste", "por_sequencia", "acuracia"))
        except FormatoInesperado as erro:
            avisar(f"{caminho.name} fora do formato ({erro})")
            continue
        janela.append(j)
        sequencia.append(q)
        epocas.append(dados.get("epoca_escolhida"))
        usados.append(seed)
        hp = dados.get("hiperparametros") if isinstance(dados.get("hiperparametros"), dict) else {}
        assinaturas.append((caminho.name, {
            "epocas": hp.get("epocas"), "janela": hp.get("janela"),
            "protocolo": dados.get("protocolo"), "n_parametros": dados.get("n_parametros")}))
    nome = "Rede temporal (ML-04)"
    conferir_consistencia(nome, assinaturas, avisar)
    _conferir_quantidade("rede_temporal", len(janela), avisar, "rede_temporal_s2_seed<seed>.json")
    return _linha(
        nome, UNIDADE_JANELA, janela, sequencia, len(janela),
        "época escolhida pela acurácia por sequência na validação (S04/S14/S24/S34); "
        "treina com 28 sujeitos, sem retreino com os 32; teste avaliado uma vez",
        "sim", True, "rede_temporal_s2_seed<seed>.json",
        {"seeds": usados, "epocas_escolhidas": epocas} if usados else None)


def _sem_seed(comando: Optional[str]) -> Optional[str]:
    return re.sub(r"\s--seed\s+\S+", "", comando) if isinstance(comando, str) else comando


def linha_rede_neural(pasta: Path, versao: int, avisar: Avisar) -> Dict:
    janela, sequencia, assinaturas, tempos = [], [], [], []
    n_execucoes, seed_controlada, contorno, epocas = 0, None, None, None
    commits, comandos = set(), set()
    for _, caminho in descobrir(pasta, rf"rede_neural_v{versao}_exec(\d+)\.json"):
        dados = _ler(caminho, avisar)
        if dados is None:
            continue
        origem = dados.get("origem") if isinstance(dados.get("origem"), dict) else {}
        try:
            if versao == 1:
                j, q = _numero(_pegar(dados, "acuracia_final")), None
            else:
                j = _numero(_pegar(dados, "acuracia_final_janela"))
                q = dados.get("acuracia_final_sequencia")
                if q is None:
                    avisar(f"{caminho.name} sem 'acuracia_final_sequencia': entra só por janela")
                else:
                    q = _numero(q)
        except FormatoInesperado as erro:
            avisar(f"{caminho.name} fora do formato ({erro})")
            continue
        n_execucoes += 1
        janela.append(j)
        if q is not None:
            sequencia.append(q)
        commits.add(origem.get("commit"))
        comandos.add(origem.get("comando"))
        seed_controlada = origem.get("seed_controlada", seed_controlada)
        contorno = origem.get("contorno_execucao") or contorno
        epocas = dados.get("epocas", epocas)
        if isinstance(origem.get("tempo_total_s"), (int, float)):
            tempos.append(float(origem["tempo_total_s"]))
        assinaturas.append((caminho.name, {
            "commit": origem.get("commit"), "comando": _sem_seed(origem.get("comando")),
            "epocas": dados.get("epocas"), "split": dados.get("split"),
            "protocolo": dados.get("protocolo")}))

    chave = f"rede_neural_v{versao}"
    ep = f" ({epocas})" if epocas else ""
    if versao == 1:
        modelo, unidade = "Rede neural v1 — CNN multitarefa, quadro de 100 ms (main, 98a1f49)", UNIDADE_QUADRO_V1
        escolha = f"última época{ep}, fixa; mede só por quadro"
    else:
        modelo, unidade = "Rede neural v2 — MLP sobre atributos de 3 s (rede-neural-sml, 68d96a9)", UNIDADE_JANELA
        escolha = f"última época{ep}, fixa"
    conferir_consistencia(modelo, assinaturas, avisar)
    _conferir_quantidade(chave, n_execucoes, avisar, f"rede_neural_v{versao}_exec<i>.json")

    notas = []
    if versao == 1 and 0 < n_execucoes < EXECUCOES_PLANEJADAS:
        tempo = (f" e levou ~{np.mean(tempos) / 60:.0f} min (origem.tempo_total_s)"
                 if tempos else "")
        notas.append(
            f"{n_execucoes} execução(ões) em vez de {EXECUCOES_PLANEJADAS}, por decisão do "
            f"usuário (tempo): a execução dividiu a CPU com a rede temporal{tempo}; sem "
            "controle de seed, execuções extras mediriam só o ruído do treino")
    detalhes = None
    if n_execucoes:
        detalhes = {"commits": sorted(c for c in commits if c),
                    "comandos": sorted(c for c in comandos if c),
                    "contorno_execucao": contorno,
                    "tempo_total_s": tempos}
    return _linha(modelo, unidade, janela, sequencia, n_execucoes, escolha,
                  SEM_OLHAR_REDE_NEURAL, seed_controlada, f"rede_neural_v{versao}_exec<i>.json",
                  detalhes, notas, mede_sequencia=versao != 1)


def comparar(pasta: Path, avisar: Avisar = print) -> Dict:
    """Todas as linhas, na ordem da tabela, mais o acaso."""
    pasta = Path(pasta)
    linhas = linhas_rf(pasta, avisar) + [
        linha_rede_temporal(pasta, avisar),
        linha_rede_neural(pasta, 1, avisar),
        linha_rede_neural(pasta, 2, avisar),
    ]
    return {"teste": "S2 oficial (S05, S10, …, S40; 216 sequências, 27 ações)",
            "acaso": 1.0 / N_CLASSES, "leitura": list(LEITURA), "linhas": linhas}


def _pct(resumo: Optional[Dict]) -> str:
    if resumo is None:
        return "—"
    if resumo["desvio"] is None:          # uma execução: não há desvio a mostrar
        return f"{100 * resumo['media']:.1f}"
    return f"{100 * resumo['media']:.1f} ± {100 * resumo['desvio']:.1f}"


def _celula_execucoes(l: Dict) -> str:
    texto = str(l["n_execucoes"])
    if l["mede_sequencia"] and l["n_por_sequencia"] != l["n_por_janela"]:
        texto = f"{l['n_por_janela']} (janela) / {l['n_por_sequencia']} (sequência)"
    return texto


def tabela_markdown(comparacao: Dict) -> str:
    cabecalho = ("| Modelo | Unidade | Acurácia por janela/quadro (%) | Acurácia por sequência (%) "
                 "| Execuções | Seed controlada | Como o número é escolhido | Sem olhar o teste? |")
    linhas = [cabecalho, "|" + "---|" * 8]
    notas: List[str] = []
    for l in comparacao["linhas"]:
        if not l["executado"]:
            linhas.append(f"| {l['modelo']} | {l['unidade']} | não executado | não executado "
                          f"| 0 | — | {l['como_e_escolhido']} | {l['escolhido_sem_olhar_o_teste']} |")
            continue
        seed = {True: "sim", False: "não", None: "—"}[l["seed_controlada"]]
        execucoes = _celula_execucoes(l)
        for nota in l.get("notas", []):
            notas.append(nota)
            execucoes += f" (nota {len(notas)})"
        linhas.append(f"| {l['modelo']} | {l['unidade']} | {_pct(l['por_janela'])} "
                      f"| {_pct(l['por_sequencia'])} | {execucoes} | {seed} "
                      f"| {l['como_e_escolhido']} | {l['escolhido_sem_olhar_o_teste']} |")
    acaso = f"{100 * comparacao['acaso']:.1f}"
    linhas.append(f"| Acaso (1/{N_CLASSES}) | — | {acaso} | {acaso} | — | — | — | — |")
    if notas:
        linhas.append("")
        linhas += [f"Nota {i}: {texto}." for i, texto in enumerate(notas, 1)]
    linhas.append("")
    linhas += [f"Leitura: {texto}." for texto in comparacao.get("leitura", [])]
    return "\n".join(linhas)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Tabela de comparação dos modelos no S2 (ML-04)")
    parser.add_argument("--resultados", default=str(RAIZ_PROJETO / "artifacts" / "resultados"))
    parser.add_argument("--saida", default=None,
                        help="JSON de saída (padrão: <resultados>/comparacao_modelos.json)")
    args = parser.parse_args(argv)
    pasta = Path(args.resultados)
    saida = Path(args.saida) if args.saida else pasta / "comparacao_modelos.json"

    avisos: List[str] = []

    def avisar(texto: str) -> None:
        avisos.append(texto)
        print(f"AVISO: {texto}")

    comparacao = comparar(pasta, avisar)
    tabela = tabela_markdown(comparacao)
    print(f"\nTeste: {comparacao['teste']}; média ± desvio entre execuções.\n")
    print(tabela)
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps({**comparacao, "avisos": avisos, "tabela_markdown": tabela,
                                 "data": time.strftime("%Y-%m-%d %H:%M:%S")},
                                ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n--> Comparação: {saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
