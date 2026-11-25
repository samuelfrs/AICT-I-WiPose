"""Gera as figuras de resultados para a apresentação e o relatório.

Lê só o que já foi medido (JSON em ``artifacts/resultados/``) — não treina nada.
As duas figuras do sinal leem o dataset (fase) e o cache de atributos do ML-03 (Doppler).

Saída em ``artifacts/figuras/``:

- ``resultado_1_comparacao_modelos.png``  — acerto de cada modelo no S2 (ML-04)
- ``resultado_2_assinatura_pessoa.png``   — normalização × S1/S2 × identificar a pessoa (ML-02)
- ``resultado_3_acerto_por_acao.png``     — acerto por ação do RF + Doppler (ML-03)
- ``sinal_1_fase_sanitizada.png``          — fase crua → sem rampa (tarefa d)
- ``sinal_2_doppler.png``                  — "ritmo" do sinal em quatro ações (ML-03; lê o cache de atributos)
- ``resultado_4_cdf_por_acao.png``         — CDF da probabilidade da ação certa, por ação
  (lê ``artifacts/resultados/probabilidades/*.npz``; grava também ``resumo_cdf_por_acao.json`` lá)

Uso::

    python scripts/gerar_figuras_resultados.py              # todas
    python scripts/gerar_figuras_resultados.py --sem-sinal  # só as de resultados (sem dataset)
    python scripts/gerar_figuras_resultados.py --apenas-cdf # só a CDF por ação
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.mmfi import ACOES, FRONTEIRA_20MHZ  # noqa: E402

RESULTADOS = RAIZ / "artifacts" / "resultados"
FIGURAS = RAIZ / "artifacts" / "figuras"

# Paleta categórica fixa (slot 1 azul, 2 laranja, 3 verde-água) e tinta de texto.
AZUL, LARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_2, GRADE, FUNDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ACASO_ACAO = 1 / 27
ACASO_PESSOA = 1 / 40

plt.rcParams.update({
    "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
    "font.family": "DejaVu Sans", "font.size": 10,
    "text.color": TINTA, "axes.labelcolor": TINTA_2,
    "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "axes.edgecolor": GRADE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
})


def _ler(nome: str) -> dict:
    with open(RESULTADOS / nome, encoding="utf-8") as f:
        return json.load(f)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}".replace(".", ",") + "%"


def _linha_acaso(ax, valor: float, rotulo: str, horizontal: bool = True) -> None:
    """Linha tracejada do acaso, com o rótulo junto (não na legenda)."""
    if horizontal:
        ax.axvline(valor, color=TINTA_2, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.annotate(rotulo, (valor, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -2), textcoords="offset points",
                    ha="left", va="top", fontsize=8.5, color=TINTA_2)
    else:  # rótulo vai para a legenda, para não cobrir as barras
        ax.axhline(valor, color=TINTA_2, lw=1.2, ls=(0, (4, 3)), zorder=1, label=rotulo)


def _salvar(fig, nome: str) -> Path:
    destino = FIGURAS / nome
    fig.savefig(destino, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"--> {destino.relative_to(RAIZ) if destino.is_relative_to(RAIZ) else destino}")
    return destino


# ---------------------------------------------------------------------------
# 1. Comparação dos modelos (ML-04)
# ---------------------------------------------------------------------------

NOMES_CURTOS = {
    "RF + Doppler (ML-03)": "Random Forest + Doppler",
    "RF base (ML-01)": "Random Forest base",
    "Rede temporal (ML-04)": "Rede temporal (CNN + GRU)",
    "Rede neural v2 — MLP sobre atributos de 3 s (rede-neural-sml, 68d96a9)": "Rede neural v2 (MLP sobre atributos)",
    "Rede neural v1 — CNN multitarefa, quadro de 100 ms (main, 98a1f49)": "Rede neural v1 (CNN multitarefa)",
}


def figura_comparacao() -> Path:
    comp = _ler("comparacao_modelos.json")
    linhas = [l for l in comp["linhas"] if l.get("executado") and l["modelo"] in NOMES_CURTOS]

    def chave(l):
        seq = l.get("por_sequencia") or {}
        return seq.get("media") if seq.get("media") is not None else -1
    linhas.sort(key=chave)  # de baixo para cima no gráfico horizontal

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    y = np.arange(len(linhas))
    h = 0.36
    for i, l in enumerate(linhas):
        for desloc, campo, cor in ((+h / 2, "por_janela", AZUL), (-h / 2, "por_sequencia", LARANJA)):
            m = l.get(campo) or {}
            if m.get("media") is None:
                if campo == "por_sequencia":
                    ax.text(ACASO_ACAO + 0.006, i + desloc, "não mede por gravação", va="center",
                            fontsize=8.5, color=TINTA_2, style="italic")
                continue
            ax.barh(i + desloc, m["media"], height=h - 0.04, color=cor, zorder=2)
            if m.get("desvio"):
                ax.errorbar(m["media"], i + desloc, xerr=m["desvio"], fmt="none",
                            ecolor=TINTA, elinewidth=1, capsize=2.5, zorder=3)
            fim = m["media"] + (m.get("desvio") or 0)
            ax.text(fim + 0.004, i + desloc, _pct(m["media"]), va="center",
                    fontsize=8.5, color=TINTA)

    rotulos = []
    for l in linhas:
        n = l.get("n_execucoes")
        unidade = "olha 100 ms" if "100 ms" in l["unidade"] else "olha 3 s"
        rotulos.append(f"{NOMES_CURTOS[l['modelo']]}\n{unidade} · {n} execuç{'ão' if n == 1 else 'ões'}")
    ax.set_yticks(y, rotulos, fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.30)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.grid(axis="x", color=GRADE, lw=0.8, zorder=0)
    _linha_acaso(ax, ACASO_ACAO, "chute: 3,7%")
    ax.set_xlabel("acerto no teste com 8 pessoas novas (S2, 27 ações)")
    ax.set_title("Qual modelo reconhece melhor a ação de pessoas novas?")
    ax.legend(handles=[matplotlib.patches.Patch(color=AZUL, label="por pedaço de 3 s (ou quadro)"),
                       matplotlib.patches.Patch(color=LARANJA, label="por gravação inteira (~30 s)")],
              loc="lower right", frameon=False, fontsize=8.5)
    fig.text(0.0, -0.03,
             "Média ± desvio entre execuções (seeds) — o ± mede só a sorte do treino, não a incerteza "
             "de ter 8 pessoas no teste.\nA rede temporal treina com 28 pessoas (4 ficam para a validação); "
             "as demais com 32. Fonte: artifacts/resultados/comparacao_modelos.json",
             fontsize=7.5, color=TINTA_2, ha="left", va="top")
    return _salvar(fig, "resultado_1_comparacao_modelos.png")


# ---------------------------------------------------------------------------
# 2. A assinatura da pessoa (ML-02)
# ---------------------------------------------------------------------------

def figura_assinatura() -> Path:
    modos = [("nenhuma", "", "nenhuma\n(padrão)"), ("media", "_nmedia", "média"),
             ("zscore", "_nzscore", "z-score")]
    s2, s1, pessoa = [], [], []
    for _, suf, _ in modos:
        a2 = _ler(f"baseline_s2_rf_j30_p30_seed0{suf}.json")
        a1 = _ler(f"baseline_s1_rf_j30_p30_seed0{suf}.json")
        s2.append(a2["metricas"]["por_janela"]["acuracia"])
        s1.append(a1["metricas"]["por_janela"]["acuracia"])
        pessoa.append(a2["identificacao_sujeito"]["acuracia"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(modos))
    w = 0.36
    for desloc, vals, cor, rot in ((-w / 2, s2, AZUL, "pessoas novas (S2)"),
                                   (+w / 2, s1, LARANJA, "pessoas conhecidas (S1)")):
        ax1.bar(x + desloc, vals, width=w - 0.03, color=cor, label=rot, zorder=2)
        for xi, v in zip(x + desloc, vals):
            ax1.text(xi, v + 0.003, _pct(v), ha="center", va="bottom", fontsize=8.5)
    ax1.set_xticks(x, [m[2] for m in modos])
    ax1.set_ylim(0, 0.20)
    ax1.set_title("Acerto da ação")
    ax1.set_xlabel("normalização de cada gravação")
    _linha_acaso(ax1, ACASO_ACAO, "chute: 3,7%", horizontal=False)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper right")

    ax2.bar(x, pessoa, width=0.5, color=AQUA, zorder=2)
    for xi, v in zip(x, pessoa):
        ax2.text(xi, v + 0.01, _pct(v), ha="center", va="bottom", fontsize=8.5)
    ax2.set_xticks(x, [m[2] for m in modos])
    ax2.set_ylim(0, 0.9)
    ax2.set_title("Quanto o sinal identifica a pessoa")
    ax2.set_xlabel("normalização de cada gravação")
    _linha_acaso(ax2, ACASO_PESSOA, "chute: 2,5%", horizontal=False)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper right")

    ax1.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.20))
    for ax in (ax1, ax2):
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
        ax.grid(axis="y", color=GRADE, lw=0.8, zorder=0)
        ax.tick_params(axis="x", length=0)
    fig.suptitle("O sinal reconhece quem está na sala melhor do que o que a pessoa faz",
                 x=0.0, ha="left", fontweight="bold", fontsize=12.5, y=1.02)
    fig.text(0.0, -0.04,
             "Random Forest, seed 0, acerto por pedaço de 3 s. Sem normalizar, o sinal identifica a pessoa em 78% "
             "e o S1 fica pior que o S2: o modelo 'cola pela pessoa'.\nCom z-score a assinatura some (17%) e o S1 "
             "sobe — mas o S2 não melhora. Fonte: artifacts/resultados/baseline_s{1,2}_rf_j30_p30_seed0*.json",
             fontsize=7.5, color=TINTA_2, ha="left", va="top")
    fig.tight_layout()
    return _salvar(fig, "resultado_2_assinatura_pessoa.png")


# ---------------------------------------------------------------------------
# 3. Acerto por ação do melhor modelo (ML-03, seed 0)
# ---------------------------------------------------------------------------

def figura_por_acao() -> Path:
    dados = _ler("baseline_s2_rf_j30_p30_seed0_abase-dop_amp-dop_cm.json")
    conf = np.asarray(dados["metricas"]["por_janela"]["confusao"], dtype=float)
    acoes = dados.get("acoes") or sorted(ACOES)
    if isinstance(acoes, str) or len(acoes) != len(conf):
        acoes = sorted(ACOES)
    acerto = np.diag(conf) / conf.sum(axis=1)
    mais_chutada = conf.sum(axis=0).argmax()
    ordem = np.argsort(acerto)

    fig, ax = plt.subplots(figsize=(8.6, 8.4))
    y = np.arange(len(ordem))
    ax.barh(y, acerto[ordem], height=0.72, color=AZUL, zorder=2)
    for yi, i in zip(y, ordem):
        ax.text(acerto[i] + 0.006, yi, _pct(acerto[i]), va="center", fontsize=8, color=TINTA)
    ax.set_yticks(y, [f"{acoes[i]}  {ACOES[acoes[i]]}" for i in ordem], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(0.75, acerto.max() + 0.08))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.grid(axis="x", color=GRADE, lw=0.8, zorder=0)
    _linha_acaso(ax, ACASO_ACAO, "chute: 3,7%")
    ax.set_xlabel("acerto por pedaço de 3 s (72 pedaços por ação: 8 pessoas × 9)")
    ax.set_title("Random Forest + Doppler: o que ele acerta e o que erra (pessoas novas)")
    total = conf.sum()
    fig.text(0.0, -0.01,
             f"Seed 0, teste S2. Acerto geral {_pct(np.trace(conf) / total)}. A ação mais chutada pelo modelo é "
             f"{acoes[mais_chutada]} – {ACOES[acoes[mais_chutada]]}, com "
             f"{_pct(conf[:, mais_chutada].sum() / total)} dos palpites.\nCom só 8 pessoas, cada barra é incerta. "
             "Matriz completa: artifacts/figuras/confusao_s2_rf_j30_p30_seed0_abase-dop_amp-dop_cm.png",
             fontsize=7.5, color=TINTA_2, ha="left", va="top")
    return _salvar(fig, "resultado_3_acerto_por_acao.png")


# ---------------------------------------------------------------------------
# 4. Sinal: fase crua → sanitizada (tarefa d)
# ---------------------------------------------------------------------------

def figura_fase(sujeito: str = "S05", acao: str = "A12", quadro: int = 100) -> Path:
    from src.mmfi import carregar_csi_sequencia, sanitizar_fase

    _, fase = carregar_csi_sequencia(sujeito, acao)
    crua = fase[quadro]                       # (3, 114, 10)
    limpa = sanitizar_fase(crua)
    sub = np.arange(crua.shape[1])
    antena, amostra = 0, 0

    desenrolada = np.unwrap(crua[antena, :, amostra])
    retas = np.empty_like(desenrolada)
    for ini, fim in ((0, FRONTEIRA_20MHZ), (FRONTEIRA_20MHZ, len(sub))):
        coef = np.polyfit(sub[ini:fim], desenrolada[ini:fim], 1)
        retas[ini:fim] = np.polyval(coef, sub[ini:fim])

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    axs[0].plot(sub, crua[antena, :, amostra], color=AZUL, lw=1.6)
    axs[0].set_title("1. Fase crua")
    axs[0].set_ylabel("fase (rad)")
    axs[0].text(0.02, 0.03, "enrolada em ±π:\nsaltos de 2π", transform=axs[0].transAxes,
                fontsize=8.5, color=TINTA_2)

    axs[1].plot(sub, desenrolada, color=AZUL, lw=1.6, label="fase desenrolada (unwrap)")
    axs[1].plot(sub, retas, color=LARANJA, lw=2, ls=(0, (5, 2)), label="reta ajustada (uma por metade)")
    axs[1].axvline(FRONTEIRA_20MHZ - 0.5, color=TINTA_2, lw=1, ls=":")
    axs[1].set_title("2. Desenrolar e achar a rampa")
    axs[1].legend(frameon=False, fontsize=8, loc="best")

    for t in range(crua.shape[2]):
        axs[2].plot(sub, limpa[antena, :, t], color=AZUL, lw=1, alpha=0.35 if t else 1)
    axs[2].axvline(FRONTEIRA_20MHZ - 0.5, color=TINTA_2, lw=1, ls=":")
    axs[2].set_title("3. Fase sem a rampa")
    axs[2].text(0.02, 0.03, "as 10 amostras do quadro", transform=axs[2].transAxes,
                fontsize=8.5, color=TINTA_2)

    for ax in axs:
        ax.set_xlabel("subportadora (0–113)")
        ax.grid(color=GRADE, lw=0.8)
    fig.suptitle("Limpeza da fase (tarefa d): tirar o erro do próprio rádio", x=0.0, ha="left",
                 fontweight="bold", fontsize=12.5, y=1.04)
    fig.text(0.0, -0.05,
             f"{sujeito}/{acao} ({ACOES[acao]}), quadro {quadro}, antena 1. O atraso de detecção do pacote e os "
             "desvios de oscilador/relógio criam uma rampa ao longo das subportadoras.\nO canal de 40 MHz são dois "
             "blocos de 20 MHz (linha pontilhada), cada um com sua referência — por isso uma reta por metade.",
             fontsize=7.5, color=TINTA_2, ha="left", va="top")
    fig.tight_layout()
    return _salvar(fig, "sinal_1_fase_sanitizada.png")


# ---------------------------------------------------------------------------
# 5. Sinal: o "ritmo" (Doppler) em três ações
# ---------------------------------------------------------------------------

def figura_doppler(acoes=("A12", "A26", "A13", "A17")) -> Path:
    """Espectro médio do Doppler (atributo dop_amp) de quatro ações, nas 40 pessoas.

    Lê o cache de atributos do ML-03 (``artifacts/cache_atributos/``); figura
    descritiva, não é avaliação de modelo.
    """
    from src.doppler import abrir_cache_atributos, frequencias_dop_amp

    cache = abrir_cache_atributos(RAIZ / "artifacts" / "cache_atributos")
    dop = cache.conjuntos["dop_amp"]                 # (sequências, janelas, 3 × 51)
    freqs = frequencias_dop_amp()
    corte = freqs <= 15
    cores = (AZUL, LARANJA, AQUA, "#eda100")

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for acao, cor in zip(acoes, cores):
        linhas = sorted(i for (_, a), i in cache.linhas.items() if a == acao)
        x = np.asarray(dop[linhas]).reshape(len(linhas), dop.shape[1], -1, len(freqs))
        media = x.mean(axis=(0, 1, 2))               # pessoas, janelas e antenas
        ax.plot(freqs[corte], media[corte], color=cor, lw=2, marker="o", ms=3.5,
                label=f"{acao} {ACOES[acao]}")
    ax.set_xlim(0, 15.3)
    ax.set_xticks(range(0, 16, 1))
    ax.grid(color=GRADE, lw=0.8)
    ax.set_xlabel("frequência da oscilação do sinal (Hz)  —  ligada à velocidade do movimento")
    ax.set_ylabel("energia em cada frequência\n(log, relativa ao total)")
    ax.set_title("O 'ritmo' do movimento (Doppler): corpo inteiro × só o braço")
    ax.legend(frameon=False, fontsize=9, loc="center right")
    ax.annotate("movimentos de corpo inteiro\n(agachar, saltar): mais energia\nem 1–4 Hz",
                xy=(2, -2.15), xytext=(5.2, -2.0), fontsize=8.5, color=TINTA_2,
                arrowprops=dict(arrowstyle="->", color=TINTA_2, lw=0.8))
    fig.text(0.0, -0.03,
             "Média das 40 pessoas, das 9 janelas de 3 s de cada gravação e das 3 antenas. Espectro de Welch "
             "(segmentos de 1 s) da amplitude, sem a parte parada da sala;\nmostrado até 15 Hz (o atributo vai "
             "até 50 Hz, o limite de Nyquist para 100 amostras/s). É o atributo dop_amp do ML-03 "
             "(artifacts/cache_atributos/).",
             fontsize=7.5, color=TINTA_2, ha="left", va="top")
    return _salvar(fig, "sinal_2_doppler.png")


# ---------------------------------------------------------------------------
# 6. CDF por ação: quanta probabilidade o modelo dá à ação certa
# ---------------------------------------------------------------------------

PROBABILIDADES = RESULTADOS / "probabilidades"

# (nome na figura, arquivo em artifacts/resultados/probabilidades/, cor)
FONTES_CDF = (
    ("Random Forest + Doppler", "probabilidades_baseline_s2_rf_j30_p30_seed0_abase-dop_amp-dop_cm.npz", AZUL),
    ("Rede temporal (CNN + GRU)", "probabilidades_rede_temporal_s2_seed0.npz", LARANJA),
)


def resumo_cdf_por_acao(fontes=FONTES_CDF, pasta: Path = PROBABILIDADES) -> dict:
    """Probabilidade da ação certa por janela, separada por ação, para cada modelo.

    Devolve ``{"acoes": [...], "modelos": [{"nome", "cor", "arquivo", "acuracia_janela",
    "por_acao": {acao: {"p_correta": array, "mediana", "acima_do_acaso", "acerto"}}}]}``.
    Modelos sem arquivo ficam de fora; sem nenhum, ``FileNotFoundError``.
    """
    from src.avaliacao import carregar_probabilidades, probabilidade_da_correta

    modelos, acoes = [], None
    for nome, arquivo, cor in fontes:
        caminho = Path(pasta) / arquivo
        if not caminho.is_file():
            print(f"    (sem {caminho.relative_to(RAIZ) if caminho.is_relative_to(RAIZ) else caminho}: "
                  f"{nome} fica de fora)")
            continue
        dados = carregar_probabilidades(caminho)
        acoes_arq = [str(a) for a in dados["acoes"]]
        if acoes is None:
            acoes = acoes_arq
        elif acoes_arq != acoes:
            raise ValueError(f"{arquivo}: ações diferentes das do primeiro modelo.")
        proba, y = dados["probabilidades"], dados["y"]
        p_certa = probabilidade_da_correta(proba, y)
        acerto = proba.argmax(axis=1) == y
        acaso = 1.0 / len(acoes)
        por_acao = {}
        for i, acao in enumerate(acoes):
            m = y == i
            por_acao[acao] = {
                "p_correta": p_certa[m],
                "n": int(m.sum()),
                "mediana": float(np.median(p_certa[m])),
                "acima_do_acaso": float((p_certa[m] > acaso).mean()),
                "acerto": float(acerto[m].mean()),
            }
        modelos.append({"nome": nome, "cor": cor, "arquivo": arquivo,
                        "acuracia_janela": float(acerto.mean()), "por_acao": por_acao})
    if not modelos:
        raise FileNotFoundError(f"Nenhum arquivo de probabilidades em {pasta}.")
    return {"acoes": acoes, "modelos": modelos}


def _nome_curto(acao: str, limite: int = 24) -> str:
    nome = ACOES.get(acao, "")
    return nome if len(nome) <= limite else nome[: limite - 1].rstrip() + "…"


def figura_cdf_por_acao(fontes=FONTES_CDF, pasta: Path = PROBABILIDADES,
                        limite_x: float = 0.5) -> Path:
    """CDF da probabilidade dada à ação certa, um painel por ação (pedido do professor).

    Cada curva diz, para uma ação, em que fração dos pedaços de 3 s o modelo deu
    à ação certa probabilidade ≤ x. Curva mais **à direita e mais baixa** = o
    modelo dá mais confiança à resposta certa. A linha tracejada é o chute (1/27).
    Os painéis vão da ação mais fácil para a mais difícil (mediana do 1º modelo).
    O eixo x vai até ``limite_x``: as probabilidades da ação certa raramente
    passam disso, e a nota de rodapé conta as janelas que ficam de fora.
    Também grava a tabela ``resumo_cdf_por_acao.json`` (a versão em números).
    """
    from src.avaliacao import cdf_empirica

    resumo = resumo_cdf_por_acao(fontes, pasta)
    acoes, modelos = resumo["acoes"], resumo["modelos"]
    acaso = 1.0 / len(acoes)
    referencia = modelos[0]["por_acao"]
    ordem = sorted(acoes, key=lambda a: -referencia[a]["mediana"])

    n_col = 7
    n_lin = int(np.ceil((len(acoes) + 1) / n_col))
    fig, axs = plt.subplots(n_lin, n_col, figsize=(2.05 * n_col, 1.95 * n_lin),
                            sharex=True, sharey=True)
    axs = axs.ravel()

    def desenhar(ax, curvas, titulo, negrito=False):
        ax.axvline(acaso, color=TINTA_2, lw=1, ls=(0, (4, 3)), zorder=1)
        for valores, cor in curvas:
            x, f = cdf_empirica(valores)
            ax.step(np.concatenate([[0.0], x, [1.0]]), np.concatenate([[0.0], f, [1.0]]),
                    where="post", color=cor, lw=1.8, zorder=2)
        ax.set_title(titulo, fontsize=8.2, loc="left", pad=3,
                     fontweight="bold" if negrito else "normal")
        ax.grid(color=GRADE, lw=0.6, zorder=0)
        ax.set_xlim(0, limite_x)
        ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=7, length=2)

    for ax, acao in zip(axs, ordem):
        desenhar(ax, [(m["por_acao"][acao]["p_correta"], m["cor"]) for m in modelos],
                 f"{acao} {_nome_curto(acao)}")

    # Último painel: todas as ações juntas + legenda.
    ax_total = axs[len(acoes)]
    desenhar(ax_total,
             [(np.concatenate([m["por_acao"][a]["p_correta"] for a in acoes]), m["cor"])
              for m in modelos],
             "Todas as ações", negrito=True)
    for ax in axs[len(acoes) + 1:]:
        ax.set_visible(False)

    for ax in axs[(n_lin - 1) * n_col:]:
        ax.set_xlabel("prob. da ação certa", fontsize=7.5)
    for ax in axs[::n_col]:
        ax.set_ylabel("fração ≤ x", fontsize=7.5)
    em_pct = matplotlib.ticker.FuncFormatter(lambda v, _: f"{100 * v:.0f}%")
    for ax in axs:
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(
            np.linspace(0, limite_x, 3)))
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 0.5, 1]))
        ax.xaxis.set_major_formatter(em_pct)
        ax.yaxis.set_major_formatter(em_pct)

    fora = []
    for m in modelos:
        todas = np.concatenate([m["por_acao"][a]["p_correta"] for a in acoes])
        n_fora = int((todas > limite_x).sum())
        if n_fora:
            fora.append(f"{n_fora} de {todas.size} janelas da {m['nome']}")
    nota_fora = (f"Eixo cortado em {100 * limite_x:.0f}%; passam disso: "
                 + "; ".join(fora) + "." if fora
                 else f"Eixo cortado em {100 * limite_x:.0f}%; nenhuma janela passa disso.")

    alcas = [matplotlib.lines.Line2D([], [], color=m["cor"], lw=2,
                                     label=f"{m['nome']} (acerto {_pct(m['acuracia_janela'])})")
             for m in modelos]
    alcas.append(matplotlib.lines.Line2D([], [], color=TINTA_2, lw=1, ls=(0, (4, 3)),
                                         label="chute (1/27 = 3,7%)"))
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.legend(handles=alcas, loc="lower left", bbox_to_anchor=(0.0, 0.958), ncol=len(alcas),
               frameon=False, fontsize=9)
    fig.suptitle("CDF por ação: quanta probabilidade o modelo dá à ação certa (pessoas novas, S2)",
                 x=0.0, ha="left", fontweight="bold", fontsize=12.5, y=1.025)
    fig.text(0.0, -0.012,
             "Cada curva: as 72 janelas de 3 s da ação no teste (8 pessoas × 9). Curva mais à direita e mais "
             "baixa = mais confiança na ação certa; onde a curva cruza o\nchute, lê-se a fração de janelas em que o "
             "modelo deu à ação certa menos que um palpite aleatório. Painéis da ação mais fácil para a mais "
             f"difícil ({modelos[0]['nome']}).\n{nota_fora} Seed 0 de cada modelo. Números: "
             "artifacts/resultados/probabilidades/resumo_cdf_por_acao.json",
             fontsize=7.5, color=TINTA_2, ha="left", va="top")

    tabela = {
        "o_que_e": "probabilidade dada à ação verdadeira, por janela de 3 s, teste S2; "
                   "mediana, fração acima do acaso (1/27) e acerto top-1, por ação",
        "acaso": acaso,
        "modelos": [{
            "nome": m["nome"], "arquivo": m["arquivo"], "acuracia_janela": m["acuracia_janela"],
            "por_acao": {a: {k: v for k, v in m["por_acao"][a].items() if k != "p_correta"}
                         for a in ordem},
        } for m in modelos],
    }
    Path(pasta).mkdir(parents=True, exist_ok=True)
    (Path(pasta) / "resumo_cdf_por_acao.json").write_text(
        json.dumps(tabela, ensure_ascii=False, indent=1), encoding="utf-8")
    return _salvar(fig, "resultado_4_cdf_por_acao.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sem-sinal", action="store_true",
                        help="Não gera as figuras que leem o dataset (fase e Doppler).")
    parser.add_argument("--apenas-cdf", action="store_true",
                        help="Gera só a CDF por ação (lê artifacts/resultados/probabilidades/).")
    args = parser.parse_args()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    if args.apenas_cdf:
        figura_cdf_por_acao()
        return 0
    figura_comparacao()
    figura_assinatura()
    figura_por_acao()
    try:
        figura_cdf_por_acao()
    except FileNotFoundError as erro:
        print(f"    (CDF por ação pulada: {erro})")
    if not args.sem_sinal:
        figura_fase()
        figura_doppler()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
