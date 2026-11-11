"""
Testes do ML-04: rede temporal própria, carregamento sem vazamento, tabela de
comparação e guardas do script que roda as redes neurais v1 e v2.

Todos rodam sem o dataset (sinal sintético).

Uso:
    .venv/Scripts/python.exe -m pytest tests/test_ml04.py -v
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import rede_temporal as rt  # noqa: E402
from src.mmfi import PROTOCOLO_3, SUJEITOS_TESTE_S2  # noqa: E402
from src.protocolos import SUJEITOS_VALIDACAO_S2  # noqa: E402


def _carregar_script(nome):
    especificacao = importlib.util.spec_from_file_location(nome, RAIZ / "scripts" / f"{nome}.py")
    modulo = importlib.util.module_from_spec(especificacao)
    especificacao.loader.exec_module(modulo)
    return modulo


comparar = _carregar_script("comparar_modelos")
rodador = _carregar_script("rodar_rede_neural")
treino_script = _carregar_script("treinar_rede_temporal")

ACOES = tuple(sorted(PROTOCOLO_3))


# --- modelo ---------------------------------------------------------------

def test_forma_da_saida_e_numero_de_parametros():
    modelo = rt.RedeTemporal()
    modelo.eval()
    with torch.no_grad():
        saida = modelo(torch.randn(2, 30, rt.N_CANAIS, 114))
    assert saida.shape == (2, 27)
    assert 280_000 < rt.contar_parametros(modelo) < 290_000


def test_gradiente_chega_a_todos_os_parametros():
    torch.manual_seed(0)
    modelo = rt.RedeTemporal()
    perda = torch.nn.functional.cross_entropy(modelo(torch.randn(4, 8, rt.N_CANAIS, 114)),
                                              torch.tensor([0, 1, 2, 3]))
    perda.backward()
    for nome, p in modelo.named_parameters():
        assert p.grad is not None, nome
        assert torch.isfinite(p.grad).all(), nome
        assert p.grad.abs().sum() > 0, nome


# --- dados ----------------------------------------------------------------

def test_canais_da_sequencia_ordem_e_media_da_sequencia():
    rng = np.random.default_rng(0)
    amp = rng.normal(size=(12, 3, 114, 10)).astype(np.float32) + 5.0
    fase = rng.normal(size=(12, 3, 114, 10)).astype(np.float32)
    x = rt.canais_da_sequencia(amp, fase)
    assert x.shape == (12, 60, 114) and x.dtype == np.float32
    np.testing.assert_allclose(x.mean(axis=0), 0.0, atol=1e-5)
    # canal = sinal*30 + antena*10 + amostra
    np.testing.assert_allclose(x[:, 1 * 10 + 7, 40], amp[:, 1, 40, 7] - amp[:, 1, 40, 7].mean(),
                               atol=1e-5)
    np.testing.assert_allclose(x[:, 30 + 2 * 10 + 3, 5], fase[:, 2, 5, 3] - fase[:, 2, 5, 3].mean(),
                               atol=1e-5)
    with pytest.raises(ValueError):
        rt.canais_da_sequencia(amp[:, :2], fase[:, :2])


def test_divisao_sem_vazamento():
    treino, validacao, teste = rt.dividir_rede(ACOES)
    sujeitos = [{s for s, _ in lado} for lado in (treino, validacao, teste)]
    assert sujeitos[1] == set(SUJEITOS_VALIDACAO_S2)
    assert sujeitos[2] == set(SUJEITOS_TESTE_S2)
    assert not (sujeitos[0] & sujeitos[1]) and not (sujeitos[0] & sujeitos[2])
    assert not (sujeitos[1] & sujeitos[2])
    assert (len(treino), len(validacao), len(teste)) == (28 * 27, 4 * 27, 8 * 27)


@pytest.mark.parametrize("sujeitos, mensagem", [
    (("S04", "S05"), "no teste"),
    (("S05",), "no teste"),
    ((), "Sem sujeitos"),
    (("S41",), "fora do treino"),
])
def test_validacao_vazia_no_teste_ou_fora_do_treino_falha(sujeitos, mensagem):
    with pytest.raises(ValueError, match=mensagem):
        rt.dividir_rede(ACOES, sujeitos)


def _cache_falso(sequencias, quadros=40, seed=0):
    """amp/fase ``(n, quadros, 3, 114, 10)`` e ``linhas`` como os do cache_csi."""
    rng = np.random.default_rng(seed)
    forma = (len(sequencias), quadros, 3, 114, 10)
    amp = rng.normal(size=forma).astype(np.float32)
    fase = rng.normal(size=forma).astype(np.float32)
    linhas = {tuple(s): i for i, s in enumerate(sequencias)}
    return amp, fase, linhas


def test_carregamento_e_estatisticas_nao_usam_validacao_nem_teste():
    seqs_tr = [("S01", "A01"), ("S01", "A02"), ("S02", "A03")]
    seqs_val = [("S04", "A01")]
    seqs_te = [("S05", "A02")]
    todas = seqs_tr + seqs_val + seqs_te
    amp, fase, linhas = _cache_falso(todas)

    tr = rt.carregar_dados(amp, fase, linhas, seqs_tr, ACOES)
    assert tr.x.shape == (3, 40, 60, 114) and tr.x.dtype == np.float16
    assert list(tr.y) == [0, 1, 2]
    media, desvio = rt.estatisticas_canal(tr.x)

    # Validação e teste com valores absurdos: nada do treino muda.
    amp2, fase2 = amp.copy(), fase.copy()
    for seq in seqs_val + seqs_te:
        amp2[linhas[seq]] = amp2[linhas[seq]] * 1000 + 50
        fase2[linhas[seq]] = -fase2[linhas[seq]] * 300
    tr2 = rt.carregar_dados(amp2, fase2, linhas, seqs_tr, ACOES)
    np.testing.assert_array_equal(tr.x, tr2.x)
    media2, desvio2 = rt.estatisticas_canal(tr2.x)
    np.testing.assert_array_equal(media, media2)
    np.testing.assert_array_equal(desvio, desvio2)

    # Estatísticas = média/desvio por canal do próprio treino.
    x64 = tr.x.astype(np.float64)
    np.testing.assert_allclose(media, x64.mean(axis=(0, 1, 3)), atol=1e-5)
    np.testing.assert_allclose(desvio, x64.std(axis=(0, 1, 3)), rtol=1e-4)


def test_montar_lote_padroniza_a_janela_certa():
    seqs = [("S01", "A01"), ("S02", "A05")]
    amp, fase, linhas = _cache_falso(seqs, quadros=20)
    dados = rt.carregar_dados(amp, fase, linhas, seqs, ACOES)
    media, desvio = rt.estatisticas_canal(dados.x)
    lote = rt.montar_lote(dados, np.array([1, 0]), np.array([3, 10]), 5, media, desvio)
    assert lote.shape == (2, 5, 60, 114) and lote.dtype == torch.float32
    esperado = (dados.x[1, 3:8].astype(np.float32) - media[None, :, None]) / desvio[None, :, None]
    np.testing.assert_allclose(lote[0].numpy(), esperado, rtol=1e-6)


def _dados_sinteticos(n, quadros=20, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 3
    x = rng.normal(size=(n, quadros, rt.N_CANAIS, 114)).astype(np.float32)
    x[:, :, :5, :] += y[:, None, None, None]  # sinal aprendível
    return rt.DadosRede(x=x.astype(np.float16), y=y.astype(np.int64),
                        sequencias=[(f"S{i:02d}", "A01") for i in range(n)])


def test_treino_deterministico_e_epoca_pela_validacao():
    treino, validacao = _dados_sinteticos(6), _dados_sinteticos(3, seed=1)
    media, desvio = rt.estatisticas_canal(treino.x)
    resultados = []
    for _ in range(2):
        modelo, historico, epoca = rt.treinar(treino, validacao, media, desvio, seed=0,
                                              n_classes=3, epocas=2, janela=5, lote=4,
                                              log=lambda _: None)
        proba, y, grupos = rt.prever(modelo, validacao, media, desvio, janela=5)
        resultados.append((historico, epoca, proba))
    (h1, e1, p1), (h2, e2, p2) = resultados
    assert e1 == e2 == rt.melhor_epoca(h1)
    assert [h["validacao"] for h in h1] == [h["validacao"] for h in h2]
    np.testing.assert_array_equal(p1, p2)
    assert proba.shape == (3 * 4, 3) and list(grupos) == [0] * 4 + [1] * 4 + [2] * 4
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)


def test_modelo_devolvido_e_o_da_epoca_escolhida():
    """As previsões na validação depois do treino reproduzem o histórico da época
    escolhida (e não as da última época)."""
    treino, validacao = _dados_sinteticos(6), _dados_sinteticos(3, seed=1)
    media, desvio = rt.estatisticas_canal(treino.x)
    modelo, historico, epoca = rt.treinar(treino, validacao, media, desvio, seed=3,
                                          n_classes=3, epocas=4, janela=5, lote=4,
                                          log=lambda _: None)
    assert epoca == rt.melhor_epoca(historico)
    medido = rt.medir_validacao(*rt.prever(modelo, validacao, media, desvio, janela=5))
    assert medido == historico[epoca - 1]["validacao"]


def test_seeds_diferentes_dao_resultados_diferentes():
    treino, validacao = _dados_sinteticos(6), _dados_sinteticos(3, seed=1)
    media, desvio = rt.estatisticas_canal(treino.x)
    probas = []
    for seed in (0, 1):
        modelo, _, _ = rt.treinar(treino, validacao, media, desvio, seed=seed, n_classes=3,
                                  epocas=1, janela=5, lote=4, log=lambda _: None)
        probas.append(rt.prever(modelo, validacao, media, desvio, janela=5)[0])
    assert not np.allclose(probas[0], probas[1])


def test_melhor_epoca_desempata_por_janela_e_depois_pela_mais_cedo():
    def h(seq, jan):
        return {"validacao": {"acuracia_sequencia": seq, "acuracia_janela": jan}}
    assert rt.melhor_epoca([h(0.1, 0.1), h(0.3, 0.2), h(0.3, 0.25), h(0.2, 0.9)]) == 3
    assert rt.melhor_epoca([h(0.3, 0.2), h(0.3, 0.2)]) == 1


# --- script de treino: preparação e resultado (sem dataset) ----------------

SEQS_TR = [("S01", "A01"), ("S01", "A02"), ("S02", "A03"), ("S02", "A01"),
           ("S03", "A02"), ("S03", "A03")]
SEQS_VAL = [("S04", "A01"), ("S04", "A02"), ("S04", "A03")]
SEQS_TE = [("S05", "A01"), ("S05", "A02"), ("S05", "A03")]
ACOES_PEQ = ("A01", "A02", "A03")


def _preparar(amp, fase, linhas):
    return treino_script.preparar_dados(amp, fase, linhas, ACOES_PEQ,
                                        (SEQS_TR, SEQS_VAL, SEQS_TE))


def test_preparar_dados_padroniza_so_pelo_treino():
    amp, fase, linhas = _cache_falso(SEQS_TR + SEQS_VAL + SEQS_TE, quadros=20)
    dados, media, desvio = _preparar(amp, fase, linhas)
    assert [len(dados[k].y) for k in ("treino", "validação", "teste")] == [6, 3, 3]
    m_tr, d_tr = rt.estatisticas_canal(dados["treino"].x)
    np.testing.assert_array_equal(media, m_tr)
    np.testing.assert_array_equal(desvio, d_tr)

    amp2, fase2 = amp.copy(), fase.copy()
    for seq in SEQS_VAL + SEQS_TE:
        amp2[linhas[seq]] = amp2[linhas[seq]] * 100 + 7
        fase2[linhas[seq]] = fase2[linhas[seq]] * 50
    _, media2, desvio2 = _preparar(amp2, fase2, linhas)
    np.testing.assert_array_equal(media, media2)
    np.testing.assert_array_equal(desvio, desvio2)


def _rodar_seed_sintetico(tmp_path, monkeypatch, seed=0):
    amp, fase, linhas = _cache_falso(SEQS_TR + SEQS_VAL + SEQS_TE, quadros=20)
    dados, media, desvio = _preparar(amp, fase, linhas)

    chamadas_teste = []
    prever_original = treino_script.prever

    def prever_contando(modelo, d, *args, **kwargs):
        if d is dados["teste"]:
            chamadas_teste.append(len(gravados))
        return prever_original(modelo, d, *args, **kwargs)

    monkeypatch.setattr(treino_script, "prever", prever_contando)
    gravados = []

    def gravar(caminho, conteudo):
        gravados.append((caminho, json.loads(json.dumps(conteudo))))
        treino_script._gravar(caminho, conteudo)

    cabecalho = treino_script.montar_cabecalho(ACOES_PEQ, 5, 2, {"treino": 6})
    final = treino_script.rodar_seed(
        dados, media, desvio, seed, cabecalho, ACOES_PEQ, tmp_path, tmp_path / "fig",
        epocas=2, janela=5, lote=4, gravar=gravar, log=lambda _: None)
    return final, gravados, chamadas_teste


def test_rodar_seed_grava_escolha_antes_e_avalia_o_teste_uma_vez(tmp_path, monkeypatch):
    final, gravados, chamadas_teste = _rodar_seed_sintetico(tmp_path, monkeypatch)
    assert chamadas_teste == [1]                  # uma vez, depois do registro parcial
    (c1, parcial), (c2, completo) = gravados
    assert c1 == c2 and c1.name == "rede_temporal_s2_seed0_j5_e2.json"
    assert parcial["estado"] == "época escolhida; teste ainda não avaliado"
    assert "teste" not in parcial
    assert parcial["epoca_escolhida"] == completo["epoca_escolhida"]
    assert completo["estado"] == "completo"
    assert completo["teste"]["por_sequencia"]["n"] == 3
    assert (tmp_path / "fig" / "confusao_rede_temporal_s2_seed0_j5_e2.png").is_file()
    assert final["unidade"].startswith("janela de 0.5 s (5 quadros)")


def test_nome_base_so_ganha_sufixo_fora_do_padrao():
    assert treino_script.nome_base(2) == "rede_temporal_s2_seed2"
    assert treino_script.nome_base(2, 30, 15) == "rede_temporal_s2_seed2"
    assert treino_script.nome_base(2, 60, 5) == "rede_temporal_s2_seed2_j60_e5"
    assert treino_script.descrever_unidade(30).startswith("janela de 3 s (30 quadros)")


def test_resultado_do_script_entra_na_linha_certa_da_comparacao(tmp_path, monkeypatch):
    final, _, _ = _rodar_seed_sintetico(tmp_path, monkeypatch)
    # Fora do padrão (j5_e2), o arquivo gravado não entra na tabela...
    comp = comparar.comparar(tmp_path, lambda _: None)
    assert not comp["linhas"][2]["executado"]
    # ...com o nome padrão, entra na linha da rede temporal.
    (tmp_path / "rede_temporal_s2_seed0.json").write_text(json.dumps(final), encoding="utf-8")
    comp = comparar.comparar(tmp_path, lambda _: None)
    rede = comp["linhas"][2]
    assert rede["modelo"].startswith("Rede temporal") and rede["n_execucoes"] == 1
    assert rede["por_janela"]["media"] == pytest.approx(final["teste"]["por_janela"]["acuracia"])
    assert rede["por_sequencia"]["media"] == pytest.approx(
        final["teste"]["por_sequencia"]["acuracia"])
    assert rede["por_janela"]["desvio"] is None


# --- comparação -----------------------------------------------------------

def _gravar(pasta, nome, dados):
    (pasta / nome).write_text(json.dumps(dados), encoding="utf-8")


def _execucao(seed, jan, seq):
    return {"seed": seed, "por_janela": {"acuracia": jan}, "por_sequencia": {"acuracia": seq}}


def _rede(seed, jan, seq, epocas=15, epoca=7):
    return {"estado": "completo", "epoca_escolhida": epoca, "protocolo": "s2",
            "hiperparametros": {"epocas": epocas, "janela": 30},
            "teste": {"por_janela": {"acuracia": jan}, "por_sequencia": {"acuracia": seq}}}


def _rede_neural(versao, i, jan, seq=None, commit="c", epocas=15):
    dados = {"epocas": epocas, "split": "S2", "protocolo": "P3",
             "origem": {"commit": commit, "seed_controlada": versao == 2,
                        "comando": "python scripts/treinar_rede.py --split S2"
                                   + (f" --seed {i}" if versao == 2 else ""),
                        "tempo_total_s": 6420.0}}
    if versao == 1:
        dados["acuracia_final"] = jan
    else:
        dados["acuracia_final_janela"] = jan
        if seq is not None:
            dados["acuracia_final_sequencia"] = seq
    return dados


def _resultados_sinteticos(pasta):
    _gravar(pasta, "selecao_atributos.json", {
        "escolhida": ["base", "dop_amp", "dop_cm"],
        "teste": {
            "base": {"execucoes": [_execucao(0, 0.15, 0.18), _execucao(1, 0.17, 0.20)]},
            "base,dop_amp,dop_cm": {"execucoes": [_execucao(0, 0.19, 0.24)]},
        }})
    for seed in (0, 1):
        _gravar(pasta, f"rede_temporal_s2_seed{seed}.json",
                _rede(seed, 0.2 + 0.02 * seed, 0.25, epoca=7 + seed))
    # seed 2 só com a escolha gravada (execução interrompida)
    _gravar(pasta, "rede_temporal_s2_seed2.json", {"estado": "época escolhida; teste ainda não avaliado"})
    # configuração fora do padrão: não entra
    _gravar(pasta, "rede_temporal_s2_seed5_j60_e5.json", _rede(5, 0.9, 0.9, epocas=5))
    for i in range(3):
        _gravar(pasta, f"rede_neural_v1_exec{i}.json", _rede_neural(1, i, 0.10 + 0.01 * i))
    # v2 ausente


def test_comparacao_com_json_sinteticos_e_ausente(tmp_path):
    _resultados_sinteticos(tmp_path)
    avisos = []
    comp = comparar.comparar(tmp_path, avisos.append)
    linhas = {l["modelo"].split(" (")[0]: l for l in comp["linhas"]}
    assert len(comp["linhas"]) == 5
    assert comp["acaso"] == pytest.approx(1 / 27)

    rf = linhas["RF base"]
    assert rf["n_execucoes"] == 2
    assert rf["por_janela"]["media"] == pytest.approx(0.16)
    assert rf["por_janela"]["desvio"] == pytest.approx(np.std([0.15, 0.17], ddof=1))
    assert linhas["RF + Doppler"]["n_execucoes"] == 1
    assert linhas["RF + Doppler"]["por_sequencia"]["desvio"] is None

    rede = linhas["Rede temporal"]
    assert rede["n_execucoes"] == 2 and rede["detalhes"]["epocas_escolhidas"] == [7, 8]
    assert rede["detalhes"]["seeds"] == [0, 1]            # seed5_j60_e5 ficou de fora
    assert any("rede_temporal_s2_seed2" in a for a in avisos)

    v1 = linhas["Rede neural v1 — CNN multitarefa, quadro de 100 ms"]
    assert v1["n_execucoes"] == 3 and v1["por_sequencia"] is None and not v1["notas"]
    assert "quadro" in v1["unidade"] and v1["seed_controlada"] is False
    assert v1["escolhido_sem_olhar_o_teste"].startswith("parcial")

    v2 = linhas["Rede neural v2 — MLP sobre atributos de 3 s"]
    assert not v2["executado"] and v2["n_execucoes"] == 0
    assert any("rede_neural_v2" in a for a in avisos)

    tabela = comparar.tabela_markdown(comp)
    linha_v1 = next(l for l in tabela.splitlines() if "Rede neural v1" in l)
    assert "quadro" in linha_v1 and "| — |" in linha_v1        # sem número por sequência
    linha_v2 = next(l for l in tabela.splitlines() if "Rede neural v2" in l)
    assert "não executado" in linha_v2
    assert "Acaso" in tabela and "3.7" in tabela
    assert "Leitura:" in tabela and "pareado" in tabela
    for linha in tabela.splitlines()[2:]:
        if not linha.startswith("|") or "Acaso" in linha:
            continue
        # unidade, execuções e "sem olhar o teste" preenchidos em toda linha
        celulas = [c.strip() for c in linha.strip("|").split("|")]
        assert celulas[1] and celulas[4] and celulas[7]


def test_v1_com_uma_execucao_ganha_nota_sem_desvio_e_sem_aviso_das_cortadas(tmp_path):
    _gravar(tmp_path, "rede_neural_v1_exec0.json", _rede_neural(1, 0, 0.0574))
    avisos = []
    comp = comparar.comparar(tmp_path, avisos.append)
    v1 = next(l for l in comp["linhas"] if "Rede neural v1" in l["modelo"])
    assert v1["n_execucoes"] == 1 and v1["por_janela"]["desvio"] is None
    assert "decisão do usuário" in v1["notas"][0] and "~107 min" in v1["notas"][0]
    assert not any("rede_neural_v1" in a for a in avisos)      # exec1/exec2 cortadas de propósito
    tabela = comparar.tabela_markdown(comp)
    linha = next(l for l in tabela.splitlines() if "Rede neural v1" in l)
    assert "| 5.7 |" in linha and "±" not in linha and "1 (nota 1)" in linha
    assert any(l.startswith("Nota 1:") and "tempo" in l for l in tabela.splitlines())


def test_v2_sem_acuracia_por_sequencia_avisa_e_conta_por_coluna(tmp_path):
    _gravar(tmp_path, "rede_neural_v2_exec0.json", _rede_neural(2, 0, 0.15, 0.19))
    _gravar(tmp_path, "rede_neural_v2_exec1.json", _rede_neural(2, 1, 0.16))
    _gravar(tmp_path, "rede_neural_v2_exec7.json", _rede_neural(2, 7, 0.17, 0.18))   # descoberto pelo nome
    avisos = []
    v2 = comparar.comparar(tmp_path, avisos.append)["linhas"][4]
    assert (v2["n_execucoes"], v2["n_por_janela"], v2["n_por_sequencia"]) == (3, 3, 2)
    assert any("acuracia_final_sequencia" in a for a in avisos)
    assert not any("difere" in a for a in avisos)          # só a seed muda no comando
    linha = next(l for l in comparar.tabela_markdown({"acaso": 1 / 27, "linhas": [v2]})
                 .splitlines() if "Rede neural v2" in l)
    assert "3 (janela) / 2 (sequência)" in linha


def test_execucoes_inconsistentes_geram_aviso(tmp_path):
    _gravar(tmp_path, "rede_neural_v2_exec0.json", _rede_neural(2, 0, 0.15, 0.19, commit="a"))
    _gravar(tmp_path, "rede_neural_v2_exec1.json", _rede_neural(2, 1, 0.15, 0.19, commit="b", epocas=9))
    _gravar(tmp_path, "rede_temporal_s2_seed0.json", _rede(0, 0.1, 0.1))
    _gravar(tmp_path, "rede_temporal_s2_seed1.json", _rede(1, 0.1, 0.1, epocas=20))
    avisos = []
    comparar.comparar(tmp_path, avisos.append)
    assert any("rede_neural_v2_exec1.json difere" in a and "commit" in a and "epocas" in a
               for a in avisos)
    assert any("rede_temporal_s2_seed1.json difere" in a and "epocas" in a for a in avisos)


def test_selecao_sem_escolhida_avisa_e_escolhida_em_texto_funciona(tmp_path):
    teste = {"base": {"execucoes": [_execucao(0, 0.15, 0.18)]},
             "base,dop_amp,dop_cm": {"execucoes": [_execucao(0, 0.19, 0.24)]}}
    _gravar(tmp_path, "selecao_atributos.json", {"teste": teste})
    avisos = []
    linhas = comparar.comparar(tmp_path, avisos.append)["linhas"]
    assert linhas[0]["executado"] and not linhas[1]["executado"]
    assert any("sem 'escolhida'" in a for a in avisos)

    _gravar(tmp_path, "selecao_atributos.json", {"escolhida": "base,dop_amp,dop_cm", "teste": teste})
    linhas = comparar.comparar(tmp_path, lambda _: None)["linhas"]
    assert linhas[1]["por_janela"]["media"] == pytest.approx(0.19)


def test_main_sem_nenhum_resultado_nao_quebra(tmp_path, capsys):
    assert comparar.main(["--resultados", str(tmp_path)]) == 0
    saida = capsys.readouterr().out
    assert "AVISO" in saida and "Traceback" not in saida
    dados = json.loads((tmp_path / "comparacao_modelos.json").read_text(encoding="utf-8"))
    assert all(not l["executado"] for l in dados["linhas"])


@pytest.mark.parametrize("conteudo", [
    "{quebrado", "null", "[1, 2]", "3",
    json.dumps({"teste": None}),
    json.dumps({"escolhida": ["base"], "teste": {"base": {"execucoes": [{"seed": 0}]}}}),
    json.dumps({"escolhida": ["base"], "teste": {"base": {"execucoes": "x"}}}),
    json.dumps({"escolhida": ["base"], "teste": {"base": [1]}}),
])
def test_selecao_fora_do_formato_vira_nao_executado(tmp_path, conteudo):
    (tmp_path / "selecao_atributos.json").write_text(conteudo, encoding="utf-8")
    avisos = []
    comp = comparar.comparar(tmp_path, avisos.append)
    assert not comp["linhas"][0]["executado"]
    assert avisos


@pytest.mark.parametrize("nome, conteudo", [
    ("rede_temporal_s2_seed0.json", {"estado": "completo"}),
    ("rede_temporal_s2_seed0.json", {"estado": "completo", "teste": {"por_janela": None}}),
    ("rede_temporal_s2_seed0.json", {"estado": "completo", "teste": {
        "por_janela": {"acuracia": "alta"}, "por_sequencia": {"acuracia": 0.1}}}),
    ("rede_neural_v1_exec0.json", {"origem": {}}),
    ("rede_neural_v1_exec0.json", {"acuracia_final": None}),
    ("rede_neural_v2_exec0.json", {"acuracia_final_janela": [0.1]}),
    ("rede_neural_v2_exec0.json", None),
])
def test_resultado_presente_mas_fora_do_formato_nao_quebra(tmp_path, nome, conteudo):
    (tmp_path / nome).write_text(json.dumps(conteudo), encoding="utf-8")
    avisos = []
    comp = comparar.comparar(tmp_path, avisos.append)
    assert all(not l["executado"] for l in comp["linhas"])
    assert any(nome in a for a in avisos)
    comparar.tabela_markdown(comp)


# --- redes neurais v1 e v2 ------------------------------------------------------

def test_alterados_fora_de_resultados():
    antes = {"src/a.py": "h1", "artifacts/resultados/x.json": "h2"}
    depois = {"src/a.py": "h1", "artifacts/resultados/x.json": "h3",
              "artifacts/resultados/y.json": "h4", "README.md": "h5"}
    assert rodador.alterados_fora_de_resultados(antes, depois) == ["README.md"]
    assert rodador.alterados_fora_de_resultados({"a.py": "h1"}, {}) == ["a.py"]   # revertido
    assert rodador.alterados_fora_de_resultados({}, {"a.py": None}) == ["a.py"]   # apagado


def _repo_git(pasta):
    import subprocess
    for comando in (["init", "-q"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", *comando], cwd=pasta, check=True)
    (pasta / "codigo.py").write_text("a = 1\n")
    (pasta / "artifacts" / "resultados").mkdir(parents=True)
    (pasta / ".gitignore").write_text("ignorado/\n")
    subprocess.run(["git", "add", "."], cwd=pasta, check=True)
    subprocess.run(["git", "commit", "-qm", "x"], cwd=pasta, check=True)


def test_instantaneo_compara_conteudo_e_ignora_resultados(tmp_path):
    _repo_git(tmp_path)
    (tmp_path / "codigo.py").write_text("a = 2\n")        # já modificado antes
    antes = rodador.instantaneo_repositorio(tmp_path)
    assert list(antes) == ["codigo.py"]
    (tmp_path / "codigo.py").write_text("a = 3\n")        # mesmo status " M", outro conteúdo
    (tmp_path / "artifacts" / "resultados" / "r.json").write_text("{}")
    (tmp_path / "ignorado").mkdir()
    (tmp_path / "ignorado" / "x.txt").write_text("x")     # git-ignorado: não coberto
    depois = rodador.instantaneo_repositorio(tmp_path)
    assert rodador.alterados_fora_de_resultados(antes, depois) == ["codigo.py"]


def test_criar_worktree_apaga_a_pasta_se_o_git_falhar(tmp_path, monkeypatch):
    def falhar(*args, **kwargs):
        if args[:2] == ("worktree", "add"):
            raise RuntimeError("git worktree add falhou")
        return None

    monkeypatch.setattr(rodador, "_git", falhar)
    with pytest.raises(RuntimeError):
        rodador.criar_worktree("0" * 40, str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_ambiente_do_filho_sem_pythonpath(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONPATH", "/algum/lugar")
    monkeypatch.setenv("MMFI_DATASET", str(tmp_path))
    env = rodador.ambiente_filho()
    assert "PYTHONPATH" not in env and env["MMFI_DATASET"] == str(tmp_path)
    assert "sys.path[0] = os.path.join(os.getcwd(), 'scripts')" in rodador.LANCADOR


def test_versoes_da_rede_neural_registradas():
    v1, v2 = rodador.VERSOES["1"], rodador.VERSOES["2"]
    assert v1["commit"].startswith("98a1f49") and not v1["seed_controlada"]
    assert v2["commit"].startswith("68d96a9") and v2["seed_controlada"]
    assert "--seed" not in v1["argumentos"](0)
    assert v2["argumentos"](2)[-2:] == ["--seed", "2"]
