# Resumo do projeto — onde estamos e para onde vamos

**Atualizado em:** 20/10/2026

Um resumo direto do trabalho, para quem chegou agora ou quer se situar. Os
detalhes técnicos estão nos outros documentos de `docs/`.

---

## 1. A ideia do trabalho

O sinal de WiFi atravessa a sala e bate nas paredes, nos móveis **e nas
pessoas**. Quando alguém se mexe, o sinal que chega ao receptor muda um pouco.

A pergunta é: **dá para descobrir o que a pessoa está fazendo só olhando essas
mudanças no sinal, sem câmera?**

É parecido com um morcego, que não enxerga mas sabe onde as coisas estão pelo
eco. Aqui o "eco" é o WiFi. A vantagem sobre uma câmera é a privacidade: ninguém
é filmado.

---

## 2. O que o professor pede

São cinco tarefas, conforme o `tema.pdf`:

| | Tarefa | Em palavras simples |
|---|---|---|
| **a** | Revisão bibliográfica | Ler os dois artigos (Wi-Pose e Winect) e explicar o que cada um fez |
| **b** | Manipular os dados | Abrir os arquivos do sinal e fazer gráficos |
| **c** | Classificar movimentos | Um programa que olha o sinal e diz qual ação a pessoa está fazendo |
| **d** | Limpar o sinal | O sinal vem com ruído e distorção; tirar essa sujeira |
| **e** | Análise crítica | Mostrar onde o programa erra e explicar por quê |

O enunciado pede um modelo **simples** — não exige nada sofisticado.

---

## 3. O que já foi feito

### Troca de dataset

O dataset indicado no enunciado (Wi-Pose) era incompleto: não trazia a **fase**
do sinal, só a amplitude, e as poses eram 2D. Trocamos para o **MM-Fi**:
**40 pessoas** fazendo **27 ações** (agachar, chutar, acenar, pular…) em
**4 salas diferentes**, gravadas com WiFi e com a pose 3D de referência. O
professor já está ciente da troca. O porquê completo está em
[`analise-mmfi.md`](analise-mmfi.md).

### Tarefa (b) — dados e gráficos ✅

Código que abre os dados do MM-Fi e desenha o sinal e o esqueleto 3D
(`src/mmfi.py`, `scripts/inspecionar_mmfi.py`). Também existe uma **versão
leve** do dataset (6 GB em vez de 16 GB), com um verificador de integridade.

### Tarefa (d) — limpeza do sinal 🟡

Primeira técnica implementada: **sanitização da fase**, que remove uma
distorção em forma de rampa que o próprio equipamento introduz no sinal.

### Tarefa (c) — primeiro classificador ✅

O primeiro "adivinhador" de ações:

1. Corta cada gravação em **pedaços de 3 segundos**.
2. Resume cada pedaço em números: para cada antena e cada frequência, a
   **média** do sinal (como a sala é) e o **quanto ele variou** (quanto o
   corpo mexeu no sinal).
3. Um modelo simples (**Random Forest** ou **SVM**) aprende a ligar esses
   números a uma das 27 ações.

**O teste foi feito do jeito honesto:** com pessoas que o modelo nunca viu no
treino, e sem nunca deixar pedaços da mesma gravação dos dois lados (isso faria
o modelo "colar" e inflaria a nota).

### Resultado

Chutando no escuro, a chance de acertar entre 27 ações é **3,7%** (1 em 27).

| Modelo | Pessoas novas no teste (S2) | Pessoas conhecidas (S1) |
|---|---|---|
| Random Forest | **14,5%** | 5,2% |
| SVM | **16,3%** | 6,4% |

*Acerto por pedaço de 3 s. Por gravação inteira, o SVM chega a 17,6% no S2.*

O modelo acerta **cerca de 4 vezes mais que o chute** com pessoas que nunca
viu. Parece pouco, mas é um começo honesto — e nem os autores do MM-Fi
publicaram esse número, então não há "nota certa" para comparar.

### A descoberta curiosa (vai para a tarefa e)

Com pessoas **conhecidas** (S1) o modelo foi **pior**, quase no chute. Por quê?

O modelo aprendeu a reconhecer **quem** está na sala (acerta a pessoa 78% das
vezes) melhor do que **o que** ela está fazendo. Cada corpo deixa uma
"assinatura" no sinal. No S1, o raciocínio dele vira: *"esse sinal parece do
Fulano; já vi o Fulano agachando, então é agachamento"* — mas a ação certa é
justamente uma que ele nunca viu o Fulano fazer. Com pessoas novas (S2) ele não
conhece ninguém e é obrigado a olhar o movimento.

Isso não é erro do código: é um achado real sobre o sinal de WiFi, e é
exatamente o tipo de coisa que a tarefa (e) pede para explicar.

As matrizes de confusão (quais ações o modelo confunde com quais) estão em
`artifacts/figuras/confusao_*.png`. Exemplo — SVM com pessoas novas:

![Matriz de confusão do SVM no S2](../artifacts/figuras/confusao_s2_svm_j30_p30_seed0.png)

Linhas = ação certa, colunas = palpite do modelo. Quanto mais escura a
diagonal, mais ele acerta. Repare na coluna **A01** ("alongar e relaxar"):
muitas ações são confundidas com ela.

### Experimento: apagar a "assinatura" da pessoa (ML-02)

Se o modelo reconhece **quem** está na sala, a ideia natural é apagar essa
assinatura antes de treinar. Testamos duas formas de **normalizar cada
gravação sozinha** (para cada antena e frequência, olhando só aquela gravação):

- **média:** tira o "nível de fundo" do sinal (subtrai a média);
- **z-score:** tira o nível de fundo **e** iguala o tamanho das oscilações
  (subtrai a média e divide pelo desvio).

Para saber se a assinatura sumiu mesmo, medimos também quanto os mesmos números
ainda **identificam a pessoa** (acaso: 1 em 40 = 2,5%).

Random Forest, seed 0, acerto por pedaço de 3 s (por gravação inteira entre
parênteses):

| Normalização | Pessoas novas (S2) | Pessoas conhecidas (S1) | Identifica a pessoa |
|---|---|---|---|
| nenhuma (padrão) | **14,5%** (15,3%) | 5,2% (5,2%) | **77,7%** |
| média | 15,7% (15,3%) | 7,3% (6,3%) | 60,5% |
| z-score | 10,0% (13,9%) | 9,9% (13,7%) | **16,5%** |

*Acaso: 3,7% para a ação, 2,5% para a pessoa. A coluna "identifica a pessoa" é
medida sempre na divisão S1 (no S2 as pessoas de teste nunca aparecem no
treino, então não dá para medir lá) e dá o mesmo valor nos JSON de S1 e de S2.
Os ICs 95% estão nos JSON `baseline_*_rf_j30_p30_seed0[_n<modo>].json`.
Todos os números vêm de **uma única seed (0)**.*

**Leitura em linguagem simples:**

- **A assinatura sai, sim.** Com z-score, o modelo deixa de reconhecer a pessoa
  (77% → 17%). Boa parte da "identidade" estava no nível e na escala do sinal
  de cada gravação.
- **Mas, com pessoas novas (S2), isso não ajuda.** A média fica igual dentro da
  margem de erro (14,5% → 15,7%). O z-score tende a piorar por pedaço
  (14,5% → 10,0%), mas as margens de erro se sobrepõem, e por gravação inteira
  fica parecido (15,3% → 13,9%). Uma explicação provável: junto com a
  assinatura vai embora informação útil sobre o movimento — o quanto o corpo
  mexe no sinal fica "achatado" quando dividimos pelo desvio.
- **Com pessoas conhecidas (S1), ajuda.** Sem a assinatura, o modelo para de
  "colar pela pessoa" e o S1 sobe de 5,2% para 9,9% (13,7% por gravação), ficando
  parecido com o S2. Isso confirma a explicação da seção anterior: o S1 baixo
  era culpa da assinatura.

**Decisão (20/10/2026, provisória por usar uma só seed):** fica registrado como
experimento. O padrão continua **sem normalização** — nenhuma das opções
melhorou o teste com pessoas novas, que é o que importa. A opção fica disponível para a análise (e) e para a rede neural:

```
.venv/Scripts/python.exe scripts/treinar_baseline.py --protocolo s2 --modelo rf --normalizacao zscore --diagnostico-sujeito
```

### Experimento: o "ritmo" do movimento — atributos de Doppler (ML-03)

Os números do classificador até aqui (média e desvio de cada antena) dizem
**quanto** o sinal mexeu, mas não **em que ritmo**. Um aceno e um agachamento
podem mexer "a mesma quantidade" no sinal, só que em velocidades diferentes. É
o mesmo efeito da sirene de ambulância que muda de tom quando passa (efeito
**Doppler**): o corpo em movimento faz o sinal de WiFi oscilar numa frequência
ligada à velocidade dele. Criamos três grupos novos de números para cada pedaço
de 3 s:

- **antenas:** a diferença entre as três antenas do receptor (cada uma "vê" o
  corpo de um ângulo um pouco diferente) — na força do sinal e no "atraso"
  (fase) entre elas, este guardado como cosseno e seno para não ter um "salto"
  artificial na volta do círculo;
- **dop_amp:** o "ritmo" com que a força do sinal oscila, de 0 a 50 vezes por
  segundo, em cada antena;
- **dop_cm:** o mesmo ritmo, mas comparando pares de antenas — o que também
  revela o **sentido** do movimento (aproximando ou afastando).

**Como escolhemos, sem colar.** Um teste rápido anterior já tinha indicado a
combinação vencedora, mas ela foi escolhida **olhando a nota da prova** (o
teste com pessoas novas) entre 8 opções — isso infla o resultado. Desta vez:

1. Separamos **4 pessoas do treino** (S04, S14, S24, S34 — uma de cada sala,
   nenhuma das 8 do teste) como um "simulado".
2. Treinamos cada uma das 5 combinações candidatas com as outras 28 pessoas e
   medimos no simulado, **3 vezes** (seeds 0, 1 e 2), para não depender da
   sorte de uma única rodada.
3. Ficou com a combinação de **maior acerto médio por gravação no simulado** —
   e essa escolha foi gravada **antes** de olhar o teste.
4. Só então a escolhida e o classificador antigo (`base`) foram treinados com
   as 32 pessoas do treino e avaliados **uma vez** no teste, com **5 seeds**.

Simulado (validação), Random Forest, média ± desvio de 3 seeds:

| Combinação | Nº de atributos | Acerto por pedaço de 3 s | Acerto por gravação |
|---|---|---|---|
| base (o de antes) | 1.368 | 12,2 ± 0,3% | 12,3 ± 0,5% |
| base + antenas | 3.078 | 11,8 ± 0,8% | 11,7 ± 0,5% |
| **base + dop_amp + dop_cm** | 1.821 | **14,9 ± 0,4%** | **17,6 ± 0,9%** |
| base + antenas + dop_amp + dop_cm | 3.531 | 13,0 ± 0,6% | 11,1 ± 0,0% |
| só dop_amp + dop_cm | 453 | 12,7 ± 1,2% | 15,1 ± 0,5% |

*O simulado tem só 4 pessoas (108 gravações), então cada número dele é
incerto: o IC 95% por gravação de uma seed tem ~13 pontos de largura (ex.:
escolhida, seed 0: 18,5%, IC [11,1–25,9]; base, seed 0: 12,0%, IC
[6,5–18,5]). O desvio entre seeds acima mede só a sorte do treino, não essa
incerteza.*

Teste com pessoas novas (S2), média ± desvio de 5 seeds (seeds 0–4):

| Combinação | Acerto por pedaço de 3 s | Acerto por gravação |
|---|---|---|
| **base + dop_amp + dop_cm** (escolhida) | **19,0 ± 0,3%** | **23,8 ± 1,1%** |
| base (o de antes) | 15,5 ± 0,7% | 18,1 ± 1,7% |

*Acaso: 3,7%. O IC 95% (bootstrap de gravações) de cada seed está em
`artifacts/resultados/selecao_atributos.json`; ele é largo — por exemplo, seed
0 da escolhida por gravação: 25,0%, IC [19,4–30,6]; do base: 15,3%, IC
[10,6–20,4]. O desvio entre seeds mede só a sorte do treino; o IC mede a
incerteza de ter só 216 gravações no teste.*

Como as duas são medidas **nas mesmas 216 gravações**, dá para comparar de
forma **pareada**: sorteamos gravações do teste com reposição (2.000 vezes),
sempre as mesmas para as duas, e olhamos a diferença escolhida − base.

| Diferença escolhida − base | Por pedaço de 3 s | Por gravação |
|---|---|---|
| seed 0 | +4,4 pp [+2,1; +7,0] | +9,7 pp [+4,6; +14,8] |
| seed 1 | +3,0 pp [+0,6; +5,3] | +5,6 pp [+0,5; +10,6] |
| seed 2 | +3,5 pp [+0,8; +6,1] | +3,7 pp [−1,4; +8,8] |
| seed 3 | +4,1 pp [+1,4; +6,7] | +6,0 pp [+0,5; +11,1] |
| seed 4 | +2,7 pp [+0,5; +4,9] | +3,2 pp [−1,9; +8,3] |
| **média das 5 seeds** | **+3,5 pp [+1,7; +5,5]** | **+5,6 pp [+1,6; +9,5]** |

*pp = pontos percentuais; entre colchetes, o IC 95% do bootstrap pareado. Na
média das seeds, a diferença foi ≤ 0 em nenhuma das 2.000 reamostras por
pedaço e em 0,35% delas por gravação.*

**Leitura em linguagem simples:**

- **O ritmo do movimento ajuda.** Com os atributos de Doppler, o acerto com
  pessoas novas sobe de ~15,5% para **~19%** por pedaço e de ~18% para
  **~24%** por gravação — de **5 a 6 vezes o chute**. Na comparação pareada, a
  melhora média (+3,5 pp por pedaço, +5,6 pp por gravação) tem IC inteiro
  acima de zero; por gravação, em 2 das 5 seeds o IC ainda inclui zero.
- **O Doppler sozinho já supera o base no simulado** (15,1% contra 12,3% por
  gravação; 12,7% contra 12,2% por pedaço), com só 453 números contra 1.368.
  Mas junto com o base fica melhor ainda: o "quanto mexeu" e o "em que ritmo"
  se completam.
- **A diferença entre antenas não ajudou.** Somada só ao base, fica parecida
  com ele (11,8% contra 12,2% por pedaço; 11,7% contra 12,3% por gravação —
  dentro da incerteza do simulado). A queda clara é quando entra **junto do
  Doppler**: 14,9% → 13,0% por pedaço e 17,6% → 11,1% por gravação. Uma
  explicação provável: ela quase dobra o número de atributos com informação
  que muda de pessoa para pessoa (a posição do corpo em relação às antenas), e
  a Random Forest passa a usá-la no lugar do Doppler.
- **Cuidado com o tamanho da melhora:** o teste tem só 8 pessoas, e o simulado
  só 4.

**Limitações que ficam registradas:**

- **A escolha reduz o viés, mas não o elimina.** As 5 candidatas vieram de um
  protótipo que **já tinha comparado combinações no teste**. Escolher entre
  elas pelo simulado evita escolher "a que foi melhor na prova", mas a lista de
  candidatas em si já foi influenciada pela prova.
- **Só Random Forest, só S2 com as 27 ações (P3).** O script de seleção não
  testa SVM, o S1 nem os protocolos P1/P2.
- **O eixo de frequência supõe amostragem uniforme a 100 Hz** (10 amostras por
  quadro de 100 ms). O MM-Fi não informa o instante de cada amostra; se o
  intervalo real variar, os "ritmos" medidos ficam borrados ou deslocados.

**Como rodar:**

```
.venv/Scripts/python.exe scripts/selecionar_atributos.py
.venv/Scripts/python.exe scripts/treinar_baseline.py --protocolo s2 --modelo rf --atributos base,dop_amp,dop_cm
```

Na primeira vez, os atributos de Doppler são calculados a partir do dataset e
guardados em `artifacts/cache_atributos/` (~6 min medidos, ~80 MB). O padrão do
`treinar_baseline.py` continua `base`, igual ao de antes.

---

## 4. O que falta fazer

1. ~~Tirar a "assinatura" da pessoa do sinal~~ — testado (ML-02, acima): a
   assinatura sai, mas o resultado com pessoas novas não melhora. Fica como
   opção (`--normalizacao`) para a rede neural testar de novo.
2. ~~Dar ao classificador o "ritmo" do movimento~~ — feito (ML-03, acima):
   com Doppler, o S2 sobe de ~15,5% para ~19% por pedaço (~24% por gravação),
   com a combinação escolhida no simulado, sem olhar o teste.
3. **Rede neural** — um modelo mais esperto, inspirado no artigo do Wi-Pose
   (camadas convolucionais + recorrentes), comparado contra o classificador
   simples acima. Cabe num notebook sem GPU se for enxuta.
4. **Terminar a análise (e):** quais ações se confundem (ex.: chutar com a
   perna esquerda × direita), se a fase limpa ajuda, quais antenas e
   frequências atrapalham.
5. **Terminar a tarefa (d):** comparar técnicas de limpeza do sinal.
6. **Escrever a revisão dos artigos (a).**
7. **Juntar tudo no relatório.**

**Em uma frase:** o trabalho é ensinar o computador a "ver" movimentos pelo
WiFi; já temos um modelo simples funcionando, 5 a 6 vezes melhor que o chute, e
agora vamos melhorá-lo com a rede neural e explicar os erros.

---

## 5. Como rodar o classificador

Com o ambiente e o dataset configurados (ver o [README](../README.md)):

```
.venv/Scripts/python.exe scripts/treinar_baseline.py --protocolo s2 --modelo rf
```

<sub>Linux/macOS: `.venv/bin/python` no lugar de `.venv/Scripts/python.exe`.</sub>

Na primeira vez ele monta uma base consolidada em `artifacts/cache/` (~3–4 min,
~8 GB em disco). Depois, cada treino leva de **1 a 3 minutos** num notebook
comum, sem GPU. As figuras saem em `artifacts/figuras/` e os números completos
em `artifacts/resultados/`.
