# Guia do Projeto — comece por aqui

> ### 📌 Documento histórico — primeira fase do projeto
>
> Esta análise é sobre o dataset **Wi-Pose**, que foi **abandonado em 10/09/2026**
> justamente por causa das limitações descritas aqui: anotação 2D e ausência da
> fase do sinal. O projeto migrou para o **MM-Fi**.
>
> **Para o estado atual, veja [`analise-mmfi.md`](analise-mmfi.md).**
>
> Mantido no repositório porque documenta *por que* a troca foi feita — o que é
> material da tarefa (a), revisão bibliográfica.


Este documento explica o projeto do zero, sem pressupor que você já saiba de nada.
Se você quer a análise técnica com todos os detalhes, riscos metodológicos e
referências, ela está em [`analise-inicial.md`](analise-inicial.md).

---

## 1. A ideia, em linguagem simples

Quando existe um roteador WiFi numa sala, o sinal dele não viaja só em linha reta
até o seu celular. Ele **espalha**: bate nas paredes, nos móveis — e no seu corpo.
Cada vez que você levanta um braço ou atravessa a sala, você muda o caminho que
aquelas ondas percorrem.

O roteador **mede** essas mudanças o tempo todo, porque precisa disso para
funcionar. Essa medição tem nome: **CSI** (*Channel State Information*, ou
Informação de Estado do Canal). É uma tabela de números que descreve como o sinal
chegou ao receptor.

A ideia do projeto sai daí:

> Se o seu corpo deforma o sinal, então o sinal carrega informação sobre o seu corpo.
> Dá para pegar esses números e adivinhar o que a pessoa está fazendo — **sem câmera**.

**Por que alguém iria querer isso:**

- **Privacidade.** Monitorar se um idoso que mora sozinho caiu, sem instalar câmera na casa dele.
- **Nada para vestir.** Diferente de pulseira ou smartwatch, a pessoa não carrega sensor nenhum.
- **Funciona no escuro e através de paredes.** WiFi atravessa obstáculo, câmera não.
- **O hardware já existe.** Todo mundo tem roteador em casa.

---

## 2. O que é o CSI, um pouco mais a fundo

O WiFi não transmite num único tom de rádio. Ele divide a banda em **30
subportadoras** — 30 "canais" de frequência ligeiramente diferentes, usados ao
mesmo tempo. É como uma orquestra tocando 30 notas simultâneas em vez de uma só.

Isso importa porque **cada frequência é afetada de um jeito diferente** pelos
obstáculos do ambiente. Um braço que se move pode atrapalhar bastante a
subportadora 7 e quase nada a subportadora 22. Ter 30 medições em vez de 1 dá
muito mais informação sobre o que está acontecendo na sala.

Além disso, o equipamento tem **3 antenas transmissoras e 3 receptoras**, o que dá
**9 pares** possíveis (TX1→RX1, TX1→RX2, ... TX3→RX3). Cada par "enxerga" a sala
de um ângulo diferente.

Então cada medição de CSI é um bloco de números com esta forma:

```
9 pares de antena  ×  30 subportadoras  ×  5 instantes no tempo
```

Os 5 instantes cobrem 50 milissegundos. É esse bloco que alimenta o modelo.

---

## 3. Os dados que temos

Um grupo de pesquisa chinês (Nanjing Tech) montou o seguinte cenário:

- Um transmissor e um receptor WiFi de 5 GHz, com 3 antenas cada.
- **12 voluntários** de alturas e pesos diferentes.
- **12 ações**, cada uma repetida **10 vezes** por cada pessoa:

  | | | | |
  |---|---|---|---|
  | `bend` (curvar) | `circle` (girar) | `crouch` (agachar) | `jump` (pular) |
  | `pull` (puxar) | `push` (empurrar) | `run` (correr) | `sitdown` (sentar) |
  | `standup` (levantar) | `throw` (arremessar) | `walk` (andar) | `wave` (acenar) |

- Enquanto a pessoa se movia, uma **câmera filmava**. Um programa de visão
  computacional (AlphaPose) analisou o vídeo e marcou onde estavam **18 pontos do
  corpo** (cabeça, ombros, cotovelos, punhos, quadril, joelhos, tornozelos) em cada
  quadro.

O resultado são **166.600 arquivos `.mat`**, cada um contendo:

| Variável | O que é |
|---|---|
| `CSI` | O bloco de sinal descrito acima: 9 × 30 × 5 |
| `SkeletonPoints` | Onde estavam as 18 juntas naquele instante, mais um número de confiança |

A segunda parte é o **gabarito**. É ela que permite treinar: você mostra o sinal ao
modelo, ele chuta, e você compara com o que a câmera viu.

**O rótulo da ação está no nome do arquivo**, não dentro dele:

```
walk_042-frame017.mat
 ↑     ↑        ↑
ação   sequência  quadro
```

---

## 4. Cinco coisas que descobrimos abrindo os dados

Não são detalhes — cada uma muda o que dá para fazer. Todas foram verificadas
rodando `scripts/inspecionar.py` sobre o dataset real.

### 4.1. O dataset **não é** o dos artigos

O `tema.pdf` indica dois artigos (Wi-Pose e Winect) e um dataset chamado
"Wi-Pose". Acontece que **são coisas de grupos diferentes que têm o mesmo nome**.
O artigo do MobiCom é da universidade de Buffalo; o dataset é de Nanjing.

Consequência prática: o artigo usa um sistema profissional de captura de
movimento (VICON) que dá posições **3D** com precisão de 2 mm. O nosso dataset usa
uma câmera comum, e as posições são **2D** — coordenadas na imagem, achatadas.

**Não dá para fazer reconstrução 3D com esses dados.** Isso precisa estar claro no
relatório.

### 4.2. Não existe fase no sinal — só amplitude

Um sinal de rádio tem duas partes: **amplitude** (o quão forte ele chegou) e
**fase** (em que ponto do ciclo da onda ele chegou). A fase é mais sensível a
movimento, e as tarefas (b) e (d) do `tema.pdf` pedem explicitamente para trabalhar
com ela.

Só que o dataset distribui **apenas a amplitude**. Verificamos: todos os valores
são reais e não-negativos, em 300 arquivos testados. A fase foi descartada por
quem montou o dataset.

Isso não inviabiliza nada, mas **rescopa duas tarefas** — ver a seção 6.

### 4.3. Cerca de 10% dos quadros são repetidos

O dataset "estica" cada repetição copiando o último quadro várias vezes até chegar
a um comprimento alvo. **10,6% dos pacotes são cópias exatas do anterior**
(medido sobre 180 sequências sorteadas), quase sempre a partir do quadro ~105.

São ~17,6 mil pacotes sem informação nova. E é uma armadilha séria: se você dividir
treino e teste sorteando pacotes aleatoriamente, **cópias idênticas vão parar dos
dois lados**, e a acurácia vai parecer excelente sem significar nada.

### 4.4. O tamanho real do dataset é 1.440, não 166.600

Parece que temos 166.600 exemplos, mas eles não são independentes. Pacotes
consecutivos vêm da mesma repetição da mesma pessoa, com 50 ms de diferença — são
quase idênticos.

A unidade que realmente conta é a **sequência** (uma repetição inteira). São
**12 ações × 120 sequências = 1.440**.

Por que isso importa: com ~24 sequências por classe no teste, a margem de erro de
qualquer número que a gente reportar é grande. **Todo resultado precisa vir com
intervalo de confiança**, senão a gente vai discutir diferenças que são ruído.

### 4.5. O split oficial é bem-feito

Boa notícia: o dataset já vem dividido em `Train/` e `Test/`, e a divisão respeita
as sequências — 1.152 de um lado, 288 do outro, **zero em comum**. Dá para confiar
nele como ponto de partida.

O que ele **não** faz é separar por pessoa. E dá para ver isso na estrutura: o
`Test` leva sistematicamente as sequências de id 9 e 10 de cada bloco de 10, nas
12 ações. Se os ids foram atribuídos como 12 voluntários × 10 repetições — que é
a leitura natural —, então cada pessoa contribui com 8 repetições no treino e 2
no teste. As mesmas 12 pessoas aparecem nos dois lados. Então nosso resultado mede *"o modelo reconhece
uma repetição nova de alguém que ele já viu"*, e não *"o modelo funciona com uma
pessoa nova"*. Isso é limitação do dataset e precisa estar escrito no relatório.

---

## 5. O que a disciplina cobra

O `tema.pdf` lista cinco tarefas. Elas formam uma cadeia:

```
     (a) revisão bibliográfica  ──────────────────►  (corre em paralelo)

     (b) carregar e plotar  →  (d) filtrar  →  (c) classificar  →  (e) analisar
```

| | Tarefa | O que é, na prática |
|---|---|---|
| **a** | Revisão bibliográfica | Ler os dois artigos e escrever comparando: um usa rede neural e só funciona com movimentos treinados; o outro usa matemática de sinal e lida com movimento livre. |
| **b** | Manipulação dos dados | Escrever código que abre os `.mat` e gera gráficos do sinal. **Já feito** — ver seção 7. |
| **c** | Classificação | O coração do trabalho: um modelo que recebe o sinal e responde qual das 12 ações é. |
| **d** | Pré-processamento | Aplicar filtros para limpar o ruído do sinal, e medir se ajuda ou atrapalha. |
| **e** | Análise crítica | Olhar onde o modelo erra e por quê. Quais ações ele confunde? Quais subportadoras atrapalham? |

**Um aviso sobre a tarefa (e):** o objetivo do projeto **não é acertar 99%**. É
entender o comportamento do sistema. Um resultado modesto mas bem medido e bem
explicado vale mais do que um número alto que ninguém sabe de onde veio. Aliás,
número alto demais em problema desse tipo geralmente significa erro de metodologia.

---

## 6. Como as descobertas mudam o plano

| Tarefa | Como estava no enunciado | Como fica |
|---|---|---|
| **b** | "plotar amplitude **e fase**" | Só amplitude — e documentar a ausência da fase como achado. Em compensação, dá para explorar melhor a dimensão de subportadoras e antenas. |
| **c** | "classificar as ações" | Sem mudança. É a tarefa central e está totalmente viável. |
| **d** | "remover **offsets de fase** e ruídos" | Sem fase para calibrar. Vira: filtragem temporal (Hampel, Butterworth, wavelet) e filtragem espacial (combinar os 9 pares de antena, PCA). |
| **e** | "matrizes de confusão, subportadoras ruidosas" | Sem mudança, e ganhou material extra: a discrepância dataset × artigo e os quadros duplicados são achados críticos legítimos. |

**Nenhuma tarefa foi perdida.** Duas foram rescopadas, e isso é conteúdo de
relatório, não problema.

---

## 7. O que já está pronto no repositório

```
src/dados.py               O carregador. Única porta de entrada para o dataset.
scripts/inspecionar.py     Retrato do dataset: contagens, estrutura, duplicados.
scripts/plotar_amplitude.py  Gera as figuras de amplitude (tarefa b).
tests/test_dados.py        Testes do carregador, rodam sem o dataset completo.
```

### Como rodar

Ambiente, só na primeira vez (comandos de **cmd.exe**):

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

<sub>Git Bash: `source .venv/Scripts/activate`. PowerShell: `.venv\Scripts\Activate.ps1`.</sub>

Se o dataset não estiver em `dataset/Wi-Pose`, aponte o caminho:

```cmd
set WIPOSE_DATASET=C:\datasets\wipose\Wi-Pose
```

<sub>Git Bash: `export WIPOSE_DATASET=...` — PowerShell: `$env:WIPOSE_DATASET="..."`.</sub>

Depois, os comandos do dia a dia (funcionam em qualquer shell):

```
.venv/Scripts/python.exe scripts/inspecionar.py
.venv/Scripts/python.exe scripts/plotar_amplitude.py --acao walk --sequencia 1
.venv/Scripts/python.exe scripts/plotar_amplitude.py --acao jump --sequencia 5
.venv/Scripts/python.exe -m pytest tests/ -v
```

### O contrato do carregador

Tudo passa por `src/dados.py`. A função principal:

```python
from src.dados import carregar_pacote

csi, juntas = carregar_pacote("dataset/Wi-Pose/Train/walk_001-frame001.mat")
# csi    -> (9, 30, 5)  float32 — amplitude
# juntas -> (18, 3)     float32 — (x, y, confiança)
```

**Duas regras que não podem ser quebradas:**

1. **Nunca fundir o eixo de antenas com o de subportadoras.** Um `reshape` errado
   mistura os dois e destrói a estrutura do sinal **sem dar erro nenhum**. A tarefa
   (e) depende de conseguir analisar subportadora por subportadora.
2. **Sempre agrupar por sequência ao dividir treino/teste.** Use o campo `grupo`
   que o loader devolve, com `GroupShuffleSplit` do scikit-learn. Sortear pacotes
   aleatoriamente vaza dados e infla o resultado.

---

## 8. Divisão do trabalho

A cadeia (b) → (d) → (c) → (e) é sequencial, então uma divisão ingênua deixaria
duas pessoas esperando. A saída é fixar o formato dos dados logo no início, para
que todo mundo trabalhe em paralelo.

| Pessoa | Frente | O que entrega |
|---|---|---|
| **P1** | Dados e sinal (`b`, `d`) | Melhorias no carregador, cache para não reler 166 mil arquivos, filtros temporais e espaciais, figuras. |
| **P2** | Modelagem (`c`) | Baseline trivial → SVM/Random Forest → CNN pequena. Sempre com semente fixa e divisão por sequência. |
| **P3** | Bibliografia e análise (`a`, `e`) | Revisão dos dois artigos; depois matrizes de confusão, ablação de subportadoras, redação do relatório. |

P3 pode começar **hoje** — a tarefa (a) não depende de código nenhum.

---

## 9. Próximos passos

1. ~~Baixar o dataset~~ ✅
2. ~~Confirmar a estrutura dos arquivos~~ ✅
3. ~~Carregador e primeiras figuras~~ ✅
4. **Baseline de classificação** — começar pelo mais bobo possível (chutar sempre a
   classe mais comum, que acerta 8,5%) e depois um SVM sobre estatísticas simples
   da amplitude. O primeiro número honesto do projeto.
5. **Cache dos dados** — ler 166 mil arquivos toda vez é inviável; salvar uma vez
   em formato compacto.
6. **Filtros** (tarefa d) — medindo cada um contra o baseline sem filtro.
7. **Análise dos erros** (tarefa e) — matriz de confusão, ablação de subportadoras.
8. **Revisão bibliográfica escrita** (tarefa a) — pode começar já, em paralelo.

---

## 10. Glossário rápido

| Termo | Significado |
|---|---|
| **CSI** | *Channel State Information*. A medição de como o sinal WiFi chegou ao receptor. |
| **Subportadora** | Uma das 30 frequências que o WiFi usa simultaneamente. Cada uma reage de forma diferente ao ambiente. |
| **Amplitude** | O quão forte o sinal chegou. É o que temos. |
| **Fase** | Em que ponto do ciclo da onda o sinal chegou. Mais sensível a movimento — **não temos**. |
| **AlphaPose** | Programa de visão computacional que detecta pontos do corpo em imagens. Gerou nosso gabarito. |
| **Pacote** | Um arquivo `.mat`. Corresponde a 50 ms de sinal e um quadro de vídeo. |
| **Sequência** | Uma repetição completa de uma ação. Varia bastante: de 50 a 216 pacotes (2,5 a 10,8 s), mediana 110. |
| **Baseline** | Um modelo burro de referência. Se o seu modelo não bate o baseline, ele não aprendeu nada. |
| **Vazamento** | Quando informação do teste escapa para o treino. Faz o resultado parecer ótimo sem ser. |
| **Matriz de confusão** | Tabela que mostra quais classes o modelo troca por quais. O produto principal da tarefa (e). |
