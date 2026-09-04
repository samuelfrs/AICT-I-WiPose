# Diretório do Dataset Wi-Pose

## Como instalar

Baixe o `Wi-Pose.rar` (~1,5 GB compactado, **2,4 GB extraído**) e extraia de modo
que fiquem existindo estes dois diretórios:

```
dataset/Wi-Pose/Train/    132.847 arquivos .mat
dataset/Wi-Pose/Test/      33.753 arquivos .mat
```

O código procura exatamente esse caminho. Se preferir guardar o dataset em outro
lugar — **recomendado, se este repositório estiver dentro de uma pasta
sincronizada como OneDrive ou Dropbox** —, extraia onde quiser e aponte a
variável de ambiente:

```cmd
set WIPOSE_DATASET=C:\datasets\wipose\Wi-Pose
```

São 166 mil arquivos pequenos: numa pasta sincronizada, o cliente de nuvem tenta
subir todos eles, o que trava a máquina e consome a cota. Vale a pena tirar da
sincronização.

Para conferir se deu certo:

```
.venv/Scripts/python.exe scripts/inspecionar.py
```

### Download

- **Repositório oficial:** [NjtechCVLab/Wi-PoseDataset](https://github.com/NjtechCVLab/Wi-PoseDataset)
- **Google Drive:** [link](https://drive.google.com/file/d/1WL6bJ-rSVdsclRt9RFc0l5hhtXYmfNg9/view?usp=sharing)
- **Baidu Netdisk:** [link](https://pan.baidu.com/s/14TbkIRPQN1DktDir9N2Y3A) (senha: `wqeq`)

---

## ⚠️ O README oficial está desatualizado

O repositório oficial descreve **três** variáveis dentro de cada `.mat`
(`csi_serial`, `jointsVector`, `jointsMatrix`). **O arquivo distribuído não é
assim.** Verificamos abrindo os arquivos:

| O README oficial diz | O arquivo realmente tem |
| --- | --- |
| `csi_serial`, forma `5 × 30 × 3 × 3` | **`CSI`**, forma `(3, 3, 30, 5)` ao ler com `h5py` |
| `jointsVector`, `18 × 4` no formato `(x, y, c, c)` | **`SkeletonPoints`**, vetor de 54 em blocos `[18 y][18 x][18 c]` |
| `jointsMatrix` | **não existe** |

Três diferenças importantes, todas confirmadas no dado:

1. **Os arquivos são MATLAB v7.3 (HDF5).** `scipy.io.loadmat` **não abre** — é
   preciso usar `h5py`.
2. **O CSI é amplitude real, não número complexo.** O dataset **não traz a fase**
   do sinal.
3. **No esqueleto, `y` vem antes de `x`** — ao contrário do que "(x, y, c)"
   sugere. Confirmado anatomicamente: a distância pescoço→quadril (vertical numa
   pessoa em pé) é maior no primeiro bloco em 99,4% dos casos.

**Não escreva código contra os nomes do README oficial.** Use
[`src/dados.py`](../src/dados.py), que já trata tudo isso e devolve o CSI como
`(9, 30, 5)` e as juntas como `(18, 3)` na ordem `(x, y, confiança)`.

Detalhes completos em [`docs/analise-inicial.md`](../docs/analise-inicial.md).

---

Os arquivos do dataset **não são versionados** (veja a `.gitignore`). Cada
integrante da equipe baixa a própria cópia.
