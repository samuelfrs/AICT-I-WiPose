"""
Roda as duas versões da rede neural (v1 e v2) **com o código original, sem alterações** (ML-04).

Cada versão é extraída do seu commit num ``git worktree`` descartável, criado
**fora** do repositório principal (pasta temporária do sistema). O script original
roda com o diretório de trabalho no worktree — é lá que ele grava figuras,
caches e o seu ``artifacts/resultados_rede_neural.json`` —, com ``MMFI_DATASET``
apontando para o dataset local. Depois de cada execução, o JSON gerado é copiado
para ``artifacts/resultados/rede_neural_v<versao>_exec<i>.json`` deste repositório,
acrescido de ``origem`` (commit, comando, se a seed é controlada, se o código
do worktree mudou). Os worktrees são removidos no fim, com sucesso ou erro.

Versões
-------
- **v1** (``98a1f49``, ``main``): rede multi-head por **quadro**; o script não
  tem seed. Comando registrado no README da versão (lá: ~16 min por execução em CPU;
  aqui, dividindo a CPU com outro treino, foram medidos ~107 min).
- **v2** (``68d96a9``, ``origin/rede-neural-sml``): rede residual sobre
  **janelas de 3 s** com voto suave por sequência; ``--seed`` controla torch e
  numpy. Padrões do script: 25 épocas, batch 64 (5–12 min medidos por execução,
  incluindo ~3 min de cache de janelas no primeiro uso do worktree). O script
  usa ``Optional`` sem importá-lo e falha ao carregar (``NameError``); em vez de
  editar o arquivo, ele é rodado via ``runpy`` com ``typing.Optional`` posto em
  ``builtins`` — o contorno fica registrado em ``origem.contorno_execucao``.

Nos dois, o número reportado é o da **última época** (sem escolha de época),
mas o script avalia o teste a cada época — ver scripts/comparar_modelos.py.

Guarda do repositório principal: antes e depois (num ``finally``, inclusive
com Ctrl-C), o script tira um instantâneo ``(caminho, sha1)`` dos arquivos
modificados ou não rastreados fora de ``artifacts/resultados/`` e avisa se algo
mudou. Caminhos ignorados pelo git (dataset, caches, ``.venv``) **não** são
cobertos.

Uso:
    .venv/Scripts/python.exe scripts/rodar_rede_neural.py
    .venv/Scripts/python.exe scripts/rodar_rede_neural.py --versoes 2 --execucoes 3
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROJETO))

DESTINO_PADRAO = RAIZ_PROJETO / "artifacts" / "resultados"
JSON_ORIGINAL = Path("artifacts") / "resultados_rede_neural.json"
PREFIXO_PERMITIDO = "artifacts/resultados/"

VERSOES: Dict[str, Dict] = {
    "1": {
        "commit": "98a1f49a64ab3439f41c3d16f19f244face10b92",
        "ref": "main (98a1f49)",
        "argumentos": lambda i: ["--split", "S2", "--protocolo", "P3", "--epocas", "15",
                                 "--batch-size", "32", "--stride", "20"],
        "seed_controlada": False,
        "nota_seed": "o script não tem --seed; cada execução tem inicialização e "
                     "embaralhamento diferentes",
    },
    "2": {
        "commit": "68d96a9e20ae5e4d1fb3272eb9e283fce3962a1e",
        "ref": "origin/rede-neural-sml (68d96a9)",
        "argumentos": lambda i: ["--split", "S2", "--protocolo", "P3", "--seed", str(i)],
        "seed_controlada": True,
        "nota_seed": "--seed fixa torch.manual_seed e np.random.seed",
        # O script usa ``Optional`` numa anotação sem importá-lo: NameError ao
        # carregar. Em vez de editar o arquivo original, o nome é posto em builtins
        # antes de rodá-lo (runpy); nenhuma linha do código original muda.
        "injetar_builtins": ("Optional",),
    },
}

# Equivale a ``python scripts/treinar_rede.py ...``: sys.path[0] = <worktree>/scripts
# (o que o interpretador faz ao rodar um arquivo), mais os nomes injetados.
LANCADOR = (
    "import builtins, os, runpy, sys, typing\n"
    "for nome in sys.argv.pop(1).split(','):\n"
    "    setattr(builtins, nome, getattr(typing, nome))\n"
    "sys.argv[0] = 'scripts/treinar_rede.py'\n"
    "sys.path[0] = os.path.join(os.getcwd(), 'scripts')\n"
    "runpy.run_path('scripts/treinar_rede.py', run_name='__main__')\n"
)


def _git(*args: str, cwd: Path = RAIZ_PROJETO, checar: bool = True) -> subprocess.CompletedProcess:
    resultado = subprocess.run(["git", *args], cwd=cwd, capture_output=True)
    if checar and resultado.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} falhou: "
                           f"{resultado.stderr.decode(errors='replace').strip()}")
    return resultado


def instantaneo_repositorio(raiz: Path = RAIZ_PROJETO) -> Dict[str, Optional[str]]:
    """``{caminho: sha1 do conteúdo}`` dos arquivos modificados (rastreados) ou não
    rastreados fora de ``artifacts/resultados/``; ``None`` = apagado.

    Compara conteúdo, não só o código de status do git: um arquivo já
    modificado que muda de novo também aparece. Ignorados pelo git não entram.
    """
    saida = _git("ls-files", "-z", "--modified", "--others", "--exclude-standard",
                 cwd=raiz).stdout.decode()
    estado: Dict[str, Optional[str]] = {}
    for caminho in sorted({c for c in saida.split("\0") if c}):
        if caminho.startswith(PREFIXO_PERMITIDO):
            continue
        arquivo = raiz / caminho
        estado[caminho] = (hashlib.sha1(arquivo.read_bytes()).hexdigest()
                           if arquivo.is_file() else None)
    return estado


def alterados_fora_de_resultados(antes: Dict[str, Optional[str]],
                                 depois: Dict[str, Optional[str]]) -> List[str]:
    """Caminhos cujo conteúdo mudou e que não estão em ``artifacts/resultados/``."""
    mudados = {c for c in set(antes) | set(depois) if antes.get(c, "") != depois.get(c, "")}
    return sorted(c for c in mudados if not c.startswith(PREFIXO_PERMITIDO))


def criar_worktree(commit: str, pasta_temp: Optional[str]) -> Path:
    """Worktree destacado de ``commit`` numa pasta nova fora do repositório.

    Se ``git worktree add`` falhar, a pasta criada é apagada.
    """
    pasta = Path(tempfile.mkdtemp(prefix=f"wipose_rede_neural_{commit[:7]}_", dir=pasta_temp))
    try:
        if RAIZ_PROJETO in pasta.resolve().parents:
            raise ValueError(f"O worktree precisa ficar fora do repositório: {pasta}")
        _git("worktree", "add", "--detach", "--force", str(pasta), commit)
    except BaseException:
        shutil.rmtree(pasta, ignore_errors=True)
        _git("worktree", "prune", checar=False)
        raise
    return pasta


def remover_worktree(pasta: Path) -> None:
    _git("worktree", "remove", "--force", str(pasta), checar=False)
    shutil.rmtree(pasta, ignore_errors=True)
    _git("worktree", "prune", checar=False)


def codigo_alterado(worktree: Path) -> bool:
    """O código rastreado do worktree difere do commit? (``git diff --quiet``).

    ``artifacts/`` fica de fora: o próprio script regrava ali figuras e
    o JSON que vêm versionados no commit.
    """
    resultado = _git("diff", "--quiet", "HEAD", "--", ".", ":(exclude)artifacts",
                     cwd=worktree, checar=False)
    if resultado.returncode not in (0, 1):
        raise RuntimeError(f"git diff no worktree falhou: {resultado.stderr.decode().strip()}")
    return resultado.returncode == 1


def _versao_torch() -> Optional[str]:
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def executar(versao: str, execucao: int, worktree: Path, env: Dict[str, str],
             destino: Path) -> Path:
    """Uma execução do script original; devolve o JSON copiado."""
    config = VERSOES[versao]
    argumentos = config["argumentos"](execucao)
    comando_texto = "python scripts/treinar_rede.py " + " ".join(argumentos)
    injetar = config.get("injetar_builtins", ())
    if injetar:
        comando = [sys.executable, "-c", LANCADOR, ",".join(injetar), *argumentos]
    else:
        comando = [sys.executable, "scripts/treinar_rede.py", *argumentos]

    # O commit original já traz um resultados_rede_neural.json: apagar antes evita
    # copiar o antigo se a execução falhar sem gravar.
    json_original = worktree / JSON_ORIGINAL
    json_original.unlink(missing_ok=True)

    print(f"\n--> v{versao} execução {execucao}: {comando_texto}  (cwd={worktree})", flush=True)
    inicio = time.time()
    processo = subprocess.run(comando, cwd=worktree, env=env)
    duracao = time.time() - inicio
    if processo.returncode != 0:
        raise RuntimeError(f"v{versao} execução {execucao} saiu com código {processo.returncode}.")
    if not json_original.is_file():
        raise RuntimeError(f"v{versao} execução {execucao} não gravou {JSON_ORIGINAL}.")
    alterado = codigo_alterado(worktree)
    if alterado:
        print(f"AVISO: o código do worktree v{versao} difere do commit {config['commit'][:7]}.")

    dados = json.loads(json_original.read_text(encoding="utf-8"))
    dados["origem"] = {
        "versao": f"v{versao}",
        "commit": config["commit"],
        "ref": config["ref"],
        "comando": comando_texto,
        "seed_controlada": config["seed_controlada"],
        "seed": execucao if config["seed_controlada"] else None,
        "nota_seed": config["nota_seed"],
        "execucao": execucao,
        "codigo_alterado": alterado,
        "codigo_alterado_verificado_com": "git diff --quiet HEAD -- . ':(exclude)artifacts'",
        "contorno_execucao": (
            f"{', '.join(injetar)} de typing posto em builtins antes de rodar o script "
            "(via runpy), porque ele o usa sem importar; o arquivo original não foi editado"
        ) if injetar else None,
        "torch": _versao_torch(),
        "python": sys.version.split()[0],
        "tempo_total_s": round(duracao, 1),
        "data": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    destino.mkdir(parents=True, exist_ok=True)
    saida = destino / f"rede_neural_v{versao}_exec{execucao}.json"
    saida.write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"--> Copiado para {saida}", flush=True)
    return saida


def ambiente_filho(raiz_dataset: Optional[str] = None) -> Dict[str, str]:
    from src.mmfi import caminho_dataset

    env = dict(os.environ)
    env["MMFI_DATASET"] = str(caminho_dataset(raiz_dataset))
    env.pop("MPLCONFIGDIR", None)          # o script original aponta para o próprio worktree
    env.pop("PYTHONPATH", None)            # nada de fora pode sombrear o src/ do worktree
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def rodar(versoes: Sequence[str], execucoes: Sequence[int], destino: Path,
          pasta_temp: Optional[str] = None, raiz_dataset: Optional[str] = None) -> List[Path]:
    env = ambiente_filho(raiz_dataset)
    print(f"--> MMFI_DATASET={env['MMFI_DATASET']}")

    gravados = []
    for versao in versoes:
        worktree = criar_worktree(VERSOES[versao]["commit"], pasta_temp)
        try:
            print(f"--> Worktree v{versao} ({VERSOES[versao]['ref']}) em {worktree}")
            for i in execucoes:
                gravados.append(executar(versao, i, worktree, env, destino))
        finally:
            remover_worktree(worktree)
            print(f"--> Worktree v{versao} removido", flush=True)
    return gravados


def main() -> int:
    parser = argparse.ArgumentParser(description="Roda as redes neurais v1 e v2 sem alterar o código (ML-04)")
    parser.add_argument("--versoes", default="1,2", help="Versões separadas por vírgula (1, 2)")
    parser.add_argument("--execucoes", type=int, default=3, help="Execuções por versão")
    parser.add_argument("--primeira", type=int, default=0,
                        help="Índice da primeira execução (= seed na v2)")
    parser.add_argument("--pasta-temp", default=None,
                        help="Onde criar os worktrees (padrão: pasta temporária do sistema)")
    parser.add_argument("--raiz", default=None, help="Caminho do MM-Fi (padrão: MMFI_DATASET)")
    args = parser.parse_args()

    versoes = [v.strip().lstrip("v") for v in args.versoes.split(",") if v.strip()]
    invalidas = [v for v in versoes if v not in VERSOES]
    if not versoes or invalidas or args.execucoes < 1:
        print(f"ERRO: versões válidas: {', '.join(VERSOES)}; execuções ≥ 1.")
        return 1

    antes = instantaneo_repositorio()
    codigo = 0
    try:
        gravados = rodar(versoes, range(args.primeira, args.primeira + args.execucoes),
                         DESTINO_PADRAO, args.pasta_temp, args.raiz)
        print(f"\n--> {len(gravados)} resultado(s) em {DESTINO_PADRAO}")
    except (RuntimeError, ValueError, OSError) as erro:
        print(f"ERRO: {erro}")
        codigo = 1
    finally:
        # Roda também com Ctrl-C: o repositório principal não pode ter mudado.
        alterados = alterados_fora_de_resultados(antes, instantaneo_repositorio())
        if alterados:
            print(f"AVISO: arquivos do repositório fora de {PREFIXO_PERMITIDO} mudaram "
                  f"durante a execução (inclusive edições feitas em paralelo): {alterados}")
            codigo = 1
    return codigo


if __name__ == "__main__":
    sys.exit(main())
