# Onde cada coisa mora neste repositório

Escrito em 21/09/2026, depois de uma arrumação: havia duas cópias do mesmo repositório da
Unitree, uma delas dentro do `lerobot-ext/`, e dez arquivos de log soltos. Este documento existe
para a próxima pessoa (ou a próxima sessão) não repetir isso.

## A regra de ouro

**Código de terceiros entra como SUBMÓDULO na raiz. Nada de terceiros entra no `lerobot-ext/`.**

```
prometheus-vla/
├── lerobot/                  submódulo — o LeRobot de onde partimos
├── unitree_sdk2_python/      submódulo — SDK da Unitree
├── unifolm-wma/              submódulo — world-model-action da Unitree
├── unifolm-wla/              submódulo — o WLA, trazido em 21/09
│
├── lerobot-ext/              ★ O NOSSO CÓDIGO. É aqui que se trabalha.
├── unitree-g1-mujoco/        o nosso simulador MuJoCo (cenas, XML, ponte DDS)
├── train/                    ferramentas de dataset e treino independentes do LeRobot
├── assets/  visualization/  voz/
└── docs/                     documentação de AMBIENTE e instalação (este arquivo)
```

Por que submódulo e não cópia: uma cópia congela a versão sem dizer qual, não dá para atualizar,
e polui o `git status` com milhares de arquivos que não são nossos. O submódulo registra o commit
exato e se atualiza com um comando:

```bash
git submodule update --init --recursive        # trazer todos
git submodule update --init unifolm-wla        # trazer só um
```

**A nossa leitura do código de terceiros vira documento em `lerobot-ext/docs/`**, não comentário
dentro do repositório deles — que seria perdido na próxima atualização. Exemplos:
`UNIFOLM_WLA.md`, `UNIFOLM_WMA.md`, `UNIFOLM_APRENDIZADOS.md`, `DATASETS_EXTERNOS.md`.

## Dentro do `lerobot-ext/`

| pasta | o que é |
|---|---|
| `maquinas/athena/` | lançadores da A100 de treino. Cada linha deles é uma armadilha que já custou tempo — ver o README de lá |
| `maquinas/pgx/` | código específico desta GB10: MuJoCo, ponte com o IsaacLab, remendos de ARM |
| `pontes/wma/` | ponte com o UnifoLM-WMA-0 |
| `pontes/unifolm-vla/` | ponte com o UnifoLM-VLA-0: a FK que valida as 23 dims e o laço do MuJoCo |
| `docs/` | **a documentação do projeto**, com índice em `docs/README.md`. Todo aprendizado que custou caro vira um arquivo aqui. |
| `config/train/` | receitas de treino em YAML, uma por corrida, com o raciocínio no cabeçalho |
| `policies/` | as nossas políticas, uma pasta por modelo: `pi0_depth`, `act_depth`, `fastwam_depth`, `openvla_depth`, `unifolm_wla`. Cada uma é a NOSSA adaptação — o original fica no submódulo, para dar `diff` contra |
| `robot/`, `teleop/` | driver do G1 + Dex3 e teleoperação |
| `train/` | o trainer (`policies/pi0_depth/run_train.py` é o que roda de verdade) |
| `athena/` | lançadores da máquina de treino (A100) |
| `pgx/` | código específico da GB10/PGX, incluindo a ponte com o IsaacLab |
| `unifolm/` | a nossa ponte com os modelos UnifoLM (FK, IK, cliente HTTP) |
| `assets/` | URDF e malhas do G1 |
| `meu_dataset/` | datasets — **ignorado pelo git**, tem dezenas de GB |
| `logs/` | saída de execução — **ignorada pelo git** |
| `resultados/` | figuras e tabelas de avaliação |

## O que NÃO entra no git

Já coberto pelo `.gitignore` da raiz, mas vale saber por quê:

| padrão | motivo |
|---|---|
| `*/meu_dataset/`, `lerobot-ext/train_output/` | dezenas de GB de vídeo e checkpoint |
| `logs/`, `*.log` | saída de execução. Dez logs soltos na raiz do `lerobot-ext` foram para `logs/` em 21/09 |
| `*.antes_*`, `*.original` | cópias feitas à mão antes de editar. Servem para desfazer DURANTE a sessão; quem guarda histórico é o git |
| `*.pt`, `*.pth`, `*.tar.gz` | pesos e pacotes |
| `g1_29_model_cache*.pkl` | cache do Pinocchio, específico da versão instalada |

## Convenções que a gente segue

**Lançador em vez de comando decorado.** Toda corrida que precisa de variável de ambiente para
funcionar ganha um `.sh` com o motivo escrito dentro. O `grava_pega.sh` existe porque
`OMP_NUM_THREADS=1` foi esquecido uma vez e custou uma corrida de 30 horas virar 10.

**O porquê fica junto do código.** Os cabeçalhos longos nos YAML e nos scripts não são enfeite:
são o registro do que foi medido e do que já falhou. Um número medido vale mais que uma
explicação plausível — e quando houver medição, ela vem com a data.

**Documento de sessão quando a sessão descobre algo.** `docs/SESSAO_AAAA-MM-DD.md`, com o que
ficou rodando, o que ficou pendente e as armadilhas encontradas.
