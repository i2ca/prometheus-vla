# mrwlker/

Todo o trabalho do Miguel (mrwlker) na branch `lerobot`, reunido numa pasta só para
entrar na `main` **sem mudar nenhum arquivo fora dela**. Cópia da `lerobot` no commit
`942ce36` (2026-09-24).

A pasta repete o layout da raiz do repositório (`lerobot-ext/`, `unitree-g1-mujoco/`,
`docs/`, `voz/`...), então os caminhos relativos do código (`parents[1] / "unitree-g1-mujoco"`
e parecidos) continuam valendo aqui dentro. A versão do README da raiz que estava na
`lerobot` está em `README_raiz.md`.

## O que é arquivo e o que é link

- **Arquivo de verdade**: tudo que foi criado ou modificado por Miguel/mrwlker na `lerobot`,
  inclusive os arquivos que outras pessoas também mexeram.
- **Link simbólico para a raiz**: o que está igual à `main` (ex.: `lerobot-ext/assets`,
  as malhas do G1 em `unitree-g1-mujoco/assets`). Nada é duplicado e a árvore fica completa:
  `cd mrwlker/lerobot-ext` e o código importa e roda como na `lerobot`.

Ficaram de fora:
- arquivos só de outras pessoas: `config/train/dataset_luiz-*.yaml`, a versão da `lerobot`
  de `config/train/pi05depth_get_the_cup_depth.yaml` (aqui é link para a da `main`) e
  `realsense_server.py`;
- lixo local: `MUJOCO_LOG.TXT`, `scene_43dof.xml.bak`, `.claude/settings.local.json`;
- submódulos (git não os aninha numa subpasta sem mexer no `.gitmodules` da raiz):

| submódulo | url | commit na `lerobot` |
|---|---|---|
| `lerobot` | github.com/Breno-de-Angelo/lerobot | d52a098 (a `main` aponta para fe0f359) |
| `unifolm-wla` | github.com/unitreerobotics/unifolm-wla | f33d0e7 + `lerobot-ext/remendos/unifolm-wla.patch` |
| `unifolm-wma` | github.com/unitreerobotics/unifolm-world-model-action | 3e198de |

`mrwlker/lerobot`, `mrwlker/unifolm-wla` e `mrwlker/unifolm-wma` são links para essas
pastas **na raiz**: o código que sobe até a raiz (`RAIZ / "unifolm-wla"`) as acha por aqui.
Na `main` o `unifolm-wla` e o `unifolm-wma` não são submódulos: clone-os na raiz nos
commits acima (e aplique `lerobot-ext/remendos/unifolm-wla.patch`).

Caminhos absolutos nos scripts apontam para `~/DEV/prometheus-vla/mrwlker/...`; as
máquinas (athena, pgx) precisam estar na `main` para achar os scripts em `maquinas/`.

## Por onde começar

| pasta | o quê |
|---|---|
| `lerobot-ext/pontes/cabine/` | a cabine: painel web com tarefa, câmeras e motores (WLA, pi05) |
| `lerobot-ext/pontes/groot/` | GR00T N1.6/N1.7 da NVIDIA no simulador deles com a cabine, cena do café, SONIC |
| `lerobot-ext/pontes/groot/n17/` | N1.7-ApplePnP-V1 em TensorRT no GB10 (README próprio) |
| `lerobot-ext/remendos/` | patches dos repositórios de fora (Isaac-GR00T, robosuite, WBC, unifolm-wla) |
| `lerobot-ext/policies/unifolm_wla/`, `config/wla/` | WLA (ER-Flow + DiT) e as receitas de treino |
| `lerobot-ext/maquinas/` | scripts das máquinas athena e pgx |
| `lerobot-ext/docs/` | anotações de sessão, passo a passo de treino, schemas |
| `docs/INSTALL.md`, `docs/ORGANIZACAO.md` | instalação e organização do repositório |
