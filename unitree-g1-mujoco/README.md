# MuJoCo Sim for Unitree G1

Standalone MuJoCo physics simulator for the Unitree G1 robot, adapted from gr00t_wbc. Currently supports G1_29dof.

set use joystick to 1 to control the robot

## Cenários disponíveis

O cenário padrão (`assets/scene_43dof.xml`) é uma adaptação do ambiente de
`gpt6-astra-direct/runtime/g1-cup-grasp`: mesa de trabalho, caneca, marcadores
coloridos e câmeras externas. Ele usa o G1/Dex3 e a infraestrutura DDS/depth
deste simulador; nenhum código da política Astra é necessário.

O modelo também disponibiliza as câmeras `left_wrist_camera` e
`right_wrist_camera`. Para publicá-las, passe seus nomes no argumento
`cameras` de `run_sim.main`; os streams padrão de cabeça e depth permanecem
inalterados.

Cenários alternativos podem ser selecionados em `config.yaml`, alterando
`ROBOT_SCENE`:

- `assets/scene_43dof.xml`: cenário Astra adaptado (padrão);
- `assets/scene_astra_gonogo.xml`: variante mínima go/no-go;
- `assets/scene_43dof_smart_ia_legacy.xml`: cenário anterior da
  `feat/smart-ia`.

Os metadados originais dos marcadores e da geometria da caneca ficam em
`assets/astra_grasp/`.

## Estação de café

A cena padrão inclui chaleira, coador, pote, tampa e scoop texturizados.
São objetos dinâmicos com massa, gravidade, atrito e juntas livres; podem
ser empurrados ou agarrados por contato. As colisões usam caixas aproximadas
para manter os modelos escaneados estáveis a 250 Hz. Malhas e texturas ficam em
`assets/meshes/coffee_workspace/`; posições ficam em `assets/coffee_workspace.xml`.

Execute como antes, na raiz do repositório:

```bash
conda activate g1_new
python unitree-g1-mujoco/run_sim.py
```

Prévia da cena: `coffee-workspace-preview.jpg`.
