# Cena do café no Isaac Lab, 21/09/2026

Porta a tarefa do café para o padrão do `unitree_sim_isaaclab`, que é o caminho
que a pesquisa do lab (`Prometheus VLA/04 - Pesquisa/2026-09-02`) aponta para o
robô real, porque fala o mesmo DDS.

## Escolha de versão, e por que as duas linhas não convivem

| repo | exige |
|---|---|
| `IsaacSim_GeminiRobotics` (os Franka) | Isaac Sim **6.x** (morre no 5.1 com `ModuleNotFoundError: isaacsim.core.experimental.utils.app`) |
| `unitree_sim_isaaclab` | Isaac Sim **4.5 / 5.x**, diz o README na linha 127 |

Não dá para ter as duas no mesmo ambiente. Fui pela linha da Unitree, então
"mesmo padrão do repo dos Franka" aqui significa mesma **arquitetura** (cena no
Isaac, bridge, agente escolhendo primitiva por function calling), não mesma
versão. O `three_robot_tower.py` continua rodando na `.venv-isaac6` da 232.

## O que foi feito

Ambiente: `unitree_rl_lab` na 232 (Isaac Sim 5.1 + Isaac Lab 2.3), que já tinha
tudo. Assets em `~/Prometheus/shared/assets_custom/cafe/`.

Tarefa `Isaac-PickPlace-Cafe-G129-Dex3-Joint`, em
`tasks/g1_tasks/pick_place_cafe_g1_29dof_dex3/`, no molde da `pick_place_copo`
do lab: subclasse não-destrutiva da tarefa do cilindro, reusando robô, as três
câmeras Dex3, sensor de contato, observações, ações e eventos. Sete objetos na
mesa em vez de um. `__init__.py` do índice tem backup em `.bak_precafe`.

## Erros meus, com a medição que os pegou

**Escala.** Converti dos GLB, que vêm normalizados para caixa unitária: a
chaleira saiu com 0,998 × 0,987 × 0,794 m. As escalas por eixo necessárias
divergiam até 9,7× dentro do mesmo objeto, sinal de que os eixos também estavam
trocados. Reconverti dos OBJ que o próprio MJCF usa: chaleira 0,1147 × 0,0913 ×
0,232 m, e `min_z = 0` em todos, o que resolve o assentamento de brinde.

**Espelhamento.** No MuJoCo o robô encara +x, então a direita é −y. Aqui ele
encara +y e a direita é +x. Eu tinha escrito `x_isaac = +y_mujoco`, que inverte
os lados: a colher, que é da mão direita, foi parar do lado esquerdo.

**Altura do tampo.** Deduzi 0,665 pela altura do cilindro da tarefa base. Medido
na cena, o tampo útil é **0,731** (a caixa do asset vai até 0,8829, mas isso é
estrutura superior). A colher caía no chão porque o layout começava a 7 cm da
borda; corrigido com um deslocamento de 8 cm em y que preserva a geometria.

**Folga de assentamento.** Com 2 mm a chaleira nascia encostando no topo da base
elétrica e o contato inicial a arremessava 10 cm. Passou para 2 cm de folga.

## Dependências que faltavam no ambiente

`teleimager` (precisa de `logging_mp==0.1.5`, porque a 0.2.5 renomeou
`get_logger` para `getLogger`, e de `aiortc`), `unitree_sdk2py` (que precisa do
binding Python do `cyclonedds` compilado contra a lib C do conda, com
`CYCLONEDDS_HOME` apontando para o prefixo do env), `rerun-sdk` e `onnxruntime`.

**Atenção:** o `rerun-sdk` puxa `pyarrow`, que sobe o numpy para 2.x, e o Isaac
Sim 5.1 é compilado contra numpy 1.x: isso derruba o simulador com falha de
segmentação. Fixar `numpy<2` depois de instalar. Este é um ambiente
compartilhado e eu mexi nele.

## O que esta cena não tem

O coador é malha rígida. No MuJoCo ele é flexcomp 2D de 65 vértices, e o Isaac
Lab não tem API de tecido, então **não há filtragem por pano aqui**. Água, pó e
calor são modelos laterais em Python no `coffee-cloth` e não foram portados: a
chaleira é um corpo rígido com a massa do conjunto cheio. E todos os limiares
validados (6 mm de metal quente, 1 mm de penetração, 0,3 N·m/rad do balancim)
foram medidos contra o modelo de contato do MuJoCo; o PhysX dará outros.

## Como rodar

```bash
ssh <usuario>@<host-gpu>
R=~/Prometheus/shared/unitree_sim_isaaclab
cd $R && PROJECT_ROOT=$R PYTHONPATH=$R ~/miniforge3/envs/unitree_rl_lab/bin/python \
  sim_main.py --task Isaac-PickPlace-Cafe-G129-Dex3-Joint --headless --enable_cameras

# verificação (prims + três câmeras), no espírito do --test do repo dos Franka
cd $R && PROJECT_ROOT=$R PYTHONPATH=$R ~/miniforge3/envs/unitree_rl_lab/bin/python \
  ~/Prometheus/shared/verifica_cafe.py
```
