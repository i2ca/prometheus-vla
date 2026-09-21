# Cena do café no IsaacLab — copo e coador nossos

Cópia dos arquivos que vivem fora deste repositório, em `~/DEV/unitree_sim_isaaclab/`
(o simulador da Unitree) e em `~`. O original é lá; isto aqui é para não perdermos.

| arquivo | onde ele mora de verdade |
|---|---|
| `pose_cafe.py` | `tasks/common_scene/pose_cafe.py` |
| `base_scene_cafe.py` | `tasks/common_scene/base_scene_cafe.py` |
| `cena_cafe_vivo.py` | `tools/cena_cafe_vivo.py` |
| `pickplace_cafe_g1_29dof_dex3_joint_env_cfg.py` | `tasks/g1_tasks/pick_place_cafe_g1_29dof_dex3/` |
| `poe_cafe.py` | `~/testes_g1/poe_cafe.py` |
| `move_coador.sh`, `move_copo.sh` | `~` |

O `sim_main.py` deles tem duas linhas nossas (import + `cena_cafe_vivo.passo(env)` antes do
`controller.step()`). O original está preservado em `sim_main.py.original`.

## Como usar

```bash
TAREFA=Isaac-PickPlace-Cafe-G129-Dex3-Joint COM_JANELA=1 ~/sobe_sim_dex3.sh

~/move_coador.sh              # mostra onde estão os dois
~/move_coador.sh 32 18        # 32 cm à frente do robô, 18 cm à esquerda dele
~/move_copo.sh 33 -5          # o copo onde nascia o cubo vermelho deles
~/move_coador.sh 30 15 --giro 45 --altura 79.4
```

As medidas são **em centímetros, vistas do robô**: `frente` e `lado` (positivo = esquerda).
O que vai para o disco (`~/cena_cafe.json`) é coordenada de mundo em metros, que é o que o
simulador consome. O simulador aplica a mudança em menos de meio segundo, **sem reiniciar**.

## Três coisas medidas nesta cena

1. **Tampo da mesa em z = 0,794** — medido na caixa do prim `PackingTable`
   (x −5,537..−3,063, y −4,581..−3,819, z −0,200..0,794) e confirmado de forma independente
   pelo coador que tinha tombado: o ponto mais baixo da malha dele parou em 0,7945.
   A mesa é grande: de 12 cm a 88 cm à frente do robô, e 1,1 m para cada lado.

2. **O copo nascia deitado** porque a malha `copo_texturizado.obj` é **Y-para-cima**
   (altura 9,2 cm no eixo Y, base em y=0) e o IsaacLab é Z-para-cima. O nosso MuJoCo já
   corrigia isso com `euler="1.5708 -1.5708 0"`; aqui a correção é o quaternião de +90° em X,
   em `pose_cafe.ENDIREITA`. A malha do coador já é Z-para-cima e não precisa de correção.

3. **O coador tombava e rolava 25 cm.** Nascia 10 cm acima do tampo, caía de lado e parava em
   (−4,09, −4,25). Agora é **cinemático** (`kinematic_enabled=True`): a física não mexe nele,
   ele colide com a mão, fica exatamente onde é posto e sobrevive ao `reset_object_self`.

## Detalhe que ainda incomoda

A colisão do **copo** (convexDecomposition) não coincide com a malha visual: quando ele estava
deitado, o visual afundava 3,9 cm no tampo. De pé ele assenta certo (origem na base, z = tampo),
então para esta tarefa está resolvido — mas se ele tombar durante uma tentativa, vai afundar na
mesa. Se isso atrapalhar, o caminho é converter o copo de novo com uma aproximação melhor
(`sdf` ou `convexDecomposition` com mais cascos) em `~/converte_objetos_cafe.sh`.
