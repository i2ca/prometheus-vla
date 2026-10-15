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
para manter os modelos escaneados estáveis a 250 Hz, exceto o coador, que usa
base, haste e funil vazado separados para permitir colocar a xícara embaixo.
O coador tem 24 cm de altura (malha ampliada 1.5×) e foi girado 180° para
deixar a haste atrás e a entrada voltada ao robô. Pote e tampa foram afastados
da base para manter espaço de manobra. Malhas e texturas ficam em
`assets/meshes/coffee_workspace/`; posições ficam em `assets/coffee_workspace.xml`.

Execute como antes, na raiz do repositório:

```bash
conda activate g1_new
python unitree-g1-mujoco/run_sim.py
```

Capturas e prévias são artefatos locais: salve-as em `~/Pictures`, fora do repositório.

### Espaço para teleoperação

A mesa tem 80 × 140 cm, altura de 75 cm e borda frontal em X=0.20 m.
Os objetos estão separados na região próxima ao robô, com espaço livre nas
laterais e no fundo para manobras. Continuam sujeitos à física normal:
podem ser agarrados, empurrados, tombar e cair. O coador foi aproximado em
11 cm no eixo frontal; o pote foi deslocado para manter espaço entre eles.

O modo de treino `RECOVER_FALLEN_OBJECTS: true` em `config.yaml` devolve
copo, chaleira, coador, pote, tampa e scoop à pose original quando a base
cai 15 cm abaixo do tampo. Zera velocidade e forças do objeto recuperado.
Se a posição original estiver ocupada, procura uma posição próxima livre
sobre a mesa; se não houver espaço, aguarda e tenta novamente. Não altera
objetos levantados acima da mesa. Desative a opção para quedas sem recuperação.
A recuperação é local e não consome API.

A câmera RGB `head_camera` tem abertura vertical de 60° (antes 43°), mantendo
posição, orientação e resolução. É um ajuste para facilitar a operação;
não representa uma calibração da câmera real. Depth/IR mantêm sua abertura
de 58°; portanto RGB e depth têm campos de visão diferentes.

Para teleoperar, na raiz do repositório, use `./lerobot-ext/start_teleop_vr.sh --sim`.
Em outro terminal com `conda activate g1_new`, abra a visão da cabeça:

```bash
python lerobot-ext/mirror_cam.py --window SIM_HEAD --scale 1.5
```

O launcher inicia a simulação; não execute `run_sim.py` simultaneamente.
Na simulação, X/voz congela braços e mãos no último comando, ignorando
squeeze e trigger durante a pausa. Aperte X novamente para retomar o controle.
Os ganhos reais de impedância não são injetados no caminho de mãos simulado;
o atrito das juntas dos dedos foi reduzido para permitir retorno à pose aberta.
A HUD mostra conclusão em verde somente ao terminar o objetivo inteiro;
etapas intermediárias não exibem rótulo de andamento.

A iluminação usa duas luzes fixas difusas, sem sombras projetadas e sem
headlight ligado à câmera. Isso elimina o pontilhado de sombra e sua
cintilação com pequenos movimentos. As luzes não alteram física ou depth.
