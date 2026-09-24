# PROMETHEUS 17/09/2026 — cena do café com os NOSSOS objetos.
#
# Copiada da `base_scene_pickplace_redblock.py` deles, trocando o cubo vermelho pelo copo e
# acrescentando o coador. Os dois USDs saíram dos OBJ que já usávamos no MuJoCo
# (`unitree-g1-mujoco/assets/copo_texturizado.obj` e `coador.obj`), convertidos com o
# `scripts/tools/convert_mesh.py` do IsaacLab — ver `~/converte_objetos_cafe.sh`.
#
# Por que `convexDecomposition` na colisão dos dois: ambos são CÔNCAVOS. Com `convexHull` a
# cavidade do copo e a do coador desaparecem, e nada entra dentro deles — o robô encostaria numa
# casca sólida.
#
# O copo se chama `object` de propósito: as funções de recompensa e de término da tarefa deles
# referenciam `SceneEntityCfg("object")`, então manter o nome faz a maquinaria existente valer
# para ele sem alteração. O coador entra como `object2`.
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
import os

from tasks.common_scene.pose_cafe import pose_mundo

project_root = os.environ.get("PROJECT_ROOT")

# A pose dos dois objetos vem de `~/cena_cafe.json` (ver `pose_cafe.py`), nao de numeros fixos
# aqui: assim da para mover o coador com `~/move_coador.sh` sem editar codigo, e o
# `tools/cena_cafe_vivo.py` reaplica a mesma pose ao vivo.
POSE_COPO, GIRO_COPO = pose_mundo("copo")
POSE_COADOR, GIRO_COADOR = pose_mundo("coador")

# NOTA 17/09: `UsdFileCfg` não aceita `physics_material` (é argumento dos spawners de forma
# primitiva, tipo `CuboidCfg`). O atrito destes dois objetos vem do próprio USD. Se a mão
# escorregar na hora de segurar, o caminho é aplicar um `RigidBodyMaterialCfg` ao prim depois
# de criado, não aqui no spawn.


# Tampo da mesa: o cubo vermelho deles nasce em z=0.84 com 6 cm de aresta, logo a base dele fica
# em 0.81. `pose_cafe.TAMPO` guarda esse número e é o padrão de altura dos dois objetos.
# O `tools/cena_cafe_vivo.py` MEDE a caixa da mesa e imprime no log do simulador na primeira
# passada, para conferir o valor de verdade em vez de deduzi-lo do cubo.


@configclass
class TableCafeSceneCfg(InteractiveSceneCfg):
    """Sala e mesa da Unitree, com o nosso copo e o nosso coador em cima."""

    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse_digital_twin/small_warehouse_digital_twin.usd",
        ),
    )

    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-4.3, -4.2, -0.2], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/table_with_yellowbox.usd",
        ),
    )

    # O COPO — é o que o robô pega. Mesma posição em que o cubo vermelho deles nascia, para
    # reaproveitar a faixa de alcance que já sabemos que funciona.
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=POSE_COPO, rot=GIRO_COPO),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/cafe/copo.usd",
            # A malha do copo já está em metros: 0.106 x 0.092 x 0.080 m. Escala 1.
            # A ROTAÇÃO NÃO É IDENTIDADE: a malha é Y-para-cima e o IsaacLab é Z-para-cima —
            # com rot=[1,0,0,0] o copo nascia DEITADO. `pose_cafe.ENDIREITA` faz o +90° em X.
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
        ),
    )

    # O COADOR — o destino. Fica parado e um pouco à esquerda do copo, dentro do alcance do braço.
    object2 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=POSE_COADOR, rot=GIRO_COADOR),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/cafe/coador.usd",
            # A malha do coador vem 5x grande (0.548 x 0.534 x 0.999 m). O nosso
            # `scene_43dof.xml` do MuJoCo usa scale=0.195, que dá 10.7 x 10.4 x 19.5 cm.
            scale=(0.195, 0.195, 0.195),
            # CINEMÁTICO: o coador é MÓVEL DE PROPÓSITO (o robô não o pega), mas a física não
            # mexe nele. Antes disso ele nascia 10 cm acima do tampo, caía, TOMBAVA e rolava 25 cm
            # — medido no `rt/sim_state`: parava em (-4.09, -4.25) de lado. Cinemático ele fica
            # exatamente onde for posto, colide com a mão e sobrevive ao `reset_object_self`.
            # Quem o move é `~/move_coador.sh`, que reescreve `~/cena_cafe.json`.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
        ),
    )
