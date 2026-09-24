#!/usr/bin/env bash
# Monta uma CÓPIA do ambiente `lerobot/unitree-g1-mujoco` com a cena das duas mesas e da
# caixa azul, para o teste do `xiaopeng-wu/pi05_unitree_g1`.
#
# Por que copiar em vez de editar direto:
#  * o original vive no cache do HuggingFace, que é descartável — qualquer novo download
#    desfaz a edição, e pior, sem avisar;
#  * o `sim/model_config.py` deles FIXA o nome da cena por efetuador (`dex1` →
#    `assets/scene_33dof.xml`) e sobrescreve o `ROBOT_SCENE` do config.yaml. Então a cena nova
#    entra com o nome que ele espera, dentro da cópia;
#  * o `base_sim.py` resolve a cena como `Path(__file__).parent.parent / ROBOT_SCENE`, ou seja,
#    relativa à pasta do ambiente. Copiar a árvore inteira é o que faz isso funcionar.
#
# O original fica guardado ao lado, como `assets/scene_33dof.original.xml`.
#
#   bash maquinas/pgx/prepara_cena_caixa.sh            # cria ~/g1_mujoco_caixa
#   DESTINO=/outro/lugar bash maquinas/pgx/prepara_cena_caixa.sh
set -euo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESTINO="${DESTINO:-$HOME/g1_mujoco_caixa}"
CENA="$AQUI/cena_caixa_azul.xml"

[ -f "$CENA" ] || { echo "❌ não achei $CENA"; exit 1; }

ORIGEM=$(ls -d "$HOME"/.cache/huggingface/hub/models--lerobot--unitree-g1-mujoco/snapshots/*/ 2>/dev/null | head -1)
if [ -z "$ORIGEM" ]; then
    echo "❌ o ambiente não está no cache. Baixe antes:"
    echo "     hf download lerobot/unitree-g1-mujoco"
    exit 1
fi

echo "== origem:  $ORIGEM"
echo "== destino: $DESTINO"
mkdir -p "$DESTINO"
# -L resolve os symlinks do cache do HF: o destino tem que ter os arquivos de verdade,
# senão apagar o cache leva a cópia junto.
rsync -aL --delete "$ORIGEM" "$DESTINO/"

if [ ! -f "$DESTINO/assets/scene_33dof.original.xml" ]; then
    cp "$DESTINO/assets/scene_33dof.xml" "$DESTINO/assets/scene_33dof.original.xml"
fi
cp "$CENA" "$DESTINO/assets/scene_33dof.xml"

echo "== cena instalada:"
grep -cE "<body name=\"(mesa_a|mesa_b|caixa_azul)\"" "$DESTINO/assets/scene_33dof.xml" \
    | sed "s/^/   corpos novos encontrados: /"
python3 - "$DESTINO" <<'EOF'
import sys
from pathlib import Path
destino = Path(sys.argv[1])
try:
    import mujoco
except ImportError:
    print("   (mujoco não importável aqui — a validação do XML fica para a execução)")
    raise SystemExit
m = mujoco.MjModel.from_xml_path(str(destino / "assets" / "scene_33dof.xml"))
nomes = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
cams = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
print(f"   ✅ XML compila: {m.nbody} corpos, {m.ncam} câmeras")
print(f"   mesas/caixa: {[n for n in nomes if n in ('mesa_a', 'mesa_b', 'caixa_azul')]}")
print(f"   câmeras: {cams}")
EOF

echo "== pronto. Use com:"
echo "   python maquinas/pgx/roda_pi05_g1_sim.py --env-dir $DESTINO --passos 0 --tempo-real --v-web=8088"
