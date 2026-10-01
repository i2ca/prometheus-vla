"""Observação do estado atual para o orquestrador: uma imagem e os números.

Carrega um estado aceito (`continuation.npz` + `report.json`), renderiza a
câmera pedida e devolve em JSON o que o modelo precisa saber sem olhar o
simulador por dentro: pó no filtro, temperatura, se o aquecedor está ligado,
líquido na xícara e derramamento.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
# Três câmeras, casando com o conjunto instalado no robô real: a da cabeça
# (presa ao torso_link) e as duas de punho (presas aos wrist_yaw). Para a pega
# da alça a de punho importa: vista do tronco, o próprio braço oclui a alça.
ap.add_argument('--cameras', nargs='+',
                default=['head_camera', 'left_wrist_camera', 'right_wrist_camera'])
ap.add_argument('--width', type=int, default=960)
ap.add_argument('--height', type=int, default=720)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

report = json.loads((a.source / 'report.json').read_text())
m = mujoco.MjModel.from_xml_path(report['scene'])
d = mujoco.MjData(m)
ck = np.load(a.source / 'continuation.npz')
mujoco.mj_setState(m, d, ck['integration'], mujoco.mjtState.mjSTATE_INTEGRATION)
mujoco.mj_forward(m, d)

renderer = mujoco.Renderer(m, height=a.height, width=a.width)


def salvar(imagem, caminho):
    try:
        import imageio.v3 as iio
        iio.imwrite(caminho, imagem)
    except ImportError:
        import PIL.Image
        PIL.Image.fromarray(imagem).save(caminho)


arquivos = []
for cam in a.cameras:
    renderer.update_scene(d, camera=cam)
    nome = f'{cam}.png'
    salvar(renderer.render(), a.out / nome)
    arquivos.append(nome)
# a primeira continua como view.png, para não quebrar quem já apontava para ela
import shutil
shutil.copyfile(a.out / arquivos[0], a.out / 'view.png')

estado = {'source': str(a.source), 'cameras': a.cameras, 'arquivos': arquivos}

brew_path = a.source / 'brew-state.json'
if brew_path.exists():
    brew = json.loads(brew_path.read_text())
    g = brew.get('grounds', {})
    h = brew.get('heater', {})
    estado['po_no_filtro_g'] = round(g.get('filter_g', 0.), 4)
    estado['po_na_colher_g'] = round(g.get('spoon_g', 0.), 4)
    estado['po_derramado_g'] = round(g.get('spilled_g', 0.), 4)
    estado['temperatura_C'] = round(h.get('temperature_C', 0.), 2)
    estado['aquecedor_ligado'] = bool(h.get('on', False))
    estado['ultimo_evento_aquecedor'] = h.get('last_event')

liquid_path = a.source / 'liquid-state.json'
if liquid_path.exists():
    w = json.loads(liquid_path.read_text())
    estado['agua_na_chaleira_ml'] = round(w.get('source_ml', 0.), 2)
    estado['agua_no_coador_ml'] = round(w.get('filter_ml', 0.), 2)
    estado['agua_na_xicara_ml'] = round(w.get('receiver_ml', 0.), 2)
    estado['agua_derramada_ml'] = round(w.get('spilled_ml', 0.), 2)

# posição dos objetos livres, para o relatório (não vai no prompt do modelo)
corpos = {}
for nome in ['chaleira', 'coador', 'copo', 'pote', 'scoop', 'base_eletrica']:
    try:
        corpos[nome] = [round(float(v), 4) for v in d.xpos[m.body(nome).id]]
    except Exception:
        pass
estado['posicoes_objetos'] = corpos

(a.out / 'observation.json').write_text(json.dumps(estado, indent=2, ensure_ascii=False))
print(json.dumps({k: v for k, v in estado.items() if k != 'posicoes_objetos'},
                 indent=2, ensure_ascii=False), flush=True)
