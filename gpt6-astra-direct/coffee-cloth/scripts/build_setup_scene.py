"""Cena do cafe com os objetos do Luiz (GLB convertidos por convert_glb.py): chaleira eletrica na base com botao, coador de
pano no suporte com o copo (o mesmo copo oco do g1-cup-grasp) embaixo, pote de po com tampa e scoop. Base: cena
scene_grasp.xml do g1-cup-grasp (mesa a 0,75 m, marcadores, cameras 3x2, robo com base fixa). Visual = OBJ com textura;
colisao = pecas convexas do CoACD (grupo 3, invisiveis). Massas declaradas em MASS_KG. Claude (claude-fable-5-1).
Uso: ../g1-cup-grasp/.venv/bin/python scripts/build_setup_scene.py --out scene/setup-luiz-001.xml"""
import argparse, glob, json, math, os, shutil
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; G1 = ROOT.parent / "g1-cup-grasp"; MESH = ROOT / "assets" / "mesh"
MASS_KG = {"chaleira": 1.0, "pote": 0.5, "tampa": 0.10, "scoop": 0.03}          # declaradas, nao medidas
# posicao (x, y) na mesa e yaw (graus) de cada objeto; z e a mesa (0,75) ou o apoio
# mao direita em repouso ocupa x 0,06-0,38, y -0,26..-0,15, z 0,87-0,98; esquerda x<0,18, y 0,13-0,24 (medido). Nada dentro disso.
LAYOUT = {"chaleira": ((0.23, 0.10), 180.0), "coador": ((0.36, -0.16), -90.0), "pote": ((0.52, -0.10), 0.0), "scoop": ((0.50, 0.30), 90.0)}   # coador com a haste para +y: o copo entra por -x e a mao fica em -x/-y; a haste em +x (cena 013/016) ficava a 5 cm do copo e o indicador batia nela
SKIP_PARTS = {"coador": {"coador_col00", "coador_col30"},
               # as quatro pecas macicas da alca do pote (raio 75 mm). Cortar tambem col02, col06 e col17,
               # que invadem o vao com poucos vertices, abria buraco na parede: elas sao parede.
               "pote": {"pote_col04", "pote_col22", "pote_col19", "pote_col08"}}   # col00: casco da base anelar; col30: resto do saco antigo no meio do vao. O disco que substitui a base tem raio 34 mm, menor que a caneca (41): a chapa de 80 mm de diametro fica toda sob a xicara e um erro de colocacao nao deixa a xicara meio fora, travada na borda          # casco convexo da base anelar fecha o centro e o copo nao apoia; substituido por um disco
COPO_INICIAL = (0.26, -0.20, 90.0)   # None = copo ja sob o filtro; tupla = posicao inicial na mesa
BASE_R, BASE_H = 0.080, 0.015    # BAS-001: base de carregamento de 160 mm de diametro e 15 de altura (torre central de 40 x 15 nao modelada)                                   # base eletrica da chaleira (cilindro fixo)
ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
if a.out.exists(): raise FileExistsError(a.out)
tree = ET.parse(G1 / "scene" / "scene_grasp.xml"); root = tree.getroot(); root.set("model", "G1 cafe coado, setup do Luiz")
robot_path = a.out.with_name(a.out.stem + "-robot.xml").resolve()
robot = ET.parse(G1 / "scene" / "g1_grasp.xml"); robot.getroot().find("compiler").set("meshdir", str((G1 / "scene" / "meshes").resolve())); robot.write(robot_path, encoding="unicode")
root.find("include").set("file", str(robot_path))
asset = root.find("asset"); w = root.find("worldbody")
def quat_z(deg):
    h = math.radians(deg) / 2; return f"{math.cos(h):.6f} 0 0 {math.sin(h):.6f}"
def add_object(name, pos, yaw, free, mass=None):
    d = MESH / name; info = json.load(open(d / "info.json"))
    ET.SubElement(asset, "texture", name=f"{name}_tex", type="2d", file=str(d / f"{name}_tex.png"))
    ET.SubElement(asset, "material", name=f"{name}_mat", texture=f"{name}_tex", specular="0.3", shininess="0.3")
    ET.SubElement(asset, "mesh", name=f"{name}_visual", file=str(d / f"{name}_visual.obj"))
    # a lista vem do info.json, nao de glob: glob pegava pecas de geracoes antigas que sobraram no
    # diretorio e a cena ganhava colisao fantasma em escala errada
    parts = [str(d / f"{name}_col{i:02d}.obj") for i in range(info["collision_parts"])]
    faltando = [p for p in parts if not Path(p).exists()]
    if faltando:
        raise FileNotFoundError(f"{name}: info.json declara {info['collision_parts']} pecas mas faltam {faltando}")
    for p in parts: ET.SubElement(asset, "mesh", name=Path(p).stem, file=p)
    b = ET.SubElement(w, "body", name=name, pos=f"{pos[0]} {pos[1]} {pos[2]}", quat=quat_z(yaw))
    if free: ET.SubElement(b, "freejoint", name=f"{name}_livre")
    ET.SubElement(b, "geom", name=f"{name}_vis", type="mesh", mesh=f"{name}_visual", material=f"{name}_mat", contype="0", conaffinity="0", group="1", density="0")
    vol = None
    if mass is not None:
        import trimesh
        vol = sum(trimesh.load(p).volume for p in parts); dens = mass / vol
    for p in parts:
        if Path(p).stem in SKIP_PARTS.get(name, set()): continue
        g = ET.SubElement(b, "geom", name=Path(p).stem, type="mesh", mesh=Path(p).stem, group="3", rgba="0 0 0 0", friction="1 0.005 0.0001", condim="4")
        if mass is not None: g.set("density", f"{dens:.1f}")
    return info
notes = {}
# base eletrica + botao (fixos na mesa)
(x, y), yaw = LAYOUT["chaleira"]
base = ET.SubElement(w, "body", name="base_eletrica", pos=f"{x} {y} 0.75", quat=quat_z(-90.0))   # botao virado para +y: em -x ele cai a 8 cm do eixo do robo (botao-01) e em +x fica debaixo do bico da chaleira, que tem 102 mm de raio ali e encostava nele em 875 quadros (botao-02). Em +y ele ficava livre da chaleira mas o caminho da mao passava por cima do coador e do copo, 1501 quadros de contato (botao-05). Base no lado ESQUERDO e botao a 240 graus: o braco direito nao alcanca nada la embaixo sem cruzar o coador, entao o botao passa para o braco esquerdo, que sai do ombro oposto
ET.SubElement(base, "geom", name="base_eletrica_corpo", type="cylinder", size=f"{BASE_R} {BASE_H/2}", pos=f"0 0 {BASE_H/2}", rgba="0.15 0.15 0.16 1", friction="1 0.005 0.0001")
ET.SubElement(base, "geom", name="botao_chaleira", type="box", size="0.008 0.012 0.006", pos=f"{-BASE_R-0.008} 0 0.012", rgba="0.95 0.80 0.10 1")   # amarelo: vermelho confundia o marcador
ET.SubElement(base, "site", name="botao_chaleira_site", pos=f"{-BASE_R-0.016} 0 0.012", size="0.004", rgba="1 0 0 0.5")
notes["chaleira"] = add_object("chaleira", (x, y, 0.75 + BASE_H + 0.001), yaw, True, MASS_KG["chaleira"])
# pe da chaleira: o CoACD partiu o fundo e a chaleira apoiava so no lado do bico (tombava 14 graus); disco fino declarado.
# Raio 56 mm, medido na malha visual (p95 do raio e 55,8 na faixa de 0 a 5 mm de altura, e o JAR-001 cota 110 de
# diametro). Estava em 75, herdado de quando a jarra tinha o dobro da largura: era uma saia invisivel de 150 mm
ET.SubElement(w.find("body[@name='chaleira']"), "geom", name="chaleira_pe", type="cylinder", size="0.056 0.002", pos="0 0 0.002", group="3", rgba="0 0 0 0", friction="1 0.005 0.0001", density="300")
(x, y), yaw = LAYOUT["coador"]; notes["coador"] = add_object("coador", (x, y, 0.75), yaw, False)   # suporte fixo na mesa (declarado); yaw 180: haste do lado oposto ao robo, frente livre para o copo entrar
ET.SubElement(w.find("body[@name='coador']"), "geom", name="coador_base_disco", type="cylinder", size="0.040 0.0013", pos="0 0 0.0013", group="3", rgba="0 0 0 0", friction="1 0.005 0.0001")
# copo do g1-cup-grasp embaixo do coador, sobre a base do suporte (espessura medida no info do coador)
import trimesh, numpy as np
cv = trimesh.load(MESH / "coador" / "coador_visual.obj", process=False).vertices
sel = cv[(np.hypot(cv[:, 0], cv[:, 1]) < 0.03) & (cv[:, 2] < 0.06)]          # topo da base do suporte, onde o copo apoia
base_h = float(cv[(np.hypot(cv[:, 0], cv[:, 1]) < 0.035) & (cv[:, 2] < 0.02)][:, 2].max()); notes["coador"]["base_top_m"] = base_h   # topo da base do suporte (medido na malha real); notes["coador"]["base_visual_center_top_m"] = float(sel[:, 2].max()) if len(sel) else None
tip = cv[(np.hypot(cv[:, 0], cv[:, 1]) < 0.02) & (cv[:, 2] > 0.06)]; notes["coador"]["filter_tip_z_m"] = float(tip[:, 2].min()) if len(tip) else None
hx, hy = -0.041, 0.0                                  # haste no referencial do coador (medida na malha)
hth = math.radians(LAYOUT["coador"][1]); hwx, hwy = hx * math.cos(hth) - hy * math.sin(hth), hx * math.sin(hth) + hy * math.cos(hth)
away = np.array([-hwx, -hwy]); away /= np.linalg.norm(away)          # direcao oposta a haste
cx, cy = np.array([x, y]) + 0.016 * away          # 16 mm: a base do suporte tem o diametro da caneca e a haste sai da borda; com 8 mm o copo passava a 3 mm da haste. A ponta do filtro fica a 16 mm do eixo do copo (raio interno 37): o gotejo cai dentro
notes["coador"]["haste_world_xy"] = [round(x + hwx, 4), round(y + hwy, 4)]; notes["coador"]["copo_xy"] = [round(cx, 4), round(cy, 4)]
copo = w.find("body[@name='copo']")
if COPO_INICIAL is None:
    copo.set("pos", f"{cx} {cy} {0.75 + base_h + 0.001}"); copo.set("quat", quat_z(math.degrees(math.atan2(away[1], away[0]))))
else:
    # sequencia completa: o copo comeca na mesa e a etapa 1 o leva ate o filtro
    copo.set("pos", f"{COPO_INICIAL[0]} {COPO_INICIAL[1]} 0.7536"); copo.set("quat", quat_z(COPO_INICIAL[2]))
(x, y), yaw = LAYOUT["pote"]; notes["pote"] = add_object("pote", (x, y, 0.751), yaw, True, MASS_KG["pote"])
# PAC-001A: duas alcas tipo orelha, semicirculo de raio 15 mm, com furo. O CoACD preencheu o furo e
# a mao nao tinha onde entrar: nas 33 poses de aproximacao medidas, so o polegar alcancava a borda e
# sem contraposicao nao ha pinca. Recriadas a mao como arco de capsulas, que devolve o vao de ~23 mm.
pote_body = w.find("body[@name='pote']")
ALCA_R, ALCA_ESP, ALCA_ZC, ALCA_R0 = 0.015, 0.0035, 0.036, 0.060   # R0 a 60 e nao 55 mm: as pecas de parede do CoACD chegam a 61 mm e obstruiam o vao em 6,4 mm; 5 mm de afastamento declarado
for lado, ang0 in (("a", 0.0), ("b", math.pi)):
    ex = (math.cos(ang0), math.sin(ang0))
    pts = []
    for k in range(6):
        t = math.radians(-90 + 180 * k / 5)
        rr = ALCA_R0 + ALCA_R * math.cos(t); zz = ALCA_ZC + ALCA_R * math.sin(t)
        pts.append((rr * ex[0], rr * ex[1], zz))
    for k in range(5):
        p, q = pts[k], pts[k + 1]
        ET.SubElement(pote_body, "geom", name=f"pote_alca_{lado}{k}", type="capsule",
                      fromto=f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {q[0]:.5f} {q[1]:.5f} {q[2]:.5f}",
                      size=f"{ALCA_ESP}", group="3", rgba="0 0 0 0", friction="1 0.005 0.0001", density="800")
notes["pote"]["alcas"] = {"raio_semicirculo_m": ALCA_R, "espessura_m": ALCA_ESP, "z_centro_local_m": ALCA_ZC,
                          "vao_interno_mm": round((ALCA_R - ALCA_ESP) * 2000, 1)}
pote_h = notes["pote"]["extents_m"][2]; tampa_plug = 0.014 * notes["tampa"]["scale"] / 0.12 if False else 0.0
notes["tampa"] = add_object("tampa", (x, y, 0.751 + pote_h + 0.001), yaw, True, MASS_KG["tampa"])   # o plugue (r 6,2 cm) e mais largo que a boca (r 5,8): a tampa apoia na borda
(x, y), yaw = LAYOUT["scoop"]; notes["scoop"] = add_object("scoop", (x, y, 0.751), yaw, True, MASS_KG["scoop"])
# marcadores: com a camera da cabeca a 26 graus os de x=0,66 saem pelo topo do quadro e a chaleira tapa o ciano; reposicionados
MARKERS = {"red": (0.46, -0.40), "cyan": (0.64, 0.16), "blue": (0.62, 0.44), "magenta": (0.46, 0.40)}   # vermelho fica sob a mao em repouso (3 marcadores); ciano longe do aro do coador (0,34, -0,10) e da chaleira
for name, (mx, my) in MARKERS.items(): w.find(f"geom[@name='marker_{name}']").set("pos", f"{mx} {my} .7505")
board = json.load(open(G1 / "scene" / "markers.json")); board["markers"] = {k: list(v) for k, v in MARKERS.items()}
board["note_markers"] = "cena do cafe: marcadores em x 0,46/0,60 para caber no quadro da camera a 26 graus e nao ficar atras da chaleira"
json.dump(board, open(a.out.with_name(a.out.stem + "-markers.json"), "w"), indent=2)
tree.write(a.out, encoding="unicode")
# conferencia geometrica: nenhum objeto entre a camera da cabeca e cada marcador (cilindro envolvente de cada objeto)
import mujoco
mm = mujoco.MjModel.from_xml_path(str(a.out)); dd = mujoco.MjData(mm); mujoco.mj_forward(mm, dd); cam = dd.cam_xpos[mm.camera("head_camera").id]
occl = {}
for name, (mx, my) in MARKERS.items():
    tgt = np.array([mx, my, 0.75])
    for obj, ((ox, oy), _) in LAYOUT.items():
        ext = notes[obj]["extents_m"]; r = 0.5 * min(ext[0], ext[1]) + 0.01; top = 0.75 + ext[2] + (BASE_H if obj == "chaleira" else 0)   # corpo (sem alcas/bico) + 1 cm
        for t in np.linspace(0, 0.98, 200):
            q = cam + t * (tgt - cam)
            if np.hypot(q[0] - ox, q[1] - oy) < r and q[2] < top: occl.setdefault(name, []).append(obj); break
    for obj in ("copo",):
        pass
if occl: print(f"AVISO geometrico (cilindros envolventes, conservador): {occl}")
# prova real: foto da camera da cabeca com o robo na pose inicial e calibracao pelos 4 marcadores
import sys, subprocess
out = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_setup_scene.py"), str(a.out), "--seconds", "0.1"], capture_output=True, text=True, env={**os.environ, "MUJOCO_GL": "egl"})
sys.path.insert(0, str(G1 / "scripts")); import cv2; from vision import calibrate
img = cv2.cvtColor(cv2.imread(str(a.out.with_name(a.out.stem + "-head.png"))), cv2.COLOR_BGR2RGB)
cal = calibrate(img, board); print("calibracao na foto inicial:", cal["markers_used"], "ajuste", round(cal["fit_rms_px"], 2), "px, validacao", round(cal["heldout_max_px"], 2), "px")
json.dump({"layout": LAYOUT, "markers": MARKERS, "mass_kg": MASS_KG, "objects": notes, "coador_fixed": True, "calibration_check": {k: cal[k] for k in ("markers_used", "fit_rms_px", "heldout_max_px")}}, open(a.out.with_suffix(".json"), "w"), indent=2)
print("cena:", a.out)
