"""Etapa 3, primeira metade: o robo aperta o botao da base eletrica da chaleira.

Percepcao nao entra aqui: o botao e um site declarado na cena, e a posicao dele vem do modelo.
Isso esta declarado no relatorio. O que este script prova e a parte fisica: o braco alcanca o botao,
encosta nele com o punho fechado, e nem a base nem a chaleira saem do lugar.

Uso: MUJOCO_GL=egl ../g1-cup-grasp/.venv/bin/python scripts/press_button.py --cena scene/setup-luiz-031.xml --saida results/botao-01
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import cv2
import mujoco

G1 = Path(__file__).resolve().parents[2] / "g1-cup-grasp" / "scripts"
sys.path.insert(0, str(G1))
from g1_sim import G1Sim, ARM_JOINTS, HAND_JOINTS
from kinematics import ArmIK
from recorder import Recorder

MODEL = "claude-fable-5-1 (Claude Code)"
IK_JOINTS = None   # definido depois de ler --braco
rod = lambda v: cv2.Rodrigues(np.asarray(v, dtype=np.float64))[0]

ap = argparse.ArgumentParser()
ap.add_argument("--cena", required=True)
ap.add_argument("--saida", required=True)
ap.add_argument("--aproximacao", type=float, default=0.07, help="recuo de onde a mao comeca, em metros, ao longo de --direcao")
ap.add_argument("--direcao", default="+x", choices=["+x", "-x"], help="de que lado a mao chega no botao")
ap.add_argument("--curso", type=float, default=0.022, help="quanto a mao avanca sobre o botao, em metros")
ap.add_argument("--pitch", type=float, default=0.8)
ap.add_argument("--yaw-mao", type=float, default=0.0, help="graus")
ap.add_argument("--roll-mao", type=float, default=-0.5,
                help="a IK nao conhece os limites de torque: pitch 1,57 com yaw 135 fechava com 0,00 mm e exigia 97,8 graus de giro do punho, o que saturava as tres juntas dele (corrida botao-04). Esta combinacao pede 7,7 graus")
ap.add_argument("--afastamento", type=float, default=0.12, help="quanto a mao recua antes de fechar o punho")
ap.add_argument("--altura-passagem", type=float, default=0.13, help="quanto a mao sobe antes de descer sobre o botao")
ap.add_argument("--offset-punho", type=float, default=0.06,
                help="distancia da palma ao ponto de contato do punho: mirar a palma no botao faz o punho passar por ele (corrida botao-03)")
ap.add_argument("--espera-s", type=float, default=3.0, help="tempo declarado de fervura, sem termodinamica")
ap.add_argument("--pose-roll", type=float, default=0.30, help="roll do ombro na pose inicial")
ap.add_argument("--ref-roll", type=float, default=None,
                help="roll do ombro usado como referencia do nullspace da IK. Sem isso a IK escolhe poses com o ombro roçando o tronco: 56 quadros de autocolisao")
ap.add_argument("--dz-alvo", type=float, default=0.0, help="sobe o alvo em relacao ao centro do botao: o dedo estendido raspava a mesa em 518 quadros")
ap.add_argument("--kp-escala", type=float, default=1.0,
                help="multiplica kp e kd do braco. O erro de 17 mm da botao-16 e estacionario: nao cai nem com 3 s parado no alvo, entao e rigidez, nao tempo")
ap.add_argument("--indicador", action="store_true", help="aperta com o indicador estendido em vez do punho fechado")
ap.add_argument("--sem-cintura", action="store_true",
                help="tira waist_yaw do encadeamento: com ela, o alvo fecha mas a cintura fica com 8,1 graus de erro sob o peso do braco estendido, e isso vira 42 mm na ponta. O braco esquerdo sozinho alcanca o botao com 34 graus de giro maximo")
ap.add_argument("--braco", default="direito", choices=["direito", "esquerdo"],
                help="o botao no lado direito da mesa exige que o braco direito cruze o coador; com a base a esquerda, o braco esquerdo chega sem cruzar nada")
ap.add_argument("--reason", default="")
a = ap.parse_args()

saida = Path(a.saida)
if saida.exists():
    raise FileExistsError(saida)
saida.mkdir(parents=True)

LADO = "left" if a.braco == "esquerdo" else "right"
BRACO = [j.replace("right_", LADO + "_") for j in ARM_JOINTS]
MAO = [j.replace("right_", LADO + "_") for j in HAND_JOINTS]
IK_JOINTS = BRACO if a.sem_cintura else ["waist_yaw_joint"] + BRACO
PALMA = f"{LADO}_wrist_yaw_link"

sim = G1Sim(a.cena)
m, d = sim.m, sim.d
fps = 30

# a pose de repouso do modelo poe o indicador direito dentro do saco do coador: 46 pontos de contato
# ja no quadro zero, medidos em 600 passos de fisica sem comando nenhum, tanto na cena da etapa 1
# quanto nesta. O run_attempt.py nunca esbarrou nisso porque aplica uma pose de braco declarada antes
# de tudo. As corridas botao-01 a botao-09 partiam do repouso e a mao saia presa.
# a photo_arm_pose da tentativa 122 tira a mao do coador mas a poe em cima da chaleira, que foi
# reposicionada para (0,20, -0,30). Testei cinco poses e esta e a unica sem nenhum contato:
# palma em (-0,216, -0,238, 0,800), braco para tras do tronco.
POSE_INICIAL = ([0.8, -0.30, 0.0, 1.2, 0.0, 0.0, 0.0] if LADO == "right"
                else [0.8, a.pose_roll, 0.0, 1.2, 0.0, 0.0, 0.0])
for nome, valor in zip(BRACO, POSE_INICIAL):
    d.qpos[m.jnt_qposadr[m.joint(nome).id]] = valor
sim.q_des[:] = d.qpos[sim.qadr]
mujoco.mj_forward(m, d)

botao_site = m.site("botao_chaleira_site").id
botao_geom = m.geom("botao_chaleira").id
base_body = m.body("base_eletrica").id
chal_body = m.body("chaleira").id
alvo_botao = d.site_xpos[botao_site].copy()
alvo_botao[2] += a.dz_alvo
chal0 = d.xpos[chal_body].copy()

ik = ArmIK(sim, IK_JOINTS, palm=PALMA)
def ref_ns():
    """Referencia do nullspace: a pose atual, mas com o roll do ombro forçado a --ref-roll."""
    q = sim.q(IK_JOINTS)
    if a.ref_roll is not None:
        q = q.copy()
        q[IK_JOINTS.index(f"{LADO}_shoulder_roll_joint")] = a.ref_roll
    return q


rec = Recorder(m, str(saida / "botao.mp4"),
               title=f"{saida.name}  |  cena {Path(a.cena).stem}  |  {MODEL.split('(')[0].strip()}  |  {time.strftime('%Y-%m-%d %H:%M')}")

estado = {"frames": 0, "fase": "settle"}
amostras, toques = [], []
TRONCO = ("torso", "pelvis", "waist", "head")
copo_body = m.body("copo").id if "copo" in [m.body(i).name for i in range(m.nbody)] else None
copo0 = d.xpos[copo_body].copy() if copo_body is not None else None


def contatos_ruins():
    """Tudo que a mao encosta e nao devia. A versao anterior do criterio so olhava o botao, e por
    isso aprovou uma corrida com 235 quadros de autocolisao, 150 de dedo na mesa e 148 de punho
    apoiado na base eletrica."""
    auto = mesa = base = 0
    prof = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        b1 = m.body(int(m.geom_bodyid[c.geom1])).name
        b2 = m.body(int(m.geom_bodyid[c.geom2])).name
        par = (b1, b2)
        meu = [b for b in par if b.startswith(LADO + "_")]
        if not meu:
            continue
        outro = b2 if b1 in meu else b1
        if outro.startswith(TRONCO) or outro.startswith(("right_", "left_hip", "left_knee", "left_ankle")):
            if not outro.startswith(LADO + "_hand") and "ankle" not in outro:
                auto += 1
        if outro == "mesa":
            mesa += 1
            prof = min(prof, float(c.dist))
        if outro == "base_eletrica" and botao_geom not in (c.geom1, c.geom2):
            # o botao e geom do corpo base_eletrica: sem esta excecao, o proprio aperto contava
            # como "punho apoiado na base" e o criterio reprovava o que queria medir
            base += 1
            prof = min(prof, float(c.dist))
    return auto, mesa, base, round(prof * 1000, 2)


# so os geoms de COLISAO: cada elo da mao tem um visual (contype 0) na frente do de colisao, e medir
# contra o visual dava -1,56 mm de "penetracao" com zero contatos na fisica
_geoms_mao = [g for g in range(m.ngeom)
              if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "").startswith(
                  (f"{LADO}_hand", f"{LADO}_wrist")) and m.geom_contype[g] != 0]


def distancia_mao_botao():
    """Menor distancia entre qualquer geom da mao e o geom do botao. Negativa = penetrando.
    Estimar isso pela posicao da palma mais um offset errou tres vezes; aqui e medido."""
    return min(mujoco.mj_geomDistance(m, d, g, botao_geom, 0.3, None) for g in _geoms_mao)


def contatos_botao():
    fora = []
    for i in range(d.ncon):
        c = d.contact[i]
        if botao_geom in (c.geom1, c.geom2):
            outro = c.geom2 if c.geom1 == botao_geom else c.geom1
            nome = m.body(int(m.geom_bodyid[outro])).name
            fora.append((nome, float(c.dist)))
    return fora


def avanca(n):
    for _ in range(n):
        ate = (estado["frames"] + 1) / fps
        while d.time < ate - m.opt.timestep / 2:
            tau = sim.kp * (sim.q_des - d.qpos[sim.qadr]) - sim.kd * d.qvel[sim.vadr]
            tau += d.qfrc_bias[sim.vadr]
            d.ctrl[:] = np.clip(tau, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        estado["frames"] += 1
        palma, _ = ik.fk(sim.q(IK_JOINTS))
        ct = contatos_botao()
        if ct:
            toques.append({"t": round(d.time, 3), "fase": estado["fase"], "contatos": ct})
        amostras.append({"t": round(d.time, 3), "fase": estado["fase"],
                         "palma": palma.round(5).tolist(),
                         "botao": d.site_xpos[botao_site].round(5).tolist(),
                         "base": d.xpos[base_body].round(5).tolist(),
                         "chaleira": d.xpos[chal_body].round(5).tolist(),
                         "chaleira_tilt_deg": round(float(np.degrees(np.arccos(
                             np.clip(d.xmat[chal_body].reshape(3, 3)[2, 2], -1, 1)))), 3),
                         "dist_mao_botao_mm": round(distancia_mao_botao() * 1000, 2),
                         "ruins": contatos_ruins(),
                         "copo": (d.xpos[copo_body].round(5).tolist() if copo_body is not None else None),
                         "toca_botao": bool(ct),
                         "mao_no_botao": any((f"{LADO}_hand" in n or f"{LADO}_wrist" in n) for n, _ in ct)})
        rec.frame(d)


def fase(nome, texto):
    estado["fase"] = nome
    rec.event(f"{nome}: {texto}")


def move_juntas(q_alvo, segundos):
    """Interpola no espaco de JUNTAS ate uma pose resolvida de uma vez, com orcamento grande.
    Interpolar no espaco cartesiano e deixar a IK reconvergir a cada quadro, com limite de passo,
    leva a uma solucao final diferente da avaliada: na corrida botao-05 o punho pitch terminou com
    62,6 graus de erro e torque de 44 N.m contra limite de 5, saturado, apesar de a pose escolhida
    offline pedir so 7,7 graus de giro."""
    q_ini = sim.q(IK_JOINTS)
    n = max(1, round(segundos * fps))
    for i in range(n):
        u = (i + 1) / n
        u = u * u * (3 - 2 * u)
        sim.set_targets(IK_JOINTS, q_ini + u * (np.asarray(q_alvo) - q_ini))
        avanca(1)


def move(pos, rot, segundos):
    ini, ini_r = ik.fk(sim.q(IK_JOINTS))
    q = sim.q(IK_JOINTS)
    rv = cv2.Rodrigues(np.asarray(rot @ ini_r.T, dtype=np.float64))[0].ravel()
    n = max(1, round(segundos * fps))
    passo = np.radians(240) / fps
    for i in range(n):
        u = (i + 1) / n
        u = u * u * (3 - 2 * u)
        alvo_r = rod(rv * u) @ ini_r
        q, _ = ik.solve(ini + u * (np.asarray(pos) - ini), alvo_r, q, q, iterations=35, max_step=passo)
        sim.set_targets(IK_JOINTS, q)
        avanca(1)


# a mao chega pela direcao que vai da base para o botao, seja ela qual for: nas corridas 01 e 02 essa
# direcao estava chumbada em x e o botao mudou de lado duas vezes
radial = alvo_botao[:2] - d.xpos[base_body][:2]
radial = radial / max(float(np.linalg.norm(radial)), 1e-9)
dir3 = np.array([radial[0], radial[1], 0.0])
R = rod([0, 0, np.radians(a.yaw_mao)]) @ rod([0, a.pitch, 0]) @ rod([a.roll_mao, 0, 0])
# punho totalmente fechado poe o indicador abaixo da palma e ele bate no tampo: 131 quadros de
# contato mao-mesa na corrida botao-14, e o ombro fica com 5,5 graus de erro por estar apoiado.
# Aperta-se um botao com o indicador estendido e o resto fechado, que e o que um humano faz.
_lim = m.jnt_range[[m.joint(n).id for n in MAO]][:, 1]
punho = _lim * 0.9
if a.indicador:
    for _k, _n in enumerate(MAO):
        if "index" in _n:
            punho[_k] = 0.0

if a.kp_escala != 1.0:
    for _n in BRACO:
        _i = sim.act_joint[_n]
        sim.kp[_i] *= a.kp_escala
        sim.kd[_i] *= a.kp_escala ** 0.5

fase("settle", "robo em repouso")
avanca(20)

# a mao em punho, na pose de repouso, fica a 1,0 mm do coador e 1,3 mm do copo (medido com
# mj_geomDistance ao longo do caminho). Fechar o punho ali ja cria contato, e dai em diante todo
# movimento arrasta a mao contra o filtro. Afasta primeiro, fecha depois.
fase("afasta", f"leva a mao {a.afastamento * 100:.0f} cm para tras antes de fechar o punho")
pos_livre, _ = ik.fk(sim.q(IK_JOINTS))
pos_livre = pos_livre + np.array([-a.afastamento, -a.afastamento * 0.5, 0.0])
q_livre, i_livre = ik.solve(pos_livre, ik.fk(sim.q(IK_JOINTS))[1], sim.q(IK_JOINTS), sim.q(IK_JOINTS), iterations=400)
print(f'  IK afasta: alvo {np.round(pos_livre,4)} erro {i_livre["position_error_m"]*1000:.2f} mm')
move_juntas(q_livre, 1.2)

fase("punho", "fecha a mao longe do coador: o botao e apertado com o punho, nao com a ponta do dedo")
q_ab = sim.q(MAO)
for i in range(20):
    u = (i + 1) / 20
    sim.set_targets(MAO, q_ab + u * (punho - q_ab))
    avanca(1)

# ponto de passagem por cima: em linha reta do repouso ate o botao a mao varre o coador e o copo,
# 1501 e 184 quadros de contato na corrida botao-05. Sobe primeiro, depois desce sobre o botao.
fase("sobe", f"sobe {a.altura_passagem * 100:.0f} cm acima do ponto de aproximacao, longe do coador")
passagem = alvo_botao + (a.offset_punho + a.aproximacao) * dir3 + np.array([0, 0, a.altura_passagem])
q_pass, i_pass = ik.solve(passagem, R, sim.q(IK_JOINTS), ref_ns(), iterations=400)
print(f'  IK passagem: alvo {np.round(passagem,4)} erro {i_pass["position_error_m"]*1000:.2f} mm')
move_juntas(q_pass, 2.0)

fase("aproxima", f"desce ate {a.aproximacao * 100:.0f} cm do botao")
alvo_apr = alvo_botao + (a.offset_punho + a.aproximacao) * dir3
q_apr, i_apr = ik.solve(alvo_apr, R, q_pass, ref_ns(), iterations=400)
print(f'  IK aproxima: alvo {np.round(alvo_apr,4)} erro {i_apr["position_error_m"]*1000:.2f} mm')
move_juntas(q_apr, 1.5)

fase("aperta", f"avanca {a.curso * 1000:.0f} mm sobre o botao")
alvo_ap = alvo_botao + (a.offset_punho - a.curso) * dir3
q_ap, i_ap = ik.solve(alvo_ap, R, q_apr, ref_ns(), iterations=400)
print(f'  IK aperta: alvo {np.round(alvo_ap,4)} erro {i_ap["position_error_m"]*1000:.2f} mm')
move_juntas(q_ap, 1.2)
avanca(10)

fase("espera", f"tempo declarado de fervura: {a.espera_s:.0f} s (sem termodinamica)")
avanca(int(a.espera_s * fps))

fase("recua", "solta o botao e afasta a mao")
pos_now, _ = ik.fk(sim.q(IK_JOINTS))
move(pos_now + a.aproximacao * dir3, R, 1.5)
avanca(10)

rec.close(str(saida / "timeline.json"))

chal_dz = float(np.linalg.norm(d.xpos[chal_body] - chal0))
tilt = amostras[-1]["chaleira_tilt_deg"]
# so vale o toque da MAO: na corrida botao-02 o bico da chaleira encostava no botao desde o quadro
# zero e o criterio antigo dava 291 quadros de "aperto" com a mao a 20 cm de distancia
# terceira versao do criterio. A primeira contava qualquer contato e aceitou uma corrida em que so
# o bico da chaleira encostava. A segunda contava so a mao, e aceitou a botao-10, onde os tres
# quadros de toque aconteceram na fase de SUBIDA, de rocao, com a mao de passagem. Agora o toque
# so vale se acontecer na fase de aperto ou na espera que vem depois dela.
n_toque = sum(1 for s in amostras if s["mao_no_botao"] and s["fase"] in ("aperta", "espera"))
n_rocao = sum(1 for s in amostras if s["mao_no_botao"] and s["fase"] not in ("aperta", "espera"))
n_chaleira = sum(1 for s in amostras if s["toca_botao"] and not s["mao_no_botao"])
prof = min([c[1] for t in toques for c in t["contatos"] if ("right_hand" in c[0] or "wrist" in c[0])], default=0.0)
_auto = sum(s["ruins"][0] for s in amostras)
_mesa = sum(s["ruins"][1] for s in amostras)
_base = sum(s["ruins"][2] for s in amostras)
_P = np.array([s["palma"] for s in amostras]); _dt = 1.0 / fps
_A = np.gradient(np.gradient(_P, _dt, axis=0), _dt, axis=0)
_pico = float(np.linalg.norm(_A, axis=1).max())
_copo_mm = (float(np.linalg.norm(np.array(amostras[-1]["copo"]) - copo0)) * 1000
            if copo0 is not None else None)

relatorio = {
    "modelo_controlador": MODEL,
    "cena": a.cena,
    "reason": a.reason,
    "percepcao": "nenhuma; a posicao do botao vem do site declarado na cena",
    "agua": "nenhuma; a fervura e um tempo declarado, sem termodinamica",
    "menor_distancia_mao_botao_mm": round(min(s["dist_mao_botao_mm"] for s in amostras), 2),
    "menor_distancia_na_fase_de_aperto_mm": round(min([s["dist_mao_botao_mm"] for s in amostras
                                                       if s["fase"] in ("aperta", "espera")] or [999]), 2),
    "quadros_de_aperto": n_toque,
    "quadros_de_rocao_fora_do_aperto": n_rocao,
    "quadros_com_a_chaleira_no_botao": n_chaleira,
    "penetracao_maxima_no_botao_mm": round(prof * 1000, 2),
    "chaleira_deslocou_m": round(chal_dz, 5),
    "chaleira_inclinacao_final_deg": tilt,
    "duracao_s": round(d.time, 2),
    "autocolisao_quadros": _auto,
    "mao_na_mesa_quadros": _mesa,
    "punho_na_base_quadros": _base,
    "penetracao_maxima_mesa_ou_base_mm": round(min([s["ruins"][3] for s in amostras] or [0.0]), 2),
    "pico_aceleracao_mao_m_s2": round(_pico, 2),
    "copo_sob_o_filtro_deslocou_mm": (round(_copo_mm, 2) if _copo_mm is not None else None),
    "aceito": bool(n_toque >= 10 and chal_dz < 0.005 and tilt < 3.0
                   and _auto == 0 and _mesa == 0 and _base == 0
                   and (_copo_mm is None or _copo_mm < 6.0)),   # linha de base medida: 3,81 mm de assentamento sem o robo tocar em nada   # 10 quadros = um terco de segundo de contato, nao um rocao
}
(saida / "relatorio.json").write_text(json.dumps(relatorio, indent=2) + "\n")
(saida / "trajetoria.json").write_text(json.dumps(amostras) + "\n")
(saida / "toques.json").write_text(json.dumps(toques, indent=2) + "\n")
print(json.dumps(relatorio, indent=2))
