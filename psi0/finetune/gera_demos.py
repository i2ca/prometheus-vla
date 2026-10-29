"""Demonstrações roteirizadas (IK, sem modelo) da pega da caneca e da colocação sob o coador, para o ajuste fino do Ψ0.
Mesma montagem do cliente ../base/rodar_mujoco.py (cena, --afasta, câmera da cabeça, pose inicial, ganhos), para que a demo e a
avaliação vejam o mesmo mundo. Fluxo completo e opções no README.md desta pasta.
Uso: MUJOCO_GL=egl python gera_demos.py saida/demos --n 60 --semente 0 [--cintura --humano --colocar]
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, mujoco, cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "base"))
from rodar_mujoco import Sim, CENA, JUNTAS, MAO, BRACO, CINTURA, POSE_HE, afasta, monta

BRACO_D = BRACO[7:]
ABERTA = np.array([0, 0, 0, 0.35, 0.35, 0.35, 0.35])               # mão direita, ordem thumb0-2, middle0-1, index0-1; polegar aberto
FECHADA = np.array([-0.1, -0.6, -0.9, 0.9, 1.6, 0.9, 1.4])
TRONCO_COLOCAR = float(os.environ.get("TRONCO_COLOCAR", 21))   # graus; varredura de 08/10: 18° quebra o punho 12-24°, 21° dá tronco 21-22° e punho 6-8°, 26° punho 2-6°
SOLTA = np.array([0, 0.6, 0, 0, 0, 0, 0])   # para largar a caneca: dedos retos e polegar aberto para fora; com ABERTA o polegar seguia encostado e a arrastava 40 mm        # alvo além do contato: o PD aperta (ação de fecho real da Task 3)
def _rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def _ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
INSTR = "the robot uses its right hand to pick up the white mug kept on the right side of the desk"


def ik(sim, alvo_p, alvo_R, q, it=150, ref=None, fixa=None, nomes=BRACO_D, peso=None, ponto=None, so_z=False, peso_rot=0.3, k_ref=0.05):
    """IK do punho direito; trabalha numa cópia de MjData. nomes: juntas que ela move (padrão o braço direito).
    ref: postura puxada no espaço nulo (mantém o ramo). fixa: {junta: valor} posto na cópia antes (ex. a cintura).
    peso: por junta, < 1 faz a junta mexer menos (passo ponderado).
    ponto: ponto no referencial do punho que vai a alvo_p (padrão a origem do punho). so_z: só alinha o eixo z do punho com o z de
    alvo_R e deixa livre o giro em torno dele (a mão escolhe o giro que o braço dá, sem forçar o punho)."""
    m = sim.m; d = mujoco.MjData(m); d.qpos[:] = sim.d.qpos
    for j, v in (fixa or {}).items(): d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
    jid = [m.joint(j).id for j in nomes]; qa, va, lim = m.jnt_qposadr[jid], m.jnt_dofadr[jid], m.jnt_range[jid]
    W = np.ones(len(nomes)) if peso is None else np.asarray(peso, float)
    w = m.body("right_wrist_yaw_link").id; jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv)); q = q.copy()
    for _ in range(it):
        d.qpos[qa] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        Rw = d.xmat[w].reshape(3, 3); pt = d.xpos[w] + (Rw @ ponto if ponto is not None else 0)
        ep = alvo_p - pt
        er = np.cross(Rw[:, 2], alvo_R[:, 2]) if so_z else cv2.Rodrigues(alvo_R @ Rw.T)[0].ravel()
        if np.linalg.norm(ep) < 3e-4 and np.linalg.norm(er) < 3e-3: break
        mujoco.mj_jac(m, d, jp, jr, pt, w)
        Jr = (np.eye(3) - np.outer(Rw[:, 2], Rw[:, 2])) @ jr[:, va] if so_z else jr[:, va]
        J = np.vstack((jp[:, va], peso_rot * Jr)) * W
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6)); dq = Jp @ np.r_[ep, peso_rot * er]
        if ref is not None: dq += k_ref * (np.eye(len(q)) - Jp @ J) @ ((ref - q) / W)
        dq = dq * W
        q = np.clip(q + dq * min(1, 0.1 / max(np.abs(dq).max(), 1e-9)), lim[:, 0], lim[:, 1])
    return q, float(np.linalg.norm(ep)), float(np.linalg.norm(er))


def episodio_humano(cena, out, xy, yaw, rng, t_pega, afast=0.10, colocar=False):
    """Roteiro com movimento de gente (Jeannerod): um gesto só, sem paradas. O ponto de pega da mão segue uma curva suave no espaço
    (sobe por cima do tampo, passa por um ponto de aproximação sem parar nele e chega à caneca), com velocidade em sino e
    desaceleração mais longa; a mão abre no caminho (máximo a ~60%) e fecha no fim da desaceleração. A cada quadro uma IK conjunta
    de cintura (pitch, yaw) e braço acompanha a curva; só o eixo de fecho da mão é imposto (vertical) e o punho é caro (peso 0,3,
    puxado para 0): ombro, cotovelo e tronco orientam a mão e o punho não quebra para o lado (desvio de 1-4°, só extensão de ~22°)."""
    from scipy.interpolate import CubicSpline
    sim = Sim(cena); m, d = sim.m, sim.d
    pose = np.array(POSE_HE) + rng.normal(0, 0.02, 28) * np.r_[np.zeros(14), np.ones(14)]
    _, n_sub = monta(sim, pose, xy, afast, yaw_copo=yaw, render=False)
    outros = {n: d.xpos[m.body(n).id].copy() for n in ("chaleira", "pote", "coador")}
    copo0 = d.xpos[sim.copo].copy(); z0 = copo0[2]; J = JUNTAS + CINTURA; t = np.asarray(t_pega)
    nomes = ["waist_pitch_joint", "waist_yaw_joint"] + BRACO_D; peso = [0.3, 0.3, 1, 1, 1, 1, 0.3, 0.3, 0.3]
    w = m.body("right_wrist_yaw_link").id; qb = sim.q(BRACO_D)
    ikh = lambda alvo, eixo, q, ref, it=80, k=0.3: ik(sim, alvo, eixo, q, it=it, ref=ref, nomes=nomes, peso=peso, ponto=t, so_z=True, k_ref=k)
    def ikb(alvo, eixo, sol, ref, cint):   # tronco imposto (perfil suave), só o braço resolve; punho caro e puxado para 0
        qa, e, _ = ik(sim, alvo, eixo, sol[2:], it=80, ref=ref[2:], fixa={"waist_pitch_joint": cint[0], "waist_yaw_joint": cint[1]},
                      peso=[1, 1, 1, 1, 0.3, 0.3, 0.3], ponto=t, so_z=True, k_ref=0.3)
        return np.r_[cint, qa], e
    # postura de pega (tronco e punho puxados para reto)
    pega, e0, _ = ikh(copo0, np.eye(3), np.r_[0, 0, qb], np.r_[0, 0, qb[:4], 0, 0, 0], it=2000, k=0.05)
    ref = pega.copy(); ref[6:] = 0
    dm = mujoco.MjData(m); dm.qpos[:] = d.qpos
    for j, v in zip(nomes, pega): dm.qpos[m.jnt_qposadr[m.joint(j).id]] = v
    mujoco.mj_kinematics(m, dm); Rg = dm.xmat[w].reshape(3, 3).copy()
    Rw0 = d.xmat[w].reshape(3, 3).copy(); G0 = d.xpos[w] + Rw0 @ t                       # ponto de pega no início (mão pendurada)
    apr = copo0 - 0.06 * Rg[:, 1] + [0, 0, 0.03]                                          # chega pelo lado da palma, de cima
    via = np.array([0.30, 0.5 * (G0[1] + apr[1]), max(apr[2], 0.80)])   # passa por cima do tampo
    P = np.array([G0, via, apr, copo0]); sc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]; sc /= sc[-1]
    curva = CubicSpline(sc, P, bc_type="clamped")
    def eixo(u):   # z da mão vai do inicial à vertical nos primeiros 70% do gesto
        a = min(1.0, u / 0.7); a = a * a * (3 - 2 * a); z = (1 - a) * Rw0[:, 2] + a * np.array([0, 0, 1.0]); z /= np.linalg.norm(z)
        R = np.eye(3); R[:, 2] = z; return R
    im = [J.index(j) for j in MAO[7:]]; ib = [J.index(j) for j in BRACO_D]; ic = [J.index(j) for j in CINTURA]
    a0 = sim.q(J).copy(); h0 = a0[im].copy(); qc = np.r_[a0[ic][1], a0[ic][2], a0[ib]]
    def vetor(sol, mao): v = a0.copy(); v[ic] = [0, sol[0], sol[1]]; v[ib] = sol[2:]; v[im] = mao; return v
    T = 72                                                                                  # 2,4 s de transporte
    # desaceleração bem mais longa que a aceleração: chega devagar perto da caneca, onde a posição da mão se acerta
    # (com 1,8 s e cauda curta a política v4 errava a mão 1 cm e o indicador escapava em 12/30)
    tau = lambda u: min(1.0, u ** 1.25 * (1 + 0.25 * (1 - u)))
    erro = [e0]
    S, A, sub, ded, inc, mesa, pen, pos_copo, QP = [], [], [], [], [], [], [], [], []; mesa_id = m.body("mesa").id

    def quadro(novo):
        S.append(sim.q(J).astype(np.float32)); A.append(novo.astype(np.float32))
        sim.alvo(J, novo); sim.passo(n_sub); QP.append(d.qpos.copy())
        pos_copo.append(d.xpos[sim.copo].copy()); sub.append(d.xpos[sim.copo][2] - z0); ded.append(sim.dedos_no_copo()); inc.append(np.degrees(np.arccos(np.clip(d.xmat[sim.copo][8], -1, 1))))
        pen.append(min([k.dist for k in d.contact[:d.ncon] if sim.copo in (m.geom_bodyid[k.geom1], m.geom_bodyid[k.geom2]) and {m.geom_bodyid[k.geom1], m.geom_bodyid[k.geom2]} & sim.mao_dir] + [0]))
        mesa.append(any(mesa_id in (m.geom_bodyid[c_.geom1], m.geom_bodyid[c_.geom2]) and ({m.geom_bodyid[c_.geom1], m.geom_bodyid[c_.geom2]} & sim.mao_dir) for c_ in d.contact[:d.ncon]))

    alt_w = []
    for _ in range(8): quadro(a0)                                                           # tempo de reação
    sol = qc
    for k, u in enumerate(suave(T)):
        sol, e = ikb(curva(tau(u)), eixo(u), sol, ref, tau(u) * pega[:2]); erro.append(e)   # tronco inclina junto com o braço
        ab = np.clip(u / 0.6, 0, 1); ab = ab * ab * (3 - 2 * ab)                              # abre até 60% do gesto
        fechar = np.clip((u - 0.82) / 0.18, 0, 1); fechar = fechar * fechar * (3 - 2 * fechar)
        mao = h0 + ab * (ABERTA - h0) + fechar * 0.3 * (FECHADA - ABERTA)                      # começa a fechar no fim, sem salto
        quadro(vetor(sol, mao)); alt_w.append(d.xpos[w][2])
    mao_atual = ABERTA + 0.3 * (FECHADA - ABERTA)
    for s_ in suave(24): quadro(vetor(sol, mao_atual + s_ * (FECHADA - mao_atual)))
    for _ in range(6): quadro(vetor(sol, FECHADA))
    ref_l = ref.copy(); ref_l[0] *= 0.67                                                    # o tronco endireita um terço ao levantar
    for s_ in suave(40):                                                                    # levanta 10 cm trazendo 3 cm para o corpo
        sol, e = ikb(copo0 + s_ * np.array([-0.03, 0, 0.10]), np.eye(3), sol, ref, pega[:2] * [1 - 0.33 * s_, 1]); erro.append(e); quadro(vetor(sol, FECHADA))
    if not colocar:
        for _ in range(30): quadro(vetor(sol, FECHADA))
    else:   # etapa 2: leva a caneca para baixo do coador, apoia, solta e recolhe o braço, no mesmo gesto humano
        cod = d.xpos[m.body("coador").id].copy()
        alvo = np.array([cod[0] - 0.006, cod[1] - 0.020, z0 + 0.010])          # mais central e 6 mm acima da v12: margem de gotejamento e assentamento sem atravessar o disco
        # onde a caneca está de fato na mão (escorrega ~1 cm ao levantar): os alvos valem para a caneca, não para a mão
        dm.qpos[:] = d.qpos; mujoco.mj_kinematics(m, dm)
        delta = d.xpos[sim.copo] - (dm.xpos[w] + dm.xmat[w].reshape(3, 3) @ t)
        p1 = dm.xpos[w] + dm.xmat[w].reshape(3, 3) @ t
        # janela para deslizar sob o filtro: fundo da caneca 1 cm acima do disco da base (2,6 mm) e boca 0,5 cm abaixo do filtro;
        # entra no meio dela e só desce ao disco já em cima do alvo
        # leva alta e só desce perto do suporte (como gente), entrando na janela sob o filtro já devagar
        # entra 2 cm mais longe da haste (o braço deriva para ela ao esticar) e só no fim encosta no lugar
        Pc = np.array([p1, [cod[0] - 0.13, alvo[1] - 0.02, z0 + 0.07], [cod[0] - 0.06, alvo[1] - 0.02, z0 + 0.040], [cod[0] - 0.02, alvo[1] - 0.012, z0 + 0.026], alvo])

        sc_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Pc, axis=0), axis=1))]; sc_ /= sc_[-1]
        curva_c = CubicSpline(sc_, Pc, bc_type="clamped")                                    # caminho da CANECA
        corr = np.zeros(3); cede = np.zeros(3)
        solc, _, _ = ikh(alvo - delta, np.eye(3), sol, np.r_[0, 0, sol[2:6], 0, 0, 0], it=2000, k=0.05)   # postura de colocar
        # a IK conjunta deixava o tronco ir ao batente (30°), mas o braço alcança o apoio com 15-20° (medido em 08/10): o tronco fica em
        # no máximo TRONCO_COLOCAR e o braço resolve o resto
        if solc[0] > np.radians(TRONCO_COLOCAR):
            solc[0] = np.radians(TRONCO_COLOCAR)
            qa, _, _ = ik(sim, alvo - delta, np.eye(3), solc[2:], it=2000, ref=np.r_[solc[2:6], 0, 0, 0], fixa={"waist_pitch_joint": solc[0], "waist_yaw_joint": solc[1]},
                          peso=[1, 1, 1, 1, 0.3, 0.3, 0.3], ponto=t, so_z=True, k_ref=0.05); solc[2:] = qa
        refc = solc.copy(); refc[6:] = 0; c0 = sol[:2].copy()
        def em_pe():   # como quem sente o copo inclinar e corrige: gira o alvo do eixo da mão pelo quanto a caneca saiu da vertical
            zc = d.xmat[sim.copo].reshape(3, 3)[:, 2]; zh = d.xmat[w].reshape(3, 3)[:, 2]
            ax = np.cross(zc, [0, 0, 1.0]); s_ = np.linalg.norm(ax)
            Rk = cv2.Rodrigues(ax / s_ * np.arcsin(min(1.0, s_)))[0] if s_ > 1e-6 else np.eye(3)
            R = np.eye(3); R[:, 2] = Rk @ zh; return R
        ref_inicio = sol.copy()
        def ikr(alvo_p, sol, cint, rot=1.0):
            """Só o braço, eixo de fecho vertical e giro da mão livre (o punho não quebra; travar a orientação custava 22-32° de desvio
            no punho ou 40-48° de giro no tronco). A referência postural muda junto com o transporte para não dar tranco no cotovelo."""
            ref_t = ref_inicio[2:] + rot * (refc[2:] - ref_inicio[2:])
            qa, e, _ = ik(sim, alvo_p, np.eye(3), sol[2:], it=80, ref=ref_t, fixa={"waist_pitch_joint": cint[0], "waist_yaw_joint": cint[1]},
                          peso=[1, 1, 1, 1, 0.3, 0.3, 0.3], ponto=t, so_z=True, k_ref=0.1)
            # 0,02 rad/quadro no começo, liberando até 0,06 sem degrau de velocidade na troca entre levantar e transportar
            passo = 0.02 + 0.04 * (rot * rot * (3 - 2 * rot))
            return np.r_[cint, sol[2:] + np.clip(qa - sol[2:], -passo, passo)], e
        for u in suave(75):   # 2,5 s de transporte; corrige devagar a posição pelo que vê (o braço cede mais quanto mais estica)
            desejo = curva_c(tau(u)); corr = np.clip(corr + 0.15 * (desejo - d.xpos[sim.copo]), -0.03, 0.03); cede = corr.copy()
            sol, e = ikr(desejo - delta + corr, sol, c0 + tau(u) * (solc[:2] - c0), tau(u)); erro.append(e)
            quadro(vetor(sol, FECHADA))
        # liga o fim do transporte ao alvo assentado sem trocar instantaneamente a referência da IK
        desejo0 = curva_c(tau(1.0)); corr0 = corr.copy(); cint0 = sol[:2].copy()
        for s_ in suave(8):
            desejo = desejo0 + s_ * (alvo - desejo0); corr = np.clip(corr0 + s_ * (cede - corr0), -0.03, 0.03); corr[2] = min(corr[2], cede[2])
            sol, e = ikr(desejo - delta + corr, sol, cint0 + s_ * (solc[:2] - cint0)); quadro(vetor(sol, FECHADA))
        for _ in range(6): quadro(vetor(sol, FECHADA))                                     # chega acima do disco
        solta_inicio = len(pos_copo) - 1
        for s_ in suave(45): quadro(vetor(sol, FECHADA + s_ * (SOLTA - FECHADA)))           # abre devagar e deixa a gravidade assentar
        for _ in range(15): quadro(vetor(sol, SOLTA))                                       # espera estabilizar antes de tirar a mão
        solta_fim = len(pos_copo) - 1
        # tira a mão como gente: palma se afasta da caneca na horizontal (sentido caneca -> punho), só depois sobe; recuar pelo eixo
        # do punho (versão de 08/10) enganchava os dedos abertos na caneca e a arrastava para fora da mesa
        dm.qpos[:] = d.qpos; mujoco.mj_kinematics(m, dm); p_sol = alvo - delta + corr
        fora = dm.xpos[w] - d.xpos[sim.copo]; fora[2] = 0; fora /= np.linalg.norm(fora); copo_solto = d.xpos[sim.copo].copy()
        saida = CubicSpline([0, 0.65, 1], [p_sol, p_sol + 0.06 * fora + [0, 0, 0.01], p_sol + 0.06 * fora + [0, 0, 0.05]], bc_type="clamped")
        for s_ in suave(36): sol, e = ikb(saida(s_), np.eye(3), sol, refc, sol[:2]); quadro(vetor(sol, SOLTA))
        mexeu_soltura = float(np.linalg.norm(d.xpos[sim.copo] - copo_solto))
        # recolhe como gente: a mão volta por cima da mesa para perto do corpo e só então desce, o tronco endireitando junto
        dm.qpos[:] = d.qpos; mujoco.mj_kinematics(m, dm); pr = dm.xpos[w] + dm.xmat[w].reshape(3, 3) @ t
        perto = np.array([0.24, pr[1], pr[2] + 0.01]); c1 = sol[:2].copy()                 # puxa a mão de volta, sem subir nem abrir para o lado
        for s_ in suave(45): sol, e = ikb(pr + s_ * (perto - pr), np.eye(3), sol, ref, c1 * (1 - s_)); quadro(vetor(sol, SOLTA))
        fim = vetor(sol, SOLTA)
        # a0 fecha parcialmente o dedo médio da mão esquerda; mantê-la aberta durante a descida evita um salto inútil de 2,3 rad/s
        descanso = a0.copy(); descanso[[J.index(j) for j in MAO]] = np.r_[a0[[J.index(j) for j in MAO[:7]]], SOLTA]
        for s_ in suave(45): quadro(fim + s_ * (descanso - fim))                            # desce ao descanso já atrás da borda
        a0 = descanso
        for _ in range(15): quadro(a0)
    mexeu = max(float(np.linalg.norm(d.xpos[m.body(n).id] - p)) for n, p in outros.items() if not (colocar and n == "coador"))
    iw = [J.index(f"right_wrist_{n}_joint") for n in ("roll", "pitch", "yaw")]; Sx = np.array(S)
    seg = np.array(sub[-30:]); r = dict(xy=list(xy), cintura_rpy=[0, float(pega[0]), float(pega[1])], yaw_deg=float(np.degrees(yaw)), quadros=len(S),
        subida_final_mm=1000 * float(seg.mean()), subida_min_seg_mm=1000 * float(seg.min()), dedos_seg_min=int(min(ded[-30:])),
        inclinacao_max_deg=float(max(inc)), outros_mexeram_mm=1000 * mexeu, ik_erro_max_mm=1000 * max(erro),
        quadros_mao_na_mesa=int(sum(mesa)), penetracao_max_mm=-1000 * float(min(pen)),
        punho_max_deg=float(np.degrees(np.abs(Sx[:, iw]).max())), punho_na_pega_deg=np.degrees(pega[6:]).round(1).tolist(),
        arco_acima_da_pega_cm=100 * float(max(alt_w) - alt_w[-1]), desvio_punho_max_deg=float(np.degrees(np.abs(Sx[:, iw[2]]).max())), tronco_max_deg=np.degrees(np.abs(Sx[:, ic]).max(0)).round(1).tolist())
    r["aceita"] = bool(r["subida_min_seg_mm"] > 70 and r["dedos_seg_min"] >= 3 and r["inclinacao_max_deg"] < 15 and r["outros_mexeram_mm"] < 10
                       and r["quadros_mao_na_mesa"] == 0 and r["ik_erro_max_mm"] < 5 and r["penetracao_max_mm"] < 3)
    if colocar:
        cod_id = m.body("coador").id; disco = m.geom("coador_base_disco").id; toques = 0; pen_disco = 0.0
        for q in QP:   # todos os quadros: contato com haste/aro/filtro e profundidade do apoio caneca-disco
            dm.qpos[:] = q; mujoco.mj_forward(m, dm)
            for c_ in dm.contact[:dm.ncon]:
                corpos = (m.geom_bodyid[c_.geom1], m.geom_bodyid[c_.geom2])
                if cod_id in corpos and disco not in (c_.geom1, c_.geom2): toques += 1
                if disco in (c_.geom1, c_.geom2) and sim.copo in corpos: pen_disco = max(pen_disco, -float(c_.dist))
        cxy = d.xpos[sim.copo].copy(); inc_f = float(np.degrees(np.arccos(np.clip(d.xmat[sim.copo][8], -1, 1))))
        saida_xy = cod[:2]; margem_boca = 0.035 - float(np.linalg.norm(cxy[:2] - saida_xy))
        r.update(alca_final_deg=float(np.degrees(np.arctan2(d.xmat[sim.copo].reshape(3, 3)[1, 0], d.xmat[sim.copo].reshape(3, 3)[0, 0]))), caneca_na_mao_mm=(1000 * delta).round(1).tolist(), correcao_final_mm=(1000 * corr).round(1).tolist(), erro_colocar_mm=1000 * float(np.linalg.norm(cxy[:2] - alvo[:2])), inclinacao_final_deg=inc_f, dedos_final=int(ded[-1]),
                 quadros_tocando_coador=int(toques), subida_max_mm=1000 * float(max(sub)), caneca_mexeu_na_soltura_mm=1000 * mexeu_soltura,
                 penetracao_caneca_disco_mm=1000 * pen_disco, margem_saida_na_boca_mm=1000 * margem_boca)
        A_ = np.array(A); vmax = 30 * float(np.abs(np.diff(A_, axis=0)).max()); amax = 900 * float(np.abs(np.diff(A_, n=2, axis=0)).max())
        P_ = np.array(pos_copo[solta_inicio:solta_fim + 1]); assentamento = 1000 * float(np.linalg.norm(P_[-1] - P_[0])); passo_assentamento = 1000 * float(np.linalg.norm(np.diff(P_, axis=0), axis=1).max())
        r.update(velocidade_junta_max_rad_s=vmax, aceleracao_junta_max_rad_s2=amax, movimento_durante_abertura_mm=assentamento, passo_caneca_max_abertura_mm=passo_assentamento)
        r["aceita"] = bool(r["erro_colocar_mm"] < 15 and r["caneca_mexeu_na_soltura_mm"] < 5 and passo_assentamento < 8 and inc_f < 5 and r["dedos_final"] == 0 and toques <= 1 and r["subida_max_mm"] > 70
                           and r["inclinacao_max_deg"] < 15 and r["quadros_mao_na_mesa"] == 0 and r["penetracao_max_mm"] < 3 and r["penetracao_caneca_disco_mm"] < 3
                           and r["margem_saida_na_boca_mm"] > 2 and vmax < 2.5 and amax < 35 and r["outros_mexeram_mm"] < 10)
        INS = "the robot uses its right hand to pick up the white mug and place it under the coffee filter"
    np.savez_compressed(out / "dados.npz", states=Sx, action=np.array(A), juntas=np.array(J))
    (out / "info.json").write_text(json.dumps(dict(r, instrucao=INS if colocar else INSTR), indent=1))
    np.savez_compressed(out / "qpos.npz", qpos=np.array(QP), dedos=np.array(ded))
    (out / "trace.json").write_text(json.dumps(dict(demo=True, copo=list(xy), afasta=afast, yaw_copo=float(np.degrees(yaw)), cintura=True, ft=False)))
    return r


def mao_yaw_por_y(y):
    """Giro da mão que pega sem tocar a mesa, da varredura em grade (x 0,24-0,27): -30° até y=-0,18, 0° em y=-0,06."""
    return float(np.clip(-30 + 250 * (y + 0.18), -30, 0))


def suave(n):   # mínimo jerk de 0 a 1 em n quadros
    t = np.linspace(0, 1, n + 1)[1:]; return 10 * t**3 - 15 * t**4 + 6 * t**5


def episodio(cena, out, xy, yaw, rng, t_pega, afast=0.05, mao_yaw=-45, cintura=False):
    """cintura: False (parada) ou True (pitch e yaw na IK junto com o braço); com ela grava 31 valores (28 + roll, pitch, yaw)."""
    sim = Sim(cena); m, d = sim.m, sim.d
    pose = np.array(POSE_HE) + rng.normal(0, 0.02, 28) * np.r_[np.zeros(14), np.ones(14)]
    _, n_sub = monta(sim, pose, xy, afast, yaw_copo=yaw, render=False)
    outros = {n: d.xpos[m.body(n).id].copy() for n in ("chaleira", "pote", "coador")}
    copo0 = d.xpos[sim.copo].copy(); z0 = copo0[2]
    J = JUNTAS + CINTURA if cintura else JUNTAS
    wc = np.zeros(3)
    if cintura:   # cintura (pitch, yaw) e braço numa IK só, com a cintura pesando metade e puxada para o tronco reto
        R0 = _rz(np.radians(mao_yaw) + 0.5 * np.arctan2(copo0[1], copo0[0])) @ _ry(np.radians(10))
        qw = sim.q(BRACO_D); nomes = ["waist_pitch_joint", "waist_yaw_joint"] + BRACO_D
        sol, ew, _ = ik(sim, copo0 - R0 @ t_pega, R0, np.r_[0, 0, qw], it=800, ref=np.r_[0, 0, qw], nomes=nomes, peso=[0.5, 0.5] + [1] * 7)
        wc = np.array([0.0, sol[0], sol[1]])
    fixa = dict(zip(CINTURA, wc))
    R_PEGA = _rz(np.radians(mao_yaw) + (0.5 * np.arctan2(copo0[1], copo0[0]) if cintura else 0)) @ _ry(np.radians(10))
    # pose de pega: fundo da caneca em t_pega no referencial do punho
    p_pega = copo0 - R_PEGA @ t_pega
    ib = [J.index(j) for j in BRACO_D]; im = [J.index(j) for j in MAO[7:]]; ic = [J.index(j) for j in CINTURA] if cintura else []
    alvo = sim.q(J).copy(); q0 = alvo[ib].copy()
    ref, e0, _ = ik(sim, p_pega, R_PEGA, q0, it=400, fixa=fixa)                    # postura da pega: todas as IK puxam para ela
    pre = p_pega - 0.07 * R_PEGA[:, 1] + [0, 0, 0.03]                 # 7 cm para o lado dos dedos e 3 cm acima
    cima = np.array([0.12, pre[1] - 0.04, 0.90])                       # antes, sobe a mão acima do tampo (começa pendurada abaixo dele)
    q_up, e1, _ = ik(sim, cima, R_PEGA, q0, it=400, ref=ref, fixa=fixa)
    w = m.body("right_wrist_yaw_link").id                               # 1o sobe sem girar a mão (girando antes os dedos batem sob o tampo)
    q_a, e2, _ = ik(sim, d.xpos[w] + [0, 0, 0.17], d.xmat[w].reshape(3, 3).copy(), q0, it=400)
    S, A, sub, ded, inc, mesa, pen, pos_copo, QP = [], [], [], [], [], [], [], [], []
    mesa_id = m.body("mesa").id

    def quadro(novo):
        S.append(sim.q(J).astype(np.float32)); A.append(novo.astype(np.float32))
        sim.alvo(J, novo); sim.passo(n_sub); QP.append(d.qpos.copy())
        pos_copo.append(d.xpos[sim.copo].copy()); sub.append(d.xpos[sim.copo][2] - z0); ded.append(sim.dedos_no_copo()); inc.append(np.degrees(np.arccos(np.clip(d.xmat[sim.copo][8], -1, 1))))
        pen.append(min([k.dist for k in d.contact[:d.ncon] if sim.copo in (m.geom_bodyid[k.geom1], m.geom_bodyid[k.geom2]) and {m.geom_bodyid[k.geom1], m.geom_bodyid[k.geom2]} & sim.mao_dir] + [0]))
        mesa.append(any(mesa_id in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]) and ({m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} & sim.mao_dir) for c in d.contact[:d.ncon]))

    for _ in range(10): quadro(alvo)                                         # parado
    a0, h0 = alvo.copy(), alvo[im].copy()
    for s in suave(30):                                                       # sobe acima do tampo
        alvo = a0.copy(); alvo[ib] = q0 + s * (q_a - q0); quadro(alvo)
    for s in suave(30):                                                       # gira a mão e abre
        alvo[ib] = q_a + s * (q_up - q_a); alvo[im] = h0 + s * (ABERTA - h0)
        if ic: alvo[ic] = a0[ic] + s * (wc - a0[ic])                         # o tronco gira e inclina junto
        quadro(alvo)
    q, erro = q_up, [e0, e1, e2]
    for s in suave(30):                                                       # vai à pré-pega
        q, e, _ = ik(sim, cima + s * (pre - cima), R_PEGA, q, it=60, ref=ref, fixa=fixa); erro.append(e); alvo[ib] = q; quadro(alvo)
    for s in suave(35):                                                       # aproxima em linha reta
        q, e, _ = ik(sim, pre + s * (p_pega - pre), R_PEGA, q, it=60, ref=ref, fixa=fixa); erro.append(e); alvo[ib] = q; quadro(alvo)
    for s in suave(25):                                                       # fecha
        alvo[im] = ABERTA + s * (FECHADA - ABERTA); quadro(alvo)
    for _ in range(10): quadro(alvo)
    for s in suave(40):                                                       # levanta 10 cm
        q, e, _ = ik(sim, p_pega + [0, 0, 0.10 * s], R_PEGA, q, it=60, ref=ref, fixa=fixa); erro.append(e); alvo[ib] = q; quadro(alvo)
    for _ in range(45): quadro(alvo)                                          # segura 1,5 s
    mexeu = max(float(np.linalg.norm(d.xpos[m.body(n).id] - p)) for n, p in outros.items())
    seg = np.array(sub[-45:]); r = dict(xy=list(xy), cintura_rpy=wc.tolist(), yaw_deg=float(np.degrees(yaw)), quadros=len(S), subida_final_mm=1000 * float(seg.mean()),
        subida_min_seg_mm=1000 * float(seg.min()), dedos_seg_min=int(min(ded[-45:])), inclinacao_max_deg=float(max(inc)),
        outros_mexeram_mm=1000 * mexeu, ik_erro_max_mm=1000 * max(erro), quadros_mao_na_mesa=int(sum(mesa)), penetracao_max_mm=-1000 * float(min(pen)), mesa_em=[i for i, x in enumerate(mesa) if x][:40])
    r["aceita"] = bool(r["subida_min_seg_mm"] > 70 and r["dedos_seg_min"] >= 2 and r["inclinacao_max_deg"] < 15 and r["outros_mexeram_mm"] < 10 and r["quadros_mao_na_mesa"] == 0 and r["ik_erro_max_mm"] < 5 and r["penetracao_max_mm"] < 3)
    np.savez_compressed(out / "dados.npz", states=np.array(S), action=np.array(A), juntas=np.array(J))
    (out / "info.json").write_text(json.dumps(dict(r, instrucao=INSTR), indent=1))
    # qpos completo + trace no formato de ../base/rodar_mujoco.py, para re-renderizar a demo de fora (render_psi0_ft.py)
    np.savez_compressed(out / "qpos.npz", qpos=np.array(QP), dedos=np.array(ded))
    (out / "trace.json").write_text(json.dumps(dict(demo=True, copo=list(xy), afasta=afast, yaw_copo=float(np.degrees(yaw)), cintura=cintura, ft=False)))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out"); ap.add_argument("--n", type=int, default=60); ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--x", type=float, nargs=2, help="faixa x; padrão depende do modo")
    ap.add_argument("--y", type=float, nargs=2, help="faixa y; padrão depende do modo")
    ap.add_argument("--yaw", type=float, default=20, help="± graus de giro da alça em torno dos 90° da cena")
    ap.add_argument("--t-pega", type=float, nargs=3, default=[0.13, 0.065, -0.05], help="fundo da caneca no referencial do punho na pega (-0,07 punha o indicador a 9,9 cm do fundo, acima da borda de 9 cm; -0,05 deixa as pontas a 5,1/2,0/7,7 cm)")
    ap.add_argument("--afasta", type=float, default=0.05, help="mesa e objetos para longe do robô; a avaliação tem de usar o mesmo")
    ap.add_argument("--mao-yaw", type=float, help="giro da mão na pega (graus em torno do z do mundo); sem ele, regra por y (mao_yaw_por_y)")
    ap.add_argument("--pos", type=float, nargs=2, help="posição fixa da caneca (antes do afastamento), para varreduras")
    ap.add_argument("--cintura", action="store_true", help="cintura liberada (IK conjunta), grava 31 valores; região e mesa da v3 "
                    "(mesa a 0,15, x 0,24-0,30, y -0,22 a -0,04); tenta a mão a -15, 0 e -30° e fica com a primeira aceita")
    ap.add_argument("--colocar", action="store_true", help="com --humano: etapa 2, pega e põe a caneca sob o coador (alça 120-150°, y <= -0,08)")
    ap.add_argument("--humano", action="store_true", help="com --cintura: gesto contínuo de gente e punho quase reto (episodio_humano)")
    ap.add_argument("--fixo", action="store_true", help="sem sorteio: caneca na posição padrão (teste)")
    a = ap.parse_args()
    if a.x is None: a.x = [0.24, 0.30] if a.cintura else [0.24, 0.28]
    if a.y is None: a.y = [-0.22, -0.04] if a.cintura else [-0.22, -0.07]
    if a.cintura: a.afasta = 0.15
    if a.colocar and a.y == [-0.22, -0.04]: a.y = [-0.22, -0.08]   # preserva faixa explícita passada pelo usuário
    if a.humano: a.afasta = 0.10   # robô 5 cm mais perto da mesa: tronco 8-16° em vez de 16-24° (revisão das demos v4, 08/10/2026)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(a.semente)
    try:
        from taskline import Progress
    except ImportError:
        Progress = None
    prog = Progress(f"demos-{out.name}", total=a.n, label="demos Ψ0 caneca") if Progress else None
    if prog: prog.__enter__()
    res = []
    for i in range(a.n):
        xy = tuple(a.pos) if a.pos else (0.28, -0.20) if a.fixo else (float(rng.uniform(*a.x)), float(rng.uniform(*a.y)))
        yaw = np.pi / 2 + (0 if a.fixo or a.pos else np.radians(rng.uniform(-a.yaw, a.yaw)))
        ep = out / f"ep{i:03d}"; ep.mkdir(exist_ok=True)
        if a.humano:
            if a.colocar: yaw = np.radians(rng.uniform(55, 78))   # alça entre o polegar (~165°) e os dedos (~-25°), do lado de fora da mão: com 145-170 o polegar entrava no vão da alça e arrastava a caneca ao recuar (08/10); gira até ~30° no transporte; acima de ~77° a alça ainda bate na haste (4 de 12 na v9)
            r = episodio_humano(CENA, ep, xy, yaw, rng, np.array(a.t_pega), a.afasta, colocar=a.colocar); res.append(r)
            print(f"ep{i:03d} xy=({xy[0]:.3f},{xy[1]:.3f}) subida={r['subida_min_seg_mm']:.0f}mm dedos={r['dedos_seg_min']} incl={r['inclinacao_max_deg']:.1f} "
                  f"mesa={r['quadros_mao_na_mesa']} pen={r['penetracao_max_mm']:.1f} ik={r['ik_erro_max_mm']:.1f}mm punho_pega={r['punho_na_pega_deg']} "
                  f"punho_max={r['punho_max_deg']:.0f} cint={np.degrees(r['cintura_rpy']).round(0).tolist()} " + (f"erro={r['erro_colocar_mm']:.0f}mm incl_f={r['inclinacao_final_deg']:.1f} soltura={r['caneca_mexeu_na_soltura_mm']:.1f}mm coador={r['quadros_tocando_coador']} tronco={r['tronco_max_deg']} " if 'erro_colocar_mm' in r else '') + f"aceita={r['aceita']}", flush=True)
            (out / "resumo.json").write_text(json.dumps(res, indent=1))
            if prog: prog.advance()
            continue
        maos = [a.mao_yaw] if a.mao_yaw is not None else [-15, 0, -30] if a.cintura else [mao_yaw_por_y(xy[1])]
        for mao in maos:
            r = episodio(CENA, ep, xy, yaw, rng, np.array(a.t_pega), a.afasta, mao, cintura=a.cintura); r["mao_yaw"] = mao
            if r["aceita"]: break
        res.append(r)
        print(f"ep{i:03d} xy=({xy[0]:.3f},{xy[1]:.3f}) yaw={r['yaw_deg']:.0f} subida={r['subida_min_seg_mm']:.0f}mm dedos={r['dedos_seg_min']} "
              f"incl={r['inclinacao_max_deg']:.1f} outros={r['outros_mexeram_mm']:.1f}mm ik={r['ik_erro_max_mm']:.1f}mm mesa={r['quadros_mao_na_mesa']} pen={r['penetracao_max_mm']:.1f}mm aceita={r['aceita']}", flush=True)
        (out / "resumo.json").write_text(json.dumps(res, indent=1))
        if prog: prog.advance()
    if prog: prog.__exit__(None, None, None)
    print(f"aceitas {sum(r['aceita'] for r in res)}/{len(res)}")


if __name__ == "__main__":
    main()
