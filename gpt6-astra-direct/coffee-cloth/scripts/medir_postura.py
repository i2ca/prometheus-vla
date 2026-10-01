"""Mede a postura do braco esquerdo ao longo de uma trajetoria executada.

Os numeros que denunciaram o despejo antinatural: cotovelo acima do ombro,
abducao alta, braco travado (distancia ombro-punho constante, cotovelo quase
parado) e o movimento saindo de rotacao do umero e torcao do punho.

uso: medir_postura.py <pasta-com-report-e-trajectory> [...]
"""
import json, sys
from pathlib import Path
import numpy as np, mujoco

JUNTAS = ['left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint']


def medir(pasta):
    r = json.loads((pasta / 'report.json').read_text())
    m = mujoco.MjModel.from_xml_path(r['scene']); d = mujoco.MjData(m)
    T = np.load(pasta / 'trajectory.npz')['qpos']
    t, o, c, p = [m.body(n).id for n in ['torso_link', 'left_shoulder_roll_link',
                                          'left_elbow_link', 'left_wrist_yaw_link']]
    ad = [m.jnt_qposadr[m.joint(n).id] for n in JUNTAS]
    alt, abd, ext, elev, plano = [], [], [], [], []
    for q in T[::4]:
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        Rt = d.xmat[t].reshape(3, 3)
        v = Rt.T @ (d.xpos[c] - d.xpos[o])
        alt.append(v[2] * 1000)
        abd.append(np.rad2deg(np.arctan2(abs(v[1]), -v[2])))
        # A 'abducao' acima mistura dois gestos: braco aberto para o lado e
        # braco erguido a frente dao os dois ~180 graus quando o cotovelo passa
        # do ombro. Separando: elevacao = angulo do umero a partir da vertical
        # para baixo; plano = para onde ele aponta, 0 = frente, 90 = lado.
        u = v / np.linalg.norm(v)
        elev.append(np.rad2deg(np.arccos(np.clip(-u[2], -1, 1))))
        plano.append(np.rad2deg(np.arctan2(u[1], u[0])))
        ext.append(np.linalg.norm(d.xpos[p] - d.xpos[o]) * 1000)
    Q = np.rad2deg(T[:, ad])
    amp = Q.max(0) - Q.min(0)
    return {'cotovelo_acima_do_ombro_mm_max': round(max(alt), 1),
            'elevacao_umero_deg_mediana': round(float(np.median(elev)), 1),
            'elevacao_umero_deg_max': round(max(elev), 1),
            'plano_elevacao_deg_mediana_0frente_90lado': round(float(np.median(plano)), 1),
            'abducao_deg_max': round(max(abd), 1),
            'abducao_deg_mediana': round(float(np.median(abd)), 1),
            'variacao_ombro_punho_mm': round(max(ext) - min(ext), 1),
            'amplitude_giro_umero_deg': round(float(amp[0]), 1),
            'amplitude_cotovelo_deg': round(float(amp[1]), 1),
            'amplitude_torcao_punho_deg': round(float(amp[2]), 1)}


def por_fase(pasta):
    """Amplitude das juntas e movimento da mao em cada fase da trajetoria."""
    r = json.loads((pasta / 'report.json').read_text())
    m = mujoco.MjModel.from_xml_path(r['scene']); d = mujoco.MjData(m)
    T = np.load(pasta / 'trajectory.npz')['qpos']
    fases = [x.get('phase', '?') for x in r['rows']][:len(T)]
    ad = [m.jnt_qposadr[m.joint(n).id] for n in JUNTAS]
    p = m.body('left_wrist_yaw_link').id
    saida = {}
    for f in dict.fromkeys(fases):
        idx = [i for i, x in enumerate(fases) if x == f]
        Q = np.rad2deg(T[idx][:, ad])
        pos = []
        for i in idx[::2]:
            d.qpos[:] = T[i]; mujoco.mj_kinematics(m, d); pos.append(d.xpos[p].copy())
        pos = np.array(pos)
        caminho = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))) if len(pos) > 1 else 0.
        saida[f] = {'estados': len(idx), 'umero': round(float(np.ptp(Q[:, 0])), 1),
                    'cotovelo': round(float(np.ptp(Q[:, 1])), 1),
                    'punho': round(float(np.ptp(Q[:, 2])), 1),
                    'mao_percorre_mm': round(caminho * 1000, 0),
                    'mao_desloca_mm': round(float(np.linalg.norm(pos[-1] - pos[0])) * 1000, 0)}
    return saida


if __name__ == '__main__':
    fase = '--fases' in sys.argv
    for arg in [x for x in sys.argv[1:] if x != '--fases']:
        print(arg, json.dumps(medir(Path(arg)), ensure_ascii=False))
        if fase:
            for f, v in por_fase(Path(arg)).items():
                print('   ', f.ljust(20), json.dumps(v, ensure_ascii=False))
