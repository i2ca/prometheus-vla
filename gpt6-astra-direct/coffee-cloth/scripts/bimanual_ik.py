"""IK bimanual: resolve cintura e os dois bracos de uma vez, para dois alvos de palma.

Existe porque duas ArmIK independentes nao servem. Cada uma inclui `waist_yaw_joint` no seu
encadeamento e resolve um alvo em lado oposto do objeto, entao pedem valores opostos para a MESMA
junta: medido na jarra, +118,9 graus pela direita e -111,1 pela esquerda. O segundo `set_targets`
vence e a cintura vai para -111, levando o braco direito para 820 mm atras do objeto, com a IK
reportando 0,00 mm de erro porque a palma dela estava no alvo que ela mesma resolveu.

Tirar a cintura das duas tambem nao serve: sem ela os bracos nao alcancam. Varri a mesa inteira com
as palmas a 175 mm do eixo e o menor erro foi 30 mm; com o objeto centrado a frente e a cintura fixa
no melhor valor, 73,2 mm.

Aqui a cintura e uma coluna compartilhada da jacobiana empilhada: as duas palmas puxam a mesma junta
e a solucao de minimos quadrados amortecidos negocia entre elas.
"""
import numpy as np
import cv2
import mujoco


class BimanualIK:
    def __init__(self, sim, juntas_comuns, juntas_dir, juntas_esq,
                 palma_dir="right_wrist_yaw_link", palma_esq="left_wrist_yaw_link"):
        self.sim, self.m = sim, sim.m
        self.d = mujoco.MjData(self.m)
        self.nomes = list(juntas_comuns) + list(juntas_dir) + list(juntas_esq)
        self.n_com = len(juntas_comuns)
        self.n_dir = len(juntas_dir)
        self.idx_dir = list(range(self.n_com)) + list(range(self.n_com, self.n_com + self.n_dir))
        self.idx_esq = list(range(self.n_com)) + list(range(self.n_com + self.n_dir, len(self.nomes)))
        ids = [self.m.joint(n).id for n in self.nomes]
        self.qa = self.m.jnt_qposadr[ids]
        self.va = self.m.jnt_dofadr[ids]
        self.bounds = self.m.jnt_range[ids]
        self.body_dir = self.m.body(palma_dir).id
        self.body_esq = self.m.body(palma_esq).id
        self.nullspace_gain = 0.02
        self.peso_orientacao = 0.15

    def fk(self, q):
        self.d.qpos[:] = self.sim.d.qpos
        self.d.qpos[self.qa] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return (self.d.xpos[self.body_dir].copy(), self.d.xmat[self.body_dir].reshape(3, 3).copy(),
                self.d.xpos[self.body_esq].copy(), self.d.xmat[self.body_esq].reshape(3, 3).copy())

    def solve(self, alvo_dir, rot_dir, alvo_esq, rot_esq, q, referencia=None,
              iteracoes=300, passo_max=None):
        q0 = np.clip(np.asarray(q, float), self.bounds[:, 0], self.bounds[:, 1])
        q = q0.copy()
        ref = q.copy() if referencia is None else np.asarray(referencia, float)
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        w = self.peso_orientacao
        for it in range(iteracoes):
            pd, Rd, pe, Re = self.fk(q)
            epd = np.asarray(alvo_dir) - pd
            epe = np.asarray(alvo_esq) - pe
            erd = cv2.Rodrigues(np.asarray(rot_dir, dtype=np.float64) @ Rd.T)[0].ravel()
            ere = cv2.Rodrigues(np.asarray(rot_esq, dtype=np.float64) @ Re.T)[0].ravel()
            if (np.linalg.norm(epd) < 0.0005 and np.linalg.norm(epe) < 0.0005
                    and np.linalg.norm(erd) < 0.02 and np.linalg.norm(ere) < 0.02):
                break
            # jacobiana empilhada: 12 linhas (posicao e orientacao das duas palmas) por n juntas.
            # As colunas das juntas comuns recebem contribuicao dos DOIS bracos, e e isso que faz a
            # cintura ser negociada em vez de disputada.
            J = np.zeros((12, len(self.nomes)))
            mujoco.mj_jac(self.m, self.d, jp, jr, pd, self.body_dir)
            J[0:3, self.idx_dir] = jp[:, self.va[self.idx_dir]]
            J[3:6, self.idx_dir] = w * jr[:, self.va[self.idx_dir]]
            mujoco.mj_jac(self.m, self.d, jp, jr, pe, self.body_esq)
            J[6:9, self.idx_esq] = jp[:, self.va[self.idx_esq]]
            J[9:12, self.idx_esq] = w * jr[:, self.va[self.idx_esq]]
            erro = np.r_[epd, w * erd, epe, w * ere]
            inv = J.T @ np.linalg.solve(J @ J.T + 0.004 ** 2 * np.eye(12), np.eye(12))
            delta = inv @ erro
            delta += self.nullspace_gain * (np.eye(len(q)) - inv @ J) @ (ref - q)
            delta *= min(1.0, 0.08 / max(np.max(np.abs(delta)), 1e-12))
            q = np.clip(q + delta, self.bounds[:, 0], self.bounds[:, 1])
            if passo_max is not None:
                q = q0 + np.clip(q - q0, -passo_max, passo_max)
        pd, Rd, pe, Re = self.fk(q)
        return q, {
            "erro_dir_mm": float(np.linalg.norm(np.asarray(alvo_dir) - pd)) * 1000,
            "erro_esq_mm": float(np.linalg.norm(np.asarray(alvo_esq) - pe)) * 1000,
            "orient_dir_deg": float(np.degrees(np.linalg.norm(cv2.Rodrigues(np.asarray(rot_dir, dtype=np.float64) @ Rd.T)[0]))),
            "orient_esq_deg": float(np.degrees(np.linalg.norm(cv2.Rodrigues(np.asarray(rot_esq, dtype=np.float64) @ Re.T)[0]))),
            "iteracoes": it + 1,
        }
