"""Câmera virtual da cabeça: reprojeta a imagem da câmera na pose real (cor + profundidade) para a pose de treino do v24.
Splat com z-buffer (o ponto mais perto ganha) e preenchimento dos buracos por inpaint. Câmeras no padrão do MuJoCo (olham para -z, y para cima).
quantiza=True imita o D435 do robô: profundidade em degraus de 2000/255 mm."""
import numpy as np, cv2


def reprojeta(rgb, prof, o_r, R_r, fovy_r, o_t, R_t, fovy_t, quantiza=False):
    h, w = prof.shape
    if quantiza: prof = np.round(prof * 1000 / (2000 / 255)) * (2000 / 255) / 1000
    fr = 0.5 * h / np.tan(np.radians(fovy_r) / 2); ft = 0.5 * h / np.tan(np.radians(fovy_t) / 2)
    vv, uu = np.mgrid[0:h, 0:w]; z = prof.ravel()
    cam = np.stack([(uu.ravel() - w / 2) / fr * z, -(vv.ravel() - h / 2) / fr * z, -z], 1)
    P = cam @ R_r.T + o_r                       # mundo
    pc = (P - o_t) @ R_t                         # câmera de treino (R_t.T @ (P - o_t))
    d = -pc[:, 2]; ok = d > 0.05
    x = np.round(w / 2 + ft * pc[:, 0] / np.where(ok, d, 1)).astype(int); y = np.round(h / 2 - ft * pc[:, 1] / np.where(ok, d, 1)).astype(int)
    ok &= (x >= 0) & (x < w) & (y >= 0) & (y < h)
    idx = np.flatnonzero(ok); ordem = idx[np.argsort(-d[idx])]   # do mais longe para o mais perto: o último escrito (mais perto) fica
    out = np.zeros_like(rgb); cheio = np.zeros((h, w), np.uint8)
    cor = rgb.reshape(-1, 3)
    out[y[ordem], x[ordem]] = cor[ordem]; cheio[y[ordem], x[ordem]] = 1
    buraco = (cheio == 0).astype(np.uint8)
    return cv2.inpaint(out, buraco, 3, cv2.INPAINT_TELEA), buraco.astype(bool)
