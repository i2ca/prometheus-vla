"""Servidor do checkpoint real da Task 3 do Ψ0 para malha fechada no MuJoCo, com o controlador RTC do servidor oficial
(serve_psi0_amo.RealTimeChunkController) feito de forma síncrona: o 1o bloco sai de predict_action; os seguintes,
de predict_action_with_training_rtc_flow com o resto do bloco anterior como prefixo (atraso d) e o rpy/altura do estado
trocados pela ação de 2 passos antes, como no oficial. Socket local, pickle com prefixo de tamanho.
Pedido {images {head,left_wrist,right_wrist} (ajuste fino) ou image (Task 3): uint8 480x640x3 ou bytes JPEG (20x menos rede), state float32[28] (mãos 14 + braços 14) ou [31] (+ cintura roll, pitch, yaw), primeiro bool, k int (ações executadas do bloco atual)};
resposta {acoes float32[30,36] desnormalizadas, lat_s}.
Uso: PSI0_REPO=<clone do Psi0> PSI0_CKPT=<pasta do checkpoint task3> python servidor_task3.py [porta]
Ajuste fino da caneca (../finetune): PSI0_CKPT=<run> PSI0_PASSO=<passo> PSI0_INSTR="<frase do dataset>" PSI0_SEM_CORPO=1
(sem pernas: o estado não recebe rpy/altura da ação anterior, como no treino, em que essas dimensões são zero).
"""
import io, os, sys, socket, pickle, struct, time
CKPT = os.path.abspath(os.environ.get("PSI0_CKPT", "task3"))
os.chdir(os.environ.get("PSI0_REPO", "Psi0")); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, torch
from PIL import Image
from psi.deploy.serve_psi0_amo import Server
from psi.utils import pad_to_len
import psi.models.psi0 as _psi0
# O checkpoint da Task 3 (jan/2026) foi treinado com a posição somada em TODOS os tokens [VLM..., estado] (código de 59b3fef);
# o commit 5236e63 (sonic, 30/08/2026) passou a somar só no token de estado, o que derruba a política (paridade.py: perda 3,2-3,8 -> 0,3-1,8).
_psi0.ObservationProjection._add_obs_pos = lambda self, s_t, n: s_t + self.enc_pos(s_t)

PORTA, D = (int(sys.argv[1]) if len(sys.argv) > 1 else 8778), 6
INSTR = os.environ.get("PSI0_INSTR", "g1/pick_bottle_and_turn_and_pour_into_cup"); SEM_CORPO = os.environ.get("PSI0_SEM_CORPO") == "1"
srv = Server(policy="psi", run_dir=Path(CKPT), ckpt_step=int(os.environ.get("PSI0_PASSO", 40000)), device="cuda:0", enable_rtc=True); M = srv.model
# O predict_action do checkout de jan/2026 agrupa o grid multicâmera como [B,N,3]; o Qwen recebe [B*N,3].
_vlm_forward = M.vlm_model.forward
def _forward_multicam(*a, **k):
    grid = k.get("image_grid_thw")
    if grid is not None and grid.ndim == 3: k["image_grid_thw"] = grid.flatten(0, 1)
    return _vlm_forward(*a, **k)
M.vlm_model.forward = _forward_multicam
print("pronto", flush=True)

def obs(est, rpyh_norm=None):
    est = np.concatenate([est, [0, 0, 0, 0.75]]) if len(est) == 28 else est   # 31 = com a cintura em rpy (ajuste fino com a cintura)
    o = srv.maxmin.normalize_state_func(pad_to_len(est.astype(np.float32), 36, dim=0)[0])
    if rpyh_norm is not None and not SEM_CORPO: o[28:32] = rpyh_norm   # como o oficial: rpy/altura da ação anterior, já normalizados
    return torch.from_numpy(o[None, None].astype(np.float32)).to(srv.device)

def imagem(x):
    return Image.open(io.BytesIO(x)).convert("RGB") if isinstance(x, bytes) else Image.fromarray(x)

def recv(c):
    n = struct.unpack("!I", c.recv(4, socket.MSG_WAITALL))[0]; buf = b""
    while len(buf) < n: buf += c.recv(n - len(buf))
    return pickle.loads(buf)

s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", PORTA)); s.listen(1)
while True:
    c, _ = s.accept(); A = None
    try:
        while True:
            q = recv(c); t0 = time.perf_counter()
            if "images" in q:   # ajuste fino multicâmera
                nomes = ("head", "left_wrist", "right_wrist"); prep = srv.preprocess_image({n: imagem(q["images"][n]) for n in nomes})
                img = [[prep[n][0] for n in nomes]]
            else:   # Task 3 (rodar_mujoco_task3.py): só a câmera da cabeça
                img = [srv.preprocess_image({"cam0": imagem(q["image"])})["cam0"]]
            with torch.inference_mode():
                if q["primeiro"] or A is None:
                    a = M.predict_action(observations=img, states=obs(q["state"]), traj2ds=None, instructions=[INSTR], num_inference_steps=8)
                else:
                    k = q["k"]; prev = np.concatenate([A[k:], np.zeros((k, A.shape[1]), A.dtype)])
                    a = M.predict_action_with_training_rtc_flow(observations=img, states=obs(q["state"], A[k - 2, 28:32]), traj2ds=None, instructions=[INSTR],
                            num_inference_steps=8, prev_actions=torch.from_numpy(prev[None]).to(srv.device), inference_delay=D, max_delay=8)
            A = a[0].float().cpu().numpy()
            b = pickle.dumps({"acoes": srv._postprocess_action(A).astype(np.float32), "lat_s": time.perf_counter() - t0}); c.sendall(struct.pack("!I", len(b)) + b)
    except (ConnectionError, struct.error, EOFError):
        c.close()
