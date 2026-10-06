"""Servidor do checkpoint real da Task 3 do Ψ0 para malha fechada no MuJoCo, com o controlador RTC do servidor oficial
(serve_psi0_amo.RealTimeChunkController) feito de forma síncrona: o 1o bloco sai de predict_action; os seguintes,
de predict_action_with_training_rtc_flow com o resto do bloco anterior como prefixo (atraso d) e o rpy/altura do estado
trocados pela ação de 2 passos antes, como no oficial. Socket local, pickle com prefixo de tamanho.
Pedido {image uint8 480x640x3, state float32[28] (mãos 14 + braços 14), primeiro bool, k int (ações executadas do bloco atual)};
resposta {acoes float32[30,36] desnormalizadas, lat_s}.
Uso: PSI0_REPO=<clone do Psi0> PSI0_CKPT=<pasta do checkpoint task3> python servidor_task3.py [porta]
"""
import os, sys, socket, pickle, struct, time
CKPT = os.path.abspath(os.environ.get("PSI0_CKPT", "task3"))
os.chdir(os.environ.get("PSI0_REPO", "Psi0")); sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, torch
from PIL import Image
from psi.deploy.serve_psi0_amo import Server
from psi.utils import pad_to_len

PORTA, D, INSTR = (int(sys.argv[1]) if len(sys.argv) > 1 else 8778), 6, "g1/pick_bottle_and_turn_and_pour_into_cup"
srv = Server(policy="psi", run_dir=Path(CKPT), ckpt_step=40000, device="cuda:0", enable_rtc=True); M = srv.model
print("pronto", flush=True)

def obs(est, rpyh_norm=None):
    o = srv.maxmin.normalize_state_func(pad_to_len(np.concatenate([est, [0, 0, 0, 0.75]]).astype(np.float32), 36, dim=0)[0])
    if rpyh_norm is not None: o[28:32] = rpyh_norm   # como o oficial: rpy/altura da ação anterior, já normalizados
    return torch.from_numpy(o[None, None].astype(np.float32)).to(srv.device)

def recv(c):
    n = struct.unpack("!I", c.recv(4, socket.MSG_WAITALL))[0]; buf = b""
    while len(buf) < n: buf += c.recv(n - len(buf))
    return pickle.loads(buf)

s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", PORTA)); s.listen(1)
while True:
    c, _ = s.accept(); A = None
    try:
        while True:
            q = recv(c); t0 = time.perf_counter(); img = [srv.preprocess_image({"cam0": Image.fromarray(q["image"])})["cam0"]]
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
