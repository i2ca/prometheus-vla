"""Ψ0 BASE (sem ajuste por tarefa): VLM pré-treinado pre.fast.1by1.2601091803.ckpt.ego200k.he30k + action expert
pós-treinado postpre.1by1.pad36.2601131206.ckpt.he30k, montados como no início do finetune (trainers/finetune.py
init_models) e com a configuração exata do pós-treino (scripts/train/psi0/posttrain-he-psi0.sh): estado cru
(mãos 14 + braços 14, sem normalizar), saída = 16 ações RELATIVAS normalizadas por q99 (assets/stats/he_raw_rel_stats_combined_no_static.json),
imagem egocêntrica redimensionada para 240x320.
Protocolo: socket TCP local, mensagens pickle com prefixo de tamanho. Pedido {image uint8 HxWx3 RGB, state float32[28], instruction, steps};
resposta {delta float32[16,28] (rad por passo, já desnormalizado), lat_s}. Uso: PSI0_REPO=<clone do Psi0> PSI0_PESOS=<pasta dos pesos> python servidor_base.py [porta]
"""
import os, sys, socket, pickle, struct, time
B = os.path.abspath(os.environ.get("PSI0_PESOS", "pesos"))   # pasta com pre/ e postpre/ (ver README)
os.chdir(os.environ.get("PSI0_REPO", "Psi0")); sys.path.insert(0, "src")   # clone do Psi0 com o .venv-psi ativo
import numpy as np, torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from psi.utils import parse_args_to_tyro_config, pad_to_len
from psi.models.psi0 import Psi0Model
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
cfg = parse_args_to_tyro_config("scripts/train/psi0/posttrain-he-psi0.sh")
vlm = Qwen3VLForConditionalGeneration.from_pretrained(f"{B}/pre", attn_implementation="sdpa", dtype=torch.bfloat16)
model = Psi0Model(model_cfg=cfg.model, vlm_model=vlm)
sd = load_file(f"{B}/postpre/action_header.safetensors")
miss, unexp = model.action_header.load_state_dict(sd, strict=False)
print(f"action header: {len(sd)} chaves, faltando {len(miss)}, sobrando {len(unexp)}", miss[:5], unexp[:5], flush=True)
model.vlm_processor = AutoProcessor.from_pretrained(f"{B}/pre")
model.noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=cfg.model.train_diffusion_steps)
model.action_horizon, model.action_dim, model.device = cfg.model.action_chunk_size, cfg.model.action_dim, "cuda:0"
model.to("cuda:0").eval()
field, mt = cfg.data.transform.field, cfg.data.transform.model
from torchvision.transforms import v2
prep = v2.Compose([mt.resize(), mt.center_crop()])
print("pronto: chunk", cfg.model.action_chunk_size, "dim", cfg.model.action_dim, "norma", field.action_norm_type, "normaliza estado", field.normalize_state, flush=True)

def recv(c):
    n = struct.unpack("!I", c.recv(4, socket.MSG_WAITALL))[0]; buf = b""
    while len(buf) < n: buf += c.recv(n - len(buf))
    return pickle.loads(buf)

def send(c, obj):
    b = pickle.dumps(obj); c.sendall(struct.pack("!I", len(b)) + b)

s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", PORT)); s.listen(1)
while True:
    c, _ = s.accept()
    try:
        while True:
            q = recv(c); t0 = time.perf_counter()
            st = pad_to_len(np.asarray(q["state"], np.float32), 36, dim=0)[0][None, None]
            with torch.inference_mode():
                a = model.predict_action(observations=[[prep(Image.fromarray(q["image"]))]], states=torch.from_numpy(st).cuda(),
                                         instructions=[q["instruction"].lower()], num_inference_steps=q.get("steps", 10), traj2ds=None)
            a = field.denormalize(a[0].float().cpu().numpy())[:, :28]
            send(c, {"delta": a.astype(np.float32), "lat_s": time.perf_counter() - t0})
    except (ConnectionError, struct.error, EOFError):
        c.close()
