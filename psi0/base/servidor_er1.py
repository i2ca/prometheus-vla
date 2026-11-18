"""UnifoLM-ER-1-4B (Unitree, sobre Qwen3-VL-4B) servindo apontamento de objeto por HTTP, na GPU do notebook.

POST /apontar {"jpg": base64, "objeto": "white mug"} -> {"u": px, "v": px, "texto": resposta crua}
Qwen3-VL devolve pontos em coordenadas 0-1000 da imagem; aqui viram pixels.
Uso (conda unifolm-er1): python servidor_er1.py [--modelo ~/I2CA/modelos/UnifoLM-ER-1] [--porta 8796] [--bits 4|16]
"""
import argparse, base64, io, json, re, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig

PEDIDO = ('Point to the {obj} in the image. Output the point coordinates in JSON format: '
          '[{{"point_2d": [x, y], "label": "{obj}"}}]')


def carrega(caminho, bits):
    q = None if bits == 16 else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    m = Qwen3VLForConditionalGeneration.from_pretrained(caminho, dtype=torch.bfloat16, quantization_config=q, device_map="cuda:0").eval()
    return m, AutoProcessor.from_pretrained(caminho)


def aponta(m, proc, img, obj):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PEDIDO.format(obj=obj)}]}]
    ent = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(m.device)
    with torch.inference_mode():
        out = m.generate(**ent, max_new_tokens=128, do_sample=False)
    texto = proc.batch_decode(out[:, ent["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    p = re.search(r'"point_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)', texto)
    if not p: return None, None, texto
    w, h = img.size
    return float(p.group(1)) / 1000 * w, float(p.group(2)) / 1000 * h, texto


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--modelo", default=str(Path.home() / "I2CA/modelos/UnifoLM-ER-1"))
    ap.add_argument("--porta", type=int, default=8796); ap.add_argument("--bits", type=int, choices=(4, 16), default=4); a = ap.parse_args()
    m, proc = carrega(a.modelo, a.bits); trava = threading.Lock()
    print(f"ER-1 carregado ({a.bits} bits), {torch.cuda.memory_allocated() / 2**30:.1f} GiB na GPU; porta {a.porta}", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x): pass

        def do_POST(self):
            q = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            img = Image.open(io.BytesIO(base64.b64decode(q["jpg"]))).convert("RGB")
            with trava: u, v, t = aponta(m, proc, img, q.get("objeto", "white mug"))
            corpo = json.dumps({"u": u, "v": v, "texto": t}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(corpo)

    ThreadingHTTPServer(("127.0.0.1", a.porta), H).serve_forever()


if __name__ == "__main__":
    main()
