"""Ψ0 ajustado (../finetune) no G1 REAL: mesmo protocolo do rodar_mujoco.py --ft, com o UnitreeG1Dex3 no lugar do MuJoCo.

Servidor: ../task3/servidor_task3.py na atena (GPU 1), porta 8781, escutando só em 127.0.0.1 -> túnel: ssh -L 8781:127.0.0.1:8781 <atena>
Robô: run_g1_server.py + a câmera da cabeça (ZMQ :5555) rodando nele; este script roda numa máquina que enxerga o robô e o túnel.
Por padrão é DRY-RUN: lê, pergunta ao servidor e imprime/grava o que mandaria, sem enviar nada ao robô. --enviar liga o comando de verdade
(pede Enter antes do 1o bloco; Ctrl+C para de enviar e solta o robô).

Uso:  python rodar_real.py --out saida/real01 --cintura --executa 15 --blocos 30          # dry-run
      python rodar_real.py --out saida/real01 --cintura --executa 15 --blocos 30 --enviar  # comanda o robô

Limites conhecidos (ler antes de --enviar):
 - O modelo da etapa 2 foi treinado com 3 câmeras (head, left_wrist, right_wrist). O bridge só publica a da cabeça; --punho decide o que
   vai no lugar das duas de punho (cabeca = repete a imagem da cabeça; preto = quadro preto). Ambos estão fora da distribuição de treino.
 - Cintura: o robô em upper_body não lê nem comanda a cintura. Com --cintura o estado manda cintura = 0 e as ações da cintura são descartadas,
   mas o modelo foi treinado com a cintura se mexendo; o braço compensa uma inclinação que não vai acontecer.
 - Mão esquerda e braço esquerdo: o G1 do lab tem o esquerdo com kp=0 (hardware); o estado esquerdo lido é o que a máquina devolve.
"""
import argparse, json, pickle, socket, struct, sys, time, datetime as dt
from pathlib import Path
import numpy as np, cv2

# Ordem do vetor do Ψ0 (igual ao rodar_mujoco.py): mãos 14 (esq: thumb0-2, middle0-1, index0-1; dir idem), braços 14 (esq 7, dir 7), cintura 3.
# Aqui as juntas são endereçadas pelo NOME do robô, então a ordem física dos dedos da direita (índice antes do médio no DDS) não importa.
MAO = [f"{l}_hand_{d}_{i}_joint" for l in ("left", "right") for d, n in (("thumb", 3), ("middle", 2), ("index", 2)) for i in range(n)]
MAX_VEL, MAX_ACC = 2.5 / 30, 35 / 900   # rad por passo de 30 Hz: mesmos limites do rodar_mujoco.py


def nomes_robo():
    from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex
    return [f"{n}.q" for n in MAO] + [f"{m.name}.q" for m in G1_29_JointArmIndex]


def pede(s, imagens, est, k):
    b = pickle.dumps({"images": imagens, "state": est, "primeiro": k == 0, "k": k}); s.sendall(struct.pack("!I", len(b)) + b)
    n = struct.unpack("!I", s.recv(4, socket.MSG_WAITALL))[0]; buf = b""
    while len(buf) < n: buf += s.recv(n - len(buf))
    return pickle.loads(buf)


def cabeca(img, bgr):
    """Câmera real 848x480 -> 640x480 central (mesmo FOV vertical de 42,5° da D435 da cena), RGB uint8."""
    h, w = img.shape[:2]
    if w * 3 != h * 4: x0 = (w - h * 4 // 3) // 2; img = img[:, x0:x0 + h * 4 // 3]
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img[:, :, ::-1] if bgr else img)


def rodar(robo, s, a, nomes, log=print):
    out = Path(a.out); (out / "entradas").mkdir(parents=True, exist_ok=True)
    ler = lambda o: np.array([o[n] for n in nomes], dtype=np.float32)
    est_robo = ler(robo.get_observation()); alvo = est_robo.copy(); passo_ant = np.zeros_like(alvo)
    trace = []
    if a.enviar: input("Robô pronto, você está do lado para desligar? Enter para começar (Ctrl+C para sair)... ")
    for b in range(a.blocos):
        t0 = time.perf_counter(); obs = robo.get_observation()
        est_robo = ler(obs); est = np.concatenate([est_robo, np.zeros(3, np.float32)]) if a.cintura else est_robo
        h = cabeca(obs["head_camera"], a.bgr); p = h if a.punho == "cabeca" else np.zeros_like(h)
        imagens = {"head": h, "left_wrist": p, "right_wrist": p}
        cv2.imwrite(f"{a.out}/entradas/bloco_{b:03d}_head.jpg", h[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        w0 = dt.datetime.now().isoformat(timespec="milliseconds"); r = pede(s, imagens, est, 0 if b == 0 else a.executa)
        acoes = r["acoes"][:, :len(nomes)]; filtradas = []
        if b == 0: alvo = est_robo.copy()   # parte do que o robô mede, não de onde o modelo acha que ele está
        for acao in acoes:
            passo = np.clip(acao - alvo, -MAX_VEL, MAX_VEL); passo = np.clip(passo, passo_ant - MAX_ACC, passo_ant + MAX_ACC)
            alvo = alvo + passo; passo_ant = passo; filtradas.append(alvo.copy())
        filtradas = np.array(filtradas)
        trace.append(dict(bloco=b, enviado=w0, latencia_s=round(r["lat_s"], 4), estado=est.tolist(), acoes=filtradas.tolist()))
        log(f"bloco {b:02d} lat {r['lat_s']:.2f}s | maior salto vs estado {np.abs(filtradas[0] - est_robo).max():.3f} rad | {'ENVIA' if a.enviar else 'dry-run'}")
        for k in range(a.executa):
            t = time.perf_counter()
            if a.enviar: robo.send_action({n: float(v) for n, v in zip(nomes, filtradas[k])})
            time.sleep(max(0, 1 / a.fps - (time.perf_counter() - t)))
    json.dump(dict(instrucao="(vem do servidor)", cintura=a.cintura, punho=a.punho, enviar=a.enviar, executa=a.executa, trace=trace), open(f"{a.out}/trace.json", "w"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--robot-ip", default="192.168.123.164"); ap.add_argument("--porta", type=int, default=8781)
    ap.add_argument("--blocos", type=int, default=30); ap.add_argument("--executa", type=int, default=15); ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--cintura", action="store_true", help="checkpoint treinado com a cintura (31 valores); ver limite no topo")
    ap.add_argument("--punho", choices=("cabeca", "preto"), default="cabeca"); ap.add_argument("--bgr", action="store_true", help="a câmera do robô entrega BGR")
    ap.add_argument("--enviar", action="store_true", help="comanda o robô de verdade (sem isso é dry-run)")
    a = ap.parse_args()
    from lerobot.robots.unitree_g1.unitree_g1_dex3 import UnitreeG1Dex3, UnitreeG1Dex3Config
    robo = UnitreeG1Dex3(UnitreeG1Dex3Config(robot_ip=a.robot_ip, use_loco=False)); robo.connect()
    s = socket.create_connection(("127.0.0.1", a.porta))
    try: rodar(robo, s, a, nomes_robo())
    except KeyboardInterrupt: print("interrompido, parando de enviar", file=sys.stderr)
    finally: robo.disconnect()


if __name__ == "__main__":
    main()
