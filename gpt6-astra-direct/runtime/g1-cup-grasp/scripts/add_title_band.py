"""Acrescenta a faixa de titulo (44 px, acima das cameras, sem cobrir imagem) a um video ja gravado. Sem drawtext do ffmpeg
(nao compilado nesta maquina): quadros passam pelo OpenCV. Uso: add_title_band.py entrada.mp4 saida.mp4 "titulo" """
import sys, subprocess, json, numpy as np, cv2
src, dst, title = sys.argv[1], sys.argv[2], sys.argv[3]
info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", src], capture_output=True, text=True).stdout)["streams"][0]
W, H = int(info["width"]), int(info["height"]); num, den = info["r_frame_rate"].split("/"); fps = int(num) / int(den); B = 44
rd = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
wr = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H + B}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", dst], stdin=subprocess.PIPE)
band = np.full((B, W, 3), 24, np.uint8); cv2.putText(band, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA); n = 0
while True:
    buf = rd.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    wr.stdin.write(np.concatenate([band, np.frombuffer(buf, np.uint8).reshape(H, W, 3)], axis=0).tobytes()); n += 1
wr.stdin.close(); wr.wait(); rd.wait(); print(dst, n, "quadros")
