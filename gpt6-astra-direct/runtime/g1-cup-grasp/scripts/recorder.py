"""Gravador: renderiza cameras do MuJoCo e envia os quadros ao FFmpeg por pipe (sem arquivos temporarios)."""
import os, subprocess, json, time
import numpy as np, mujoco, cv2


class Recorder:
    # layout 3x2 pedido pelo Luiz: [punho esq][rosto][punho dir] / [esquerda inteira][centro executando][direita inteira]
    GRID = ("left_wrist_camera", "head_camera", "right_wrist_camera", "view_left", "view_center", "side_view")

    BAND = 44   # faixa de titulo acima das cameras (identificacao do video + fase), sem cobrir imagem nenhuma

    def __init__(self, model, path, cameras=GRID, size=(640, 360), fps=30, captions=True, cols=3, title=None):
        if os.path.exists(path):
            raise FileExistsError(f"Recording already exists: {path}")
        self.m = model; self.cams = list(cameras); self.w, self.h = size; self.fps = fps; self.path = path
        self.r = mujoco.Renderer(model, self.h, self.w)
        n = len(self.cams); self.cols = min(cols, n); self.rows = -(-n // self.cols)
        self.W = self.w * self.cols; self.H = self.h * self.rows; self.title = title; self.band = self.BAND if (title or captions) else 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-n", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.W}x{self.H + self.band}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", path],
            stdin=subprocess.PIPE)
        self.frames = 0; self.timeline = []; self.captions = captions; self.caption = ""
        self.last = {}

    def event(self, text, **kw):
        self.caption = text; self.timeline.append(dict(t=round(self.frames / self.fps, 3), text=text, **kw))

    def frame(self, data):
        tiles = []
        for c in self.cams:
            self.r.update_scene(data, camera=c); img = self.r.render().copy(); self.last[c] = img
            cv2.putText(img, c, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, c, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        while len(tiles) % self.cols:
            tiles.append(np.zeros_like(tiles[0]))
        img = np.concatenate([np.concatenate(tiles[i:i + self.cols], axis=1) for i in range(0, len(tiles), self.cols)], axis=0)
        if self.band:
            band = np.full((self.band, self.W, 3), 24, np.uint8)
            if self.title: cv2.putText(band, self.title, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            if self.captions: cv2.putText(band, f"{self.frames / self.fps:5.1f}s  {self.caption}", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 230, 255), 1, cv2.LINE_AA)
            img = np.concatenate([band, img], axis=0)
        self.proc.stdin.write(np.ascontiguousarray(img).tobytes()); self.frames += 1

    def snapshot(self, cam, path):
        cv2.imwrite(path, cv2.cvtColor(self.last[cam], cv2.COLOR_RGB2BGR))

    def close(self, timeline_path=None):
        self.proc.stdin.close()
        code = self.proc.wait()
        self.r.close()
        if timeline_path:
            with open(timeline_path, "x") as handle:
                json.dump(dict(fps=self.fps, frames=self.frames, cameras=self.cams, events=self.timeline), handle, indent=2)
        if code:
            raise RuntimeError(f"FFmpeg failed with status {code}; keep partial recording at {self.path}")
        return self.frames / self.fps
