"""Avaliador congelado, so imagem: le camera-initial.png, camera-final.png, a calibracao pelos
marcadores e o parametro lift_height da tentativa. Nao le estado do simulador.

Versao 6: a posicao final prevista usa a direcao da subida (parametro lift_direction, padrao vertical): a partir da tentativa 30
a subida e diagonal (para cima e para a frente) e a v5, que previa subida vertical, dava concordancia 0 e copo visivel 0 numa pega correta.
Versao 2 (15/09/2026, noite): acrescenta o termo "copo em pe". A v1 nao via o copo tombado.
Versao 5: "subida" passa a ser a projecao do deslocamento observado do centroide sobre o deslocamento previsto
(em imagem, trazer o copo para o corpo pode move-lo para BAIXO no quadro, e a v4 dava zero). O termo "copo em pe"
SAI da formula: com a mao envolvendo o copo (pega de forca) nem a razao da caixa envolvente nem a boca visivel
funcionam numa camera unica (testado: a 16, em pe a 2 graus pela fisica, dava 0 nos dois). A inclinacao do copo e
criterio fisico no physics-report.json, declarado como privilegiado. A formula volta a
100 x (0,50 x subida + 0,30 x concordancia + 0,20 x copo visivel); o campo upright_proxy fica so informativo.
Versao 4: a posicao final prevista inclui o recuo em direcao ao corpo (parametro retreat_distance), quando existe.
Versao 3: os componentes brancos a menos de 80 px do pixel previsto sao unidos antes de medir (um dedo na frente
do copo dividia a silhueta em dois e a v2 lia um copo em pe como deitado). O termo "copo visivel" passa a valer 1
quando existe pelo menos um componente nessa vizinhanca e nenhum fora dela.
Pontuacao (0 a 100) = 100 x (0,40 x subida + 0,20 x concordancia + 0,25 x copo em pe + 0,15 x copo visivel), onde
- subida: deslocamento vertical do centroide branco na imagem dividido pelo deslocamento previsto
  para a altura de levantamento pedida (projetado com a calibracao), limitado a [0, 1];
- concordancia: 1 - |pixel observado - pixel previsto| / 40 px, limitado a [0, 1];
- copo em pe: 1 - |razao largura/altura observada - razao prevista| / 0,35, limitado a [0, 1], onde a razao
  prevista e a da caixa envolvente do mesh do copo EM PE projetado na posicao final prevista (copo deitado muda a razao);
- copo visivel: 1 se ha exatamente um componente branco valido na imagem final, senao 0.
Tambem informa se a base do copo saiu da linha da mesa (bbox final acima da linha do ponto de apoio).

Uso: .venv/bin/python scripts/evaluate_grasp.py results/attempt-01 [--output results/attempt-01/evaluation.json]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
from vision import locate_cup, white_components, white_mask

ROOT = Path(__file__).resolve().parent.parent
TOLERANCE_PX = 40.0
NEIGHBORHOOD_PX = 80.0


def load_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def project(point, cal):
    return cv2.projectPoints(np.asarray(point, float).reshape(1, 3), np.array(cal["rvec"]), np.array(cal["tvec"]),
                             np.array(cal["intrinsics"]), None)[0].ravel()


def predicted_bbox_ratio(vertices, position, cal):
    pts = cv2.projectPoints(np.asarray(vertices) + position, np.array(cal["rvec"]), np.array(cal["tvec"]),
                            np.array(cal["intrinsics"]), None)[0].reshape(-1, 2)
    w, h = pts.max(axis=0) - pts.min(axis=0)
    return float(w / max(h, 1e-6))


def interior_hole_area(rgb, bbox):
    """Area (px) do maior buraco nao branco cercado por branco dentro da caixa: a boca do copo vista de cima."""
    mask = white_mask(rgb)
    x, y, w, h = [int(v) for v in bbox]
    sub = mask[max(0, y - 4):y + h + 4, max(0, x - 4):x + w + 4].copy()
    filled = sub.copy(); hh, ww = filled.shape
    ff = np.zeros((hh + 2, ww + 2), np.uint8)
    cv2.floodFill(filled, ff, (0, 0), 1)
    holes = ((filled == 0) & (sub == 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes)
    return float(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0.0


def predicted_mouth_area(position, cal, r_inner=0.037, h=0.092):
    pts = np.array([[r_inner * np.cos(a), r_inner * np.sin(a), h] for a in np.linspace(0, 2 * np.pi, 72, endpoint=False)]) + position
    px = cv2.projectPoints(pts, np.array(cal["rvec"]), np.array(cal["tvec"]), np.array(cal["intrinsics"]), None)[0].reshape(-1, 2)
    return float(cv2.contourArea(px.astype(np.float32)))


def evaluate(run_dir):
    run_dir = Path(run_dir)
    board = json.loads((ROOT / "scene" / "markers.json").read_text())
    vertices = np.load(ROOT / "scene" / "cup_vertices.npy")
    cal = json.loads((run_dir / "camera-localization.json").read_text())
    params = json.loads((run_dir / "parameters.json").read_text())
    lift_height = float(params.get("lift_height", 0.16))
    retreat = float(params.get("retreat_distance", 0.0) or 0.0)
    ldir = np.asarray(params.get("lift_direction") or [0.0, 0.0, 1.0], float); ldir = ldir / np.linalg.norm(ldir)
    displacement = lift_height * ldir + np.array([-retreat, 0.0, 0.0])
    rgb0, rgb1 = load_rgb(run_dir / "camera-initial.png"), load_rgb(run_dir / "camera-final.png")
    first = locate_cup(rgb0, cal, board, vertices)
    ground = np.array(first["position"], float)
    centroid0 = ground + [0, 0, board["cup_centroid_height_m"]]
    pixel0 = np.array(first["pixel"], float)
    pixel_pred = project(centroid0 + displacement, cal)
    table_row = float(project(ground, cal)[1])
    comps = white_components(rgb1)
    near = [c for c in comps if np.linalg.norm(np.array(c["pixel"]) - pixel_pred) < NEIGHBORHOOD_PX]
    far = [c for c in comps if c not in near]
    visible = 1.0 if near and not far else 0.0
    ratio_pred = predicted_bbox_ratio(vertices, ground + displacement, cal)
    finals = near
    if finals:
        area = np.array([c["area_px"] for c in finals]); pix = np.array([c["pixel"] for c in finals])
        x0 = min(c["bbox"][0] for c in finals); y0 = min(c["bbox"][1] for c in finals)
        x1 = max(c["bbox"][0] + c["bbox"][2] for c in finals); y1 = max(c["bbox"][1] + c["bbox"][3] for c in finals)
        final = {"pixel": (area[:, None] * pix).sum(0) / area.sum(), "bbox": [x0, y0, x1 - x0, y1 - y0], "components": len(finals)}
        pixel1 = np.array(final["pixel"], float)
        pred_vec = pixel_pred - pixel0; obs_vec = pixel1 - pixel0
        rise = float(np.clip(np.dot(obs_vec, pred_vec) / max(np.dot(pred_vec, pred_vec), 1.0), 0, 1))
        agreement = float(np.clip(1 - np.linalg.norm(pixel1 - pixel_pred) / TOLERANCE_PX, 0, 1))
        bottom_row = final["bbox"][1] + final["bbox"][3]
        left_table = bool(bottom_row < table_row - 8)
        ratio_obs = float(final["bbox"][2] / max(final["bbox"][3], 1))
        hole = interior_hole_area(rgb1, final["bbox"]); mouth = predicted_mouth_area(ground + displacement, cal)
        upright = float(np.clip(hole / max(mouth, 1.0), 0, 1))
    else:
        pixel1, rise, agreement, left_table, ratio_obs, upright, hole, mouth = None, 0.0, 0.0, False, None, 0.0, 0.0, 0.0
    score = round(100 * (0.5 * rise + 0.3 * agreement + 0.2 * visible), 2)
    return {"score": score, "score_formula": "100*(0.50*rise_ratio + 0.30*pixel_agreement + 0.20*cup_visible)",
            "evaluator_version": 6, "lift_direction": ldir.tolist(), "retreat_distance_m": retreat, "upright_proxy_not_scored": upright, "interior_hole_px": hole, "predicted_mouth_px": mouth, "components_merged": final["components"] if finals else 0, "bbox_ratio_observed": ratio_obs, "bbox_ratio_predicted_upright": ratio_pred,
            "observation": "RGB da head_camera apenas; calibracao pelos marcadores; sem estado do simulador",
            "rise_ratio": rise, "pixel_agreement": agreement, "cup_visible": visible, "cup_left_table_line": left_table,
            "initial_pixel": pixel0.tolist(), "predicted_final_pixel": pixel_pred.tolist(),
            "observed_final_pixel": None if pixel1 is None else pixel1.tolist(), "table_line_row_px": table_row,
            "final_candidates": len(comps), "far_candidates": len(far), "lift_height_requested_m": lift_height, "tolerance_px": TOLERANCE_PX,
            "initial_estimate": first["position"], "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir"); ap.add_argument("--output")
    a = ap.parse_args()
    out = evaluate(a.run_dir)
    path = Path(a.output) if a.output else Path(a.run_dir) / "evaluation.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
