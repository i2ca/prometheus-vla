"""RGB-only calibration and white-cup localization using known tabletop markers.

No MuJoCo import or object state. Camera intrinsics and cup dimensions are
explicit prior information. Sixteen marker corners provide redundant checks;
four fitted center points alone cannot validate their own homography.
"""
import itertools
import cv2
import numpy as np


def project_h(points, matrix):
    return cv2.perspectiveTransform(np.asarray(points, np.float64).reshape(-1, 1, 2), matrix).reshape(-1, 2)


def calibrate(rgb, board):
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)
    common = (sat > 120) & (val > 100)
    masks = {"red": ((hue < 10) | (hue > 173)), "cyan": (hue > 75) & (hue < 100),
             "blue": (hue >= 100) & (hue < 135), "magenta": (hue > 135) & (hue < 173)}
    centers, quadrilaterals, worlds, used, missing = [], [], [], [], []
    names = list(board["markers"])
    for name in names:
        contours, _ = cv2.findContours((masks[name] & common).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 30]
        if len(contours) != 1:
            missing.append(name); continue          # marcador coberto (pelo copo, pela mao): segue com os outros
        contour = contours[0]
        quad = cv2.approxPolyDP(contour, 0.035 * cv2.arcLength(contour, True), True).reshape(-1, 2)
        if len(quad) != 4 or np.any(quad < 2) or np.any(quad[:, 0] >= w - 2) or np.any(quad[:, 1] >= h - 2):
            missing.append(name); continue
        quadrilaterals.append(quad.astype(float))
        centers.append(quad.mean(axis=0))
        worlds.append(board["markers"][name]); used.append(name)
    if len(used) < 3:
        raise ValueError(f"Need at least 3 visible markers, found {len(used)} (missing {missing})")
    if len(used) >= 4:
        rough, _ = cv2.findHomography(np.asarray(worlds), np.asarray(centers))
    else:   # 3 marcadores: afinidade como predicao grosseira para ordenar os cantos
        A = cv2.getAffineTransform(np.asarray(worlds, np.float32), np.asarray(centers, np.float32))
        rough = np.vstack([A, [0, 0, 1]])
    object_xy, pixels = [], []
    half = board["half_size"]
    for center, quad in zip(worlds, quadrilaterals):
        corners = np.asarray(center) + np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
        predicted = project_h(corners, rough)
        ordered = min(itertools.permutations(quad), key=lambda qs: np.linalg.norm(np.asarray(qs) - predicted))
        object_xy.extend(corners)
        pixels.extend(ordered)
    object_xy, pixels = np.asarray(object_xy), np.asarray(pixels)
    homography, _ = cv2.findHomography(object_xy, pixels)
    rms = float(np.sqrt(np.mean(np.sum((project_h(object_xy, homography) - pixels)**2, axis=1))))
    focal = h / (2 * np.tan(np.deg2rad(board["camera_fovy_deg"]) / 2))
    intrinsics = np.array([[focal, 0, (w-1)/2], [0, focal, (h-1)/2], [0, 0, 1]], float)

    def heldout_errors(object_xy, pixels, n_m):
        """Deixa um marcador de fora, resolve a pose da camera (PnP, intrinsecos conhecidos) com os outros e
        reprojeta o que ficou de fora. Homografia livre de 3 quadrados pequenos extrapola mal; PnP nao."""
        per = []
        for index in range(n_m):
            keep = np.ones(4 * n_m, bool); keep[4*index:4*index+4] = False
            wp = np.c_[object_xy[keep], np.full(keep.sum(), board["marker_z"])]
            ok_, rv, tv = cv2.solvePnP(wp, pixels[keep], intrinsics, None, flags=cv2.SOLVEPNP_ITERATIVE)
            wo = np.c_[object_xy[~keep], np.full((~keep).sum(), board["marker_z"])]
            pr = cv2.projectPoints(wo, rv, tv, intrinsics, None)[0].reshape(-1, 2)
            per.append(np.linalg.norm(pr - pixels[~keep], axis=1))
        return per
    n_m = len(used)
    per = heldout_errors(object_xy, pixels, n_m)
    worst = int(np.argmax([e.max() for e in per]))
    if n_m >= 4 and per[worst].max() > 6:
        # um marcador parcialmente coberto (mao, copo) distorce o quadrilatero: descarta e refaz com os outros
        missing.append(used[worst] + " (cortado)"); del used[worst]
        keep = np.ones(4 * n_m, bool); keep[4*worst:4*worst+4] = False
        object_xy, pixels = object_xy[keep], pixels[keep]; n_m -= 1
        homography, _ = cv2.findHomography(object_xy, pixels)
        rms = float(np.sqrt(np.mean(np.sum((project_h(object_xy, homography) - pixels)**2, axis=1))))
        per = heldout_errors(object_xy, pixels, n_m)
    heldout = np.concatenate(per).tolist()
    world_points = np.c_[object_xy, np.full(4 * n_m, board["marker_z"])]
    ok, rvec, tvec = cv2.solvePnP(world_points, pixels, intrinsics, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or rms > 2 or max(heldout) > (6 if n_m >= 4 else 12):   # com 3 marcadores a validacao deixando um de fora e mais frouxa
        raise ValueError(f"Poor camera calibration: fit {rms:.2f}px, held-out max {max(heldout):.2f}px")
    return {"method": f"{4*n_m} colored marker corners ({n_m} markers, missing {missing}); leave-one-marker-out validation",
            "markers_used": used, "markers_missing": missing,
            "world_to_image": homography.tolist(), "intrinsics": intrinsics.tolist(),
            "rvec": rvec.ravel().tolist(), "tvec": tvec.ravel().tolist(),
            "fit_rms_px": rms, "heldout_rms_px": float(np.sqrt(np.mean(np.square(heldout)))),
            "heldout_max_px": max(heldout), "marker_pixels": pixels.tolist(),
            "image_size": [w, h]}


def white_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 1] < 45) & (hsv[:, :, 2] > WHITE_V_MIN)).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))



WHITE_V_MIN = 150   # v3 do avaliador usava 175; paredes facetadas do copo ficam entre 150 e 200


def white_components(rgb):
    mask = white_mask(rgb)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    h, w = mask.shape
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 80 or area > w*h*.10 or x < 2 or y < 2 or x+bw >= w-2 or y+bh >= h-2:
            continue
        if not .35 < bw/max(bh, 1) < 3:
            continue
        moment = cv2.moments(contour)
        candidates.append({"pixel": [moment["m10"]/moment["m00"], moment["m01"]/moment["m00"]],
                           "area_px": float(area), "bbox": [x, y, bw, bh]})
    return sorted(candidates, key=lambda c: -c["area_px"])


def split_body_and_handle(rgb, candidate, handle_px=15):
    """Abertura morfologica com nucleo maior que a largura da alca: o que sobra e o corpo (cilindro);
    o residuo dentro da caixa do candidato e a alca. So RGB."""
    mask = white_mask(rgb)
    x, y, w, h = candidate["bbox"]
    roi = np.zeros_like(mask); roi[y:y+h, x:x+w] = mask[y:y+h, x:x+w]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (handle_px, handle_px))
    body = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
    handle = (roi > 0) & (body == 0)
    handle = cv2.morphologyEx(handle.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    def centroid(mk):
        mm = cv2.moments(mk.astype(np.uint8)); return None if mm["m00"] < 1 else [mm["m10"] / mm["m00"], mm["m01"] / mm["m00"]]
    bx, by, bw, bh = cv2.boundingRect(body) if body.any() else (x, y, w, h)
    # o residuo da abertura nao e so a alca: a borda fina da boca do copo, vista de cima, tambem e
    # estreita e sobra. Ficar com o centroide de tudo punha a "alca" 60 a 92 px ACIMA do corpo em
    # 15 de 15 casos, o que dava erro de yaw de 35 a 106 graus. A alca de uma caneca em pe sobressai
    # de lado, entao entre os componentes conexos do residuo vale o que mais se afasta em x.
    n_cc, lab, stats, cents = cv2.connectedComponentsWithStats(handle.astype(np.uint8), 8)
    best, best_dx = None, -1.0
    cbx = bx + 0.5 * bw
    for i in range(1, n_cc):
        if stats[i, cv2.CC_STAT_AREA] < 20:
            continue
        dx = abs(float(cents[i][0]) - cbx)
        if dx > best_dx:
            best, best_dx = i, dx
    if best is not None:
        handle = (lab == best).astype(np.uint8)
    return {"body_bbox": [bx, by, bw, bh], "body_pixel": centroid(body), "body_area_px": float(body.sum()),
            "handle_pixel": centroid(handle), "handle_area_px": float(handle.sum())}


def locate_cup(rgb, calibration, board, vertices=None, handle_vertices=None):
    """Entre os componentes brancos sobre a mesa, ajusta a silhueta do copo conhecido a cada um e fica com o melhor ajuste
    relativo (a cena do cafe tem coador, chaleira e pote tambem claros; antes o maior componente era tomado como o copo)."""
    candidates = white_components(rgb)
    if not candidates:
        raise ValueError("No white cup candidate")
    results, errors = [], []
    for c in sorted(candidates, key=lambda c: -c["area_px"]):
        try:
            r = _locate_candidate(rgb, calibration, board, c, vertices); r["candidate_count"] = len(candidates); results.append(r)
        except ValueError as e:
            errors.append(str(e))
    if not results:
        raise ValueError("No candidate matches the upright cup: " + " | ".join(errors[:4]))
    best = min(results, key=lambda r: (r["silhouette_fit_rel"] if r["silhouette_fit_rel"] is not None else 1.0))
    best["candidates_rejected"] = len(results) - 1 + len(errors)
    return best


def _locate_candidate(rgb, calibration, board, candidate, vertices):
    candidates = [candidate]
    cup = dict(candidates[0])
    parts = split_body_and_handle(rgb, cup)
    # A thin ring has cup-like bounds but no solid cylinder body. Do not
    # turn such a distractor into a grasp target via the centroid fallback.
    if parts["body_pixel"] is None or parts["body_area_px"] <= 100:
        raise ValueError("White candidate has no resolved cup body")
    if parts["body_pixel"] is not None and parts["body_area_px"] > 100:
        cup["pixel"], cup["bbox"] = parts["body_pixel"], parts["body_bbox"]   # o corpo, sem a alca, define a posicao
    cup["parts"] = parts
    rotation = cv2.Rodrigues(np.array(calibration["rvec"]))[0]
    translation = np.array(calibration["tvec"])
    origin = -rotation.T @ translation
    ray = rotation.T @ np.linalg.solve(np.array(calibration["intrinsics"]), np.r_[cup["pixel"], 1.0])
    plane_z = board["table_z"] + board["cup_centroid_height_m"]
    world = origin + (plane_z - origin[2]) / ray[2] * ray
    fit_rms = None; method_note = None; rel_out = None
    if vertices is not None:
        # Fit the known upright mesh silhouette bounds. The mug has a handle and
        # its body origin is not the cylinder center; a raw centroid is biased.
        x, y, width, height = cup["bbox"]
        observed = np.array([x, y, x+width-1, y+height-1], float)
        world[2] = board["table_z"]

        def bounds(position):
            points = cv2.projectPoints(np.asarray(vertices) + position,
                                      np.array(calibration["rvec"]), np.array(calibration["tvec"]),
                                      np.array(calibration["intrinsics"]), None)[0].reshape(-1, 2)
            return np.r_[points.min(axis=0), points.max(axis=0)]

        for _ in range(12):
            predicted = bounds(world)
            jac = np.column_stack([(bounds(world + np.eye(3)[i]*0.0001) - predicted)/0.0001 for i in range(2)])
            change = np.linalg.lstsq(jac, observed - predicted, rcond=None)[0]
            world[:2] += np.clip(change, -.02, .02)
            if np.linalg.norm(change) < 1e-6:
                break
        fit_rms = float(np.sqrt(np.mean((bounds(world) - observed)**2)))
        rel = fit_rms / max(float(height), 1.0); rel_out = rel   # relativo a altura do copo em pixels: nao depende do campo de visao
        if rel > 0.10:
            raise ValueError(f"Cup silhouette does not match the upright known object: {fit_rms:.2f}px ({rel*100:.0f}% da altura)")
        if rel > 0.04:
            # ajuste ruim (oclusao parcial, posicao obliqua): recua para o raio pelo centroide do corpo
            world = origin + (plane_z - origin[2]) / ray[2] * ray; world[2] = board["table_z"]
            method_note = f"centroid-ray fallback (silhouette fit {fit_rms:.1f}px)"
        else:
            method_note = None
    if not (.22 < world[0] < .68 and -.48 < world[1] < .48):
        raise ValueError("RGB cup estimate is outside the table")
    yaw_deg = None
    if parts["handle_pixel"] is not None and parts["handle_area_px"] > 20:
        # direcao corpo -> alca, retroprojetada no plano da altura media da alca; yaw = angulo dessa direcao no mundo
        def back(px, z):
            ray = rotation.T @ np.linalg.solve(np.array(calibration["intrinsics"]), np.r_[px, 1.0])
            return origin + (z - origin[2]) / ray[2] * ray
        zh = board["table_z"] + 0.5 * board["cup_height_m"]
        try:
            b3, h3 = back(np.asarray(parts["body_pixel"], float), zh), back(np.asarray(parts["handle_pixel"], float), zh)
            yaw_deg = float(np.degrees(np.arctan2(h3[1] - b3[1], h3[0] - b3[0])))
        except (TypeError, ValueError):
            yaw_deg = None          # candidato sem alca reconhecivel (coador, chaleira)
    return {**cup, "position": [float(world[0]), float(world[1]), board["table_z"]],
            "method": (method_note or "White RGB silhouette bounds fitted to known upright cup mesh") if vertices is not None else "White RGB centroid ray at known cup-centroid height; upright cup prior",
            "silhouette_fit_rms_px": fit_rms, "silhouette_fit_rel": rel_out, "yaw_deg_estimate": yaw_deg, "parts": parts,
            "known_centroid_height_m": board["cup_centroid_height_m"]}


def locate_colored(rgb, calibration, board, color, height_m, region=None, min_area_px=300):
    """Localiza um objeto de cor conhecida so por RGB: mascara HSV, componentes conectados, os que caem na mesa (raio pela
    calibracao a altura height_m do centroide), opcionalmente dentro de 'region' [(xmin,xmax),(ymin,ymax)] no mundo; fica com o
    maior. Cores: 'brown' (pote de ceramica: matiz 0-10/170-180, saturado, escuro), 'black' (scoop: V < 35).
    Devolve posicao no mundo (x, y, table_z) e o pixel. Percepcao so por camera, sem estado do simulador."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); h, s, v = cv2.split(hsv)
    if color == "brown":
        mask = (((h <= 10) | (h >= 170)) & (s > 120) & (v > 15) & (v < 130))
    elif color == "black":
        mask = (v < 35)
    else:
        raise ValueError(color)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    rot = cv2.Rodrigues(np.array(calibration["rvec"]))[0]; tr = np.array(calibration["tvec"]); origin = -rot.T @ tr
    K = np.array(calibration["intrinsics"]); found = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area_px: continue
        ray = rot.T @ np.linalg.solve(K, np.r_[centroids[i], 1.0]); w = origin + (board["table_z"] + height_m - origin[2]) / ray[2] * ray
        if not (.22 < w[0] < .68 and -.48 < w[1] < .48): continue
        if region and not (region[0][0] <= w[0] <= region[0][1] and region[1][0] <= w[1] <= region[1][1]): continue
        found.append({"position": [float(w[0]), float(w[1]), board["table_z"]], "pixel": centroids[i].tolist(), "area_px": int(stats[i, cv2.CC_STAT_AREA]), "bbox": stats[i, :4].tolist()})
    if not found:
        raise ValueError(f"No {color} object on the table")
    best = max(found, key=lambda c: c["area_px"]); best["candidates"] = len(found); best["color"] = color; return best


def fit_silhouette(bbox, vertices, calibration, board, world_xy0, iterations=12):
    """Ajusta a posicao (x, y) de um objeto de geometria conhecida (vertices em metros, base em z=0, apoiado na mesa) para que
    a caixa envolvente da sua projecao bata com a caixa observada. Devolve (x, y, rms_px). Mesma ideia do ajuste do copo."""
    x, y, w, h = bbox; observed = np.array([x, y, x + w - 1, y + h - 1], float); world = np.array([world_xy0[0], world_xy0[1], board["table_z"]], float)
    def bounds(pos):
        pts = cv2.projectPoints(np.asarray(vertices) + pos, np.array(calibration["rvec"]), np.array(calibration["tvec"]), np.array(calibration["intrinsics"]), None)[0].reshape(-1, 2)
        return np.r_[pts.min(axis=0), pts.max(axis=0)]
    for _ in range(iterations):
        pred = bounds(world); jac = np.column_stack([(bounds(world + np.eye(3)[i] * 1e-4) - pred) / 1e-4 for i in range(2)])
        step = np.linalg.lstsq(jac, observed - pred, rcond=None)[0]; world[:2] += np.clip(step, -.02, .02)
        if np.linalg.norm(step) < 1e-6: break
    return float(world[0]), float(world[1]), float(np.sqrt(np.mean((bounds(world) - observed) ** 2)))


def cylinder_vertices(radius, height, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.array([[radius * np.cos(t), radius * np.sin(t), z] for z in (0.0, height) for t in a])
