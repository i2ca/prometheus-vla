# PROMETHEUS 17/09/2026 — move o copo e o coador com o simulador RODANDO.
#
# O `sim_main.py` chama `passo(env)` uma vez por volta do laco principal. A funcao le
# `~/cena_cafe.json` e escreve a pose direto na fisica. Sem isto, mudar a posicao de um objeto
# exigia reiniciar o simulador — 1 a 2 minutos por tentativa de arranjo.
#
# Duas politicas diferentes de propósito:
#
#   COADOR  - reaplicado SEMPRE (a cada `INTERVALO` voltas). Ele e cinematico, entao reescrever a
#             pose nao briga com a fisica, e isso o traz de volta depois de um `reset_object_self`,
#             que restaura os objetos para a pose de nascimento.
#   COPO    - so quando o arquivo MUDA. Ele e o objeto que o robo pega: reaplicar a pose todo
#             passo o congelaria no ar e a tarefa ficaria impossivel.
import json
import os
import time

import torch

from tasks.common_scene.pose_cafe import ARQUIVO, pose_mundo

INTERVALO = 30            # voltas do laco entre duas reaplicacoes do coador
_estado = {"mtime": None, "proxima": 0, "mediu_mesa": False, "avisou": False}


def _escreve(env, nome_cena: str, nome_pose: str) -> None:
    """Poe o objeto na pose do arquivo, zerando a velocidade."""
    obj = env.scene[nome_cena]
    pos, quat = pose_mundo(nome_pose)
    origem = env.scene.env_origins[0].tolist()
    linha = [pos[0] + origem[0], pos[1] + origem[1], pos[2] + origem[2], *quat]
    pose = torch.tensor([linha], dtype=torch.float32, device=env.device)
    obj.write_root_pose_to_sim(pose)
    obj.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=env.device))


def _mede_mesa(env) -> None:
    """Imprime a caixa da mesa uma vez, para sabermos a altura do tampo de verdade."""
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
        palco = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        for caminho in ("/World/envs/env_0/PackingTable", "/World/envs/env_0/Room"):
            prim = palco.GetPrimAtPath(caminho)
            if not prim.IsValid():
                continue
            r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lo, hi = r.GetMin(), r.GetMax()
            print(f"[cena_cafe] {caminho}: x {lo[0]:.3f}..{hi[0]:.3f}  y {lo[1]:.3f}..{hi[1]:.3f}"
                  f"  z {lo[2]:.3f}..{hi[2]:.3f}   <-- TAMPO = {hi[2]:.3f}", flush=True)
    except Exception as e:                                             # noqa: BLE001
        print(f"[cena_cafe] nao consegui medir a mesa: {e}", flush=True)


def passo(env) -> None:
    """Uma volta. Sai na hora se a cena nao for a do cafe."""
    try:
        if "object2" not in env.scene.rigid_objects:
            return
    except Exception:                                                  # noqa: BLE001
        return

    if not _estado["mediu_mesa"]:
        _estado["mediu_mesa"] = True
        _mede_mesa(env)

    try:
        mtime = os.path.getmtime(ARQUIVO)
    except OSError:
        mtime = 0.0

    mudou = mtime != _estado["mtime"]
    _estado["mtime"] = mtime

    agora = time.time()
    if not mudou and agora < _estado["proxima"]:
        return
    _estado["proxima"] = agora + INTERVALO / 60.0

    try:
        _escreve(env, "object2", "coador")
        if mudou:
            _escreve(env, "object", "copo")
            try:
                with open(ARQUIVO) as f:
                    print(f"[cena_cafe] pose nova: {json.dumps(json.load(f))}", flush=True)
            except (OSError, ValueError):
                print("[cena_cafe] sem ~/cena_cafe.json; usando a pose padrao", flush=True)
    except Exception as e:                                             # noqa: BLE001
        if not _estado["avisou"]:
            _estado["avisou"] = True
            print(f"[cena_cafe] falhei ao aplicar a pose: {e}", flush=True)
