"""Injeção de profundidade 3D (PointNet) na política PI05 — **sem tato**.

Variante do `pi05_d_injector` para a ablação "pi05-depth": só adiciona o token de
PointNet ao prefixo, sem o token de pressão do Dex3. Útil para medir, no mesmo
dataset, o ganho isolado de depth (pi05-depth vs pi05 vanilla) e o ganho marginal
de pressão (pi05-D vs pi05-depth).

A arquitetura e o ponto de injeção são os mesmos do `pi05_d_injector`; só deixa
de criar `pressure_proj` e de anexar o token de pressão.
"""

import logging
import types
from pathlib import Path

import safetensors.torch as st
import torch

from train.depth_encoder import PointNetEncoder, depth_to_pointcloud


DEPTH_KEY = "observation.images.head_camera_depth"
LEFT_PRESSURE_KEY = "observation.left_hand_pressure"
RIGHT_PRESSURE_KEY = "observation.right_hand_pressure"


def _vlm_hidden_size(policy) -> int:
    embed = policy.model.paligemma_with_expert.paligemma.language_model.embed_tokens
    return embed.weight.shape[1]


def _load_injected_weights(policy, checkpoint_dir):
    sd_path = Path(checkpoint_dir) / "model.safetensors"
    if not sd_path.exists():
        return 0
    sd = st.load_file(str(sd_path))
    injected = {k: v for k, v in sd.items() if "pointnet" in k}
    if not injected:
        return 0
    policy.load_state_dict(injected, strict=False)
    return len(injected)


def inject_pi05_depth(policy, device, camera_intrinsics=None, load_injected_from=None):
    """Acopla PointNet ao PI05 e injeta um único token extra (depth) no prefixo."""
    print("\n[INJECAO PI05-DEPTH]: Ativando fusao 3D (PointNet) sem tato para PI05...")

    hidden_size = _vlm_hidden_size(policy)

    policy.pointnet = PointNetEncoder(output_dim=hidden_size).to(device)
    policy.camera_intrinsics = camera_intrinsics or {
        "fx": 600.0,
        "fy": 600.0,
        "cx": 320.0,
        "cy": 240.0,
    }

    model = policy.model
    model._extra_prefix_embs = None

    original_embed_prefix = model.embed_prefix

    def patched_embed_prefix(self, images, img_masks, tokens, masks):
        embs, pad_masks, att_masks = original_embed_prefix(images, img_masks, tokens, masks)

        extra = getattr(self, "_extra_prefix_embs", None)
        if extra is None:
            return embs, pad_masks, att_masks

        extra_embs, extra_pad_masks = extra
        extra_embs = extra_embs.to(dtype=embs.dtype, device=embs.device)
        extra_pad_masks = extra_pad_masks.to(device=pad_masks.device)

        bsize, k, _ = extra_embs.shape
        embs = torch.cat([embs, extra_embs], dim=1)
        pad_masks = torch.cat([pad_masks, extra_pad_masks], dim=1)
        extra_att = torch.zeros(bsize, k, dtype=att_masks.dtype, device=att_masks.device)
        att_masks = torch.cat([att_masks, extra_att], dim=1)

        return embs, pad_masks, att_masks

    model.embed_prefix = types.MethodType(patched_embed_prefix, model)

    original_forward = policy.forward
    original_predict = policy.predict_action_chunk

    def _compute_extra_tokens(self, batch):
        depth = batch.pop(DEPTH_KEY, None)
        # Drop pressure keys silently if present; they are not used here but a
        # co-loaded dataset may still carry them.
        left = batch.pop(LEFT_PRESSURE_KEY, None)
        right = batch.pop(RIGHT_PRESSURE_KEY, None)

        tokens = []
        if depth is not None:
            pc = depth_to_pointcloud(depth.float(), self.camera_intrinsics)
            depth_tok = self.pointnet(pc).unsqueeze(1)
            tokens.append(depth_tok)

        saved = (depth, left, right)
        if not tokens:
            return None, saved

        extra_embs = torch.cat(tokens, dim=1)
        bsize, k, _ = extra_embs.shape
        extra_pad = torch.ones(bsize, k, dtype=torch.bool, device=extra_embs.device)
        return (extra_embs, extra_pad), saved

    def _restore_batch(batch, saved):
        depth, left, right = saved
        if depth is not None:
            batch[DEPTH_KEY] = depth
        if left is not None:
            batch[LEFT_PRESSURE_KEY] = left
        if right is not None:
            batch[RIGHT_PRESSURE_KEY] = right

    def _run_with_extras(self, runner, batch, *args, **kwargs):
        extras, saved = _compute_extra_tokens(self, batch)
        removed_feat = self.config.input_features.pop(DEPTH_KEY, None)
        try:
            self.model._extra_prefix_embs = extras
            out = runner(batch, *args, **kwargs)
        finally:
            self.model._extra_prefix_embs = None
            if removed_feat is not None:
                self.config.input_features[DEPTH_KEY] = removed_feat
            _restore_batch(batch, saved)
        return out

    def patched_forward(self, batch, *args, **kwargs):
        return _run_with_extras(self, original_forward, batch, *args, **kwargs)

    def patched_predict(self, batch, *args, **kwargs):
        return _run_with_extras(self, original_predict, batch, *args, **kwargs)

    policy.forward = types.MethodType(patched_forward, policy)
    policy.predict_action_chunk = types.MethodType(patched_predict, policy)

    if load_injected_from is not None:
        n = _load_injected_weights(policy, load_injected_from)
        if n > 0:
            print(f"[INJECAO PI05-DEPTH]: carregados {n} pesos treinados de {load_injected_from}")

    print("[INJECAO PI05-DEPTH]: Concluida. Profundidade inserida como token de prefixo (sem tato).\n")
