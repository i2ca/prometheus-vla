"""Roda um script do Psi0 trocando attn_implementation="flash_attention_2" por "sdpa" (o trainer de finetune de jan/2026 fixa
flash_attn, que não está instalado; o VLM fica congelado e os servidores já usam sdpa). Uso: python sdpa.py <script> [args...]"""
import runpy, sys
import transformers

# O collator do checkout de jan/2026 assumia uma imagem por exemplo e fazia squeeze(0) no grid [N,3].
# Com as três câmeras isso deixava [B,N,3], mas o Qwen espera todos os grids concatenados em [B*N,3].
import psi.trainers.qwen3vl_mixin as _qwen
_orig_collate = _qwen.PaddedCollatorForTogether.__call__
def _collate(self, instances):
    out = _orig_collate(self, instances)
    if out["image_grid_thw"].ndim == 3:
        out["image_grid_thw"] = out["image_grid_thw"].flatten(0, 1)
    return out
_qwen.PaddedCollatorForTogether.__call__ = _collate

# Accelerate envolve o val_dataloader e volta a agrupar image_grid_thw como [B,N,3]. Normaliza na entrada do modelo para os dois caminhos.
import psi.models.psi0 as _psi0
_orig_forward = _psi0.Psi0Model.forward
def _forward(self, *a, **k):
    grid = k.get("image_grid_thw")
    if grid is not None and grid.ndim == 3:
        k["image_grid_thw"] = grid.flatten(0, 1)
    return _orig_forward(self, *a, **k)
_psi0.Psi0Model.forward = _forward

_orig = transformers.PreTrainedModel.from_pretrained.__func__


def _from_pretrained(cls, *a, **k):
    if k.get("attn_implementation") == "flash_attention_2": k["attn_implementation"] = "sdpa"
    return _orig(cls, *a, **k)


transformers.PreTrainedModel.from_pretrained = classmethod(_from_pretrained)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
