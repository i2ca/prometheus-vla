# Remendos em repositórios de fora

Mudanças locais que o código de `pontes/groot` e o treino da WLA precisam, guardadas
como patch para não depender do estado da máquina. Aplicar com `git apply <patch>` na
raiz de cada repositório, no commit indicado.

| patch | repositório | commit | o que faz |
|---|---|---|---|
| `isaac-gr00t-n16.patch` | github.com/NVIDIA/Isaac-GR00T (`~/DEV/Isaac-GR00T`) | 9b37aa1 | `GR00T_ATTN=sdpa` no backbone Eagle (sem flash_attn no aarch64); `GR00T_SIM_JANELA=1` abre a janela do MuJoCo no rollout |
| `gr00trobosuite.patch` | github.com/xieleo5/robosuite (em `external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/dexmg/gr00trobosuite`) | b6aa4a5 | só destrói o viewer no reset "hard": a janela não fecha e reabre a cada episódio |
| `gr00t-wholebodycontrol-sonic.patch` | github.com/NVlabs/GR00T-WholeBodyControl (`~/DEV/GR00T-WholeBodyControl`) | b042411 | `SONIC_CENA=<xml>` troca a cena do simulador; cudla opcional (só existe no Jetson) |
| `unifolm-wla.patch` | submódulo `unifolm-wla` | f33d0e7 | balanceamento de fontes no dataloader, MMDiT e ajustes do treino da WLA |

O N1.7 (`~/DEV/Isaac-GR00T-n17`, commit 51d4c89) roda sem remendo: o qwen3_backbone já cai para sdpa sozinho.
