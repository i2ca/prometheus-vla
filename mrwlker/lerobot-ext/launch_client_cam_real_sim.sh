#!/usr/bin/env bash
# OLHOS REAIS, CORPO VIRTUAL.
#
# Câmeras do G1 de verdade → inferência na athena → ações executadas no MuJoCo
# que roda NESTE notebook. O robô real não recebe um único comando: este script
# só ASSINA os streams de imagem dele (sockets SUB), nunca abre o caminho de
# `lowcmd`.
#
#   uso: bash launch_client_cam_real_sim.sh [args extras do cliente...]
#
# ── Por que este modo existe ────────────────────────────────────────────────
# O teste de 02/09 mostrou que o modelo funciona com as imagens do dataset e
# trava com as do MuJoCo: no simulador quatro juntas do braço congelam nos
# mesmos valores que ele devolve quando recebe ruído puro — a resposta média,
# de quem não reconhece a cena. A cena do MuJoCo está fora da distribuição, e
# nenhuma pose inicial conserta isso.
#
# Aqui a imagem volta a ser a bancada real, que é o que ele viu no treino, e o
# corpo continua sendo o simulador, que é o que se pode quebrar sem custo. É o
# ensaio antes de entregar o braço de verdade ao modelo.
#
# ── O que ainda NÃO fecha, e é preciso saber ao ler o resultado ─────────────
# A PROPRIOCEPÇÃO VEM DO SIMULADOR, não do robô. Então o modelo vê a cena real
# (com o braço real parado onde estiver) e lê juntas do corpo virtual, que se
# movem conforme ele comanda. Enquanto o braço simulado não divergir muito, a
# câmera da cabeça continua mostrando algo coerente; a de PULSO não, porque ela
# está montada no braço real, que não acompanha.
#
# Ler as juntas do robô resolveria — e foi decidido NÃO fazer isso: o objeto de
# controle do G1 (`UnitreeG1Dex3`) abre leitura e escrita no mesmo `connect()`,
# e o motivo deste modo existir é justamente não ter o caminho de comando aberto
# para o robô real. Um leitor só-leitura do `lowstate` é trabalho para depois.
set -euo pipefail

ATHENA="${ATHENA:-10.9.8.252}"
ROBO="${ROBO:-10.9.8.73}"
LOGIN="${LOGIN:-mrwlker}"
ENV="${ENV:-$HOME/miniconda3/envs/prometheus-vla}"
CAM_PORTA=5555
CAM_PORTA_PULSO=5556

cd "$(dirname "$(readlink -f "$0")")"

# ── Limpeza: filhos órfãos da execução anterior ─────────────────────────────
# O cliente com `--sim` sobe MuJoCo, publicador de imagens (5555) e ponte_mao
# (6002/6003). Matar só o pai deixa os filhos segurando as portas e a próxima
# subida morre com "Address already in use" disfarçado de erro de câmera.
for porta in 5555 6000 6001 6002 6003; do
    pid=$(ss -tlnp 2>/dev/null | awk -v p=":$porta " '$0 ~ p {match($0,/pid=[0-9]+/); if (RSTART) print substr($0,RSTART+4,RLENGTH-4)}' | head -1)
    [ -n "${pid:-}" ] && { echo "· matando resto da execução anterior na porta $porta (pid $pid)"; kill -9 "$pid" 2>/dev/null || true; }
done
sleep 1

# ── Como chegar nas câmeras do robô ─────────────────────────────────────────
# Caminho curto: o notebook fala direto com o robô. Caminho longo: só a athena
# fala, e o notebook chega por um túnel SSH através dela. O script escolhe
# sozinho porque a resposta muda com a rede do laboratório — e descobrir isso
# no meio de um teste custa o teste.
#
# Cuidado com o `/dev/tcp`: ele diz que ALGUÉM aceitou a conexão, não que é o
# servidor de imagem. É triagem, e o erro de verdade aparece no cliente.
alcanca() { timeout 3 bash -c "cat < /dev/null > /dev/tcp/$1/$2" 2>/dev/null; }

# E a triagem do túnel NÃO pode ser feita na ponta local. O `ssh -L` abre a porta
# em 127.0.0.1 assim que conecta na athena, independente de o destino existir:
# um `/dev/tcp` local então SEMPRE aceita, o script diz "túnel ok" e o cliente
# fica esperando imagem de um robô desligado, sem erro nenhum. Com o robô fora
# do ar isso já aconteceu aqui. Quem tem que responder é a ATHENA.
alcanca_de_athena() {
    timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=6 "$LOGIN@$ATHENA" \
        "timeout 3 bash -c 'echo > /dev/tcp/$ROBO/$1'" >/dev/null 2>&1
}

TUNEL_PID=""
if alcanca "$ROBO" $CAM_PORTA && alcanca "$ROBO" $CAM_PORTA_PULSO; then
    CAM_HOST="$ROBO"
    echo "== câmeras: direto em $ROBO:$CAM_PORTA/$CAM_PORTA_PULSO =="
else
    echo "== $ROBO não responde daqui — tentando pela athena =="
    if ! alcanca_de_athena $CAM_PORTA || ! alcanca_de_athena $CAM_PORTA_PULSO; then
        echo "❌ a athena também não alcança $ROBO:$CAM_PORTA/$CAM_PORTA_PULSO."
        echo "   O robô está desligado, fora da rede, ou os servidores de câmera"
        echo "   não subiram. NO ROBÔ:"
        echo "     python Scripts_Prometheus_int/full_realsenser_server.py"
        echo "     python Scripts_Prometheus_int/right_arm_realsense_server.py"
        exit 1
    fi
    # Portas locais idênticas às remotas para o log do cliente continuar
    # legível. `-N` não abre shell; `ExitOnForwardFailure` faz o ssh FALHAR se a
    # porta local já estiver ocupada, em vez de subir um túnel que não
    # encaminha nada e deixar o cliente esperando imagem para sempre.
    ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
        -L "127.0.0.1:$CAM_PORTA:$ROBO:$CAM_PORTA" \
        -L "127.0.0.1:$CAM_PORTA_PULSO:$ROBO:$CAM_PORTA_PULSO" \
        "$LOGIN@$ATHENA" &
    TUNEL_PID=$!
    trap '[ -n "$TUNEL_PID" ] && kill "$TUNEL_PID" 2>/dev/null || true' EXIT
    for _ in $(seq 1 20); do
        sleep 0.5
        alcanca 127.0.0.1 $CAM_PORTA && break
    done
    if ! alcanca 127.0.0.1 $CAM_PORTA; then
        echo "❌ o túnel não subiu (o robô responde da athena, então é o ssh)."
        echo "   Confira 'ssh $LOGIN@$ATHENA' e se a porta $CAM_PORTA já não está"
        echo "   ocupada aqui."
        exit 1
    fi
    CAM_HOST=127.0.0.1
    echo "== câmeras: túnel 127.0.0.1:$CAM_PORTA/$CAM_PORTA_PULSO -> $ROBO =="
fi

# O servidor de inferência não passa por túnel: ele escuta em 0.0.0.0 na athena
# e o notebook chega nele direto. Se um dia não chegar, é a mesma receita.
if ! alcanca "$ATHENA" 5600; then
    echo "❌ o servidor de inferência não responde em $ATHENA:5600. Suba com:"
    echo "   ssh $LOGIN@$ATHENA 'screen -dmS infer bash ~/DEV/prometheus-vla/lerobot-ext/athena/launch_server_fastwamd.sh 2 /data/mrwlker/checkpoints/fastwamd_lora_aug_step4250'"
    exit 1
fi

# libstdc++ do conda antes do libzmq; IK monothread; render das câmeras do
# MuJoCo na NVIDIA. Os três motivos estão no `launch_client_sim.sh`.
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$ENV/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export OMP_NUM_THREADS=1
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

echo "== servidor $ATHENA:5600 | MuJoCo local | painel http://127.0.0.1:8088/ =="

# SEM `--depth-legado`, e isto é o oposto do `launch_client_sim.sh`: a
# profundidade agora vem da RealSense do robô, que publica uint16 em
# milímetros (medido: 480x848, 0 a 2677 mm). O flag existe para o MuJoCo, que
# publica cinza de 3 canais em 8 bits; aplicá-lo aqui multiplicaria os
# milímetros por 2000/255 e entregaria uma cena 7,8x mais longe, sem erro
# nenhum na tela.
# `--pose-inicial` é obrigatório aqui, e não opcional: o corpo do MuJoCo nasce
# em zeros, e com estado zerado o modelo devolve a média mesmo olhando a bancada
# real (medido: 566 de 928 valores fora da faixa com zeros contra 2 de 928 com a
# pose do ep24, na MESMA imagem). O flag leva o corpo virtual à pose de prontidão
# das demonstrações antes de entregar o controle.
POSE="${POSE:-meu_dataset/white_cup_on_dripper_2026-08-11:24}"

exec "$ENV/bin/python" -u init_lerobot_inference_fastwamd_client.py \
    --server="$ATHENA" --sim \
    --cam-robot="$CAM_HOST" --port-cam=$CAM_PORTA --port-cam-pulso=$CAM_PORTA_PULSO \
    --pose-inicial="$POSE" --rampa=3 \
    --chunk=32 --lead=16 --fps=15 --v-web=8088 --debug "$@"
