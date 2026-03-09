# Como rodar o Async Inference na Atena

## Troubleshooting de Rede

(Robô + Laboratório + VPN)

🎯 Objetivo
	Permitir que o robô (Unitree G1) fosse acessado pela rede do laboratório (10.9.8.0/) mantendo:

	Rede interna fixa do robô ( 192.168.123.164/24 )

	Conectividade externa

	VPN ativa no computador pessoal

### 🧩 1️⃣ Problema Inicial
	O robô possuía apenas:
	
	192.168.123.164/24
	Ele não participava da rede do laboratório.
	Tentativas de SSH retornavam:
		No route to host

### 🔧 2️⃣ Solução no Robô – Adicionar Segundo IP
	Foi executado:
		sudo ip addr add 10.9.8.73/24 dev eth0
### Resultado:
	A interface passou a ter dois IPs:
	192.168.123.164/24 (rede interna)
	10.9.8.73/24 (rede laboratório/IP do Prometheus na Rede Local)
	✔ Comunicação via SSH funcionando ✔ Ping funcionando

### 🧠 Conceito Importante
	Uma interface pode ter múltiplos endereços IP.
	Isso permite que o robô participe simultaneamente de duas redes
	
## Trocar o ip fixo do Prometheus pelo ip dele na Rede Local:

nvim ~/prometheus-vla/lerobot/src/lerobot/robots/unitree_g1/config_unitree_g1.py

Mude o ip de robot_ip para "10.9.8.73":

```python
    #Socket config for ZMQ bridge
    #robot_ip: str = "192.168.123.164"  # default G1 IP
    robot_ip: str = "10.9.8.73" #Prometheus Local IP
```

## Atenção:

Caso esteja rodando o servidor na Atena, mas o cliente no seu computador pessoal, você deve mudar a flag do (--server_address) para o ip da Atena, como no exemplo:

python -m lerobot.async_inference.robot_client     --server_address 10.9.8.252:8080     --robot.type unitree_g1_dex3     --robot.is_simulation false     --robot.control_mode upper_body     --task "Pick up the kettle"     --policy_type act     --pretrained_name_or_path train/output/checkpoints/last/pretrained_model/     --policy_device cuda     --actions_per_chunk 100     --chunk_size_threshold 0.5 --debug_visualize_queue_size=True

OBS: O modelo deve estar presente no disco da Atena, se não irá ocorrer o seguinte erro: ERROR - Exception calling application: Repo id must be in the form 'repo_name' or 'namespace/repo_name'
