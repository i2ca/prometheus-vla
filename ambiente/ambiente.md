# 🚀 Guia Completo: Restaurar Ambiente G1 (Conda)

---

## 🧰 0. Instalar o Miniconda (caso não tenha)

Se você ainda não tem o Conda instalado, siga:

### 📥 Baixar o instalador

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

---

### ⚙️ Instalar

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

Durante a instalação:

* Pressione `Enter` para avançar
* Digite `yes` para aceitar
* Confirme o caminho padrão (`~/miniconda3`)
* Digite `yes` para inicializar o Conda

---

### 🔄 Atualizar terminal

```bash
source ~/.bashrc
```

---

### ✅ Verificar instalação

```bash
conda list
```

Se aparecer uma lista de pacotes → tudo certo 👍

---

## 📥 1. Baixar o Ambiente

Baixe o arquivo completo diretamente do Google Drive:

👉 https://drive.google.com/file/d/1W2yDttDVfhc8dUxTo2qCIaBQWoHlhzTV/view?usp=drive_link

> ⚠️ O arquivo já está completo (`g1_env.tar.gz`), **não é necessário reconstruir partes**.

---

## 📦 2. Preparar o Ambiente no Novo PC

Após o download, mova o arquivo `g1_env.tar.gz` para o seu PC.

Crie a pasta do ambiente:

```bash
mkdir -p ~/miniconda3/envs/g1
```

> 💡 Se estiver usando Anaconda:

```bash
mkdir -p ~/anaconda3/envs/g1
```

---

## 📂 3. Extrair o Ambiente

```bash
tar -xzf g1_env.tar.gz -C ~/miniconda3/envs/g1
```

---

## ⚡ 4. Ativar o Ambiente

```bash
conda activate g1
```

---

## 🔧 5. Corrigir Caminhos Internos (IMPORTANTE)

```bash
conda-unpack
```

---

## 🧩 6. Instalar SDK2 (Unitree)

Entre na pasta do projeto:

```bash
cd prometheus-vla/unitree_sdk2_python
```

Instale no modo editável:

```bash
pip install -e .
```

---

## ✅ 7. Pronto!

Agora você tem o ambiente funcionando exatamente como no PC original:

* ✅ Sem precisar reinstalar dependências
* ✅ Sem conflitos de versão
* ✅ Pacotes editáveis funcionando
* ✅ Ambiente pronto para uso imediato

---
## 🔨 8. Teste a simulação

Entre no diretório correto para rodar a simulação:

```bash
cd ~/prometheus-vla/lerobot-ext
```

Em seguida, rode a simulação utilizando:

```bash
python init_lerobot_teleoparate.py --config_path=config/teleoperate_key.yaml --sim
```

---

## ❌ Possível erro (Simulação com travamentos)

Caso você esteja utilizando um computador com placa dedicada NVIDIA e também tenha uma placa integrada, o conflito entre as placas pode gerar travamentos na simulação. Para corrigir, utilize:

```bash
sudo prime-select nvidia
sudo reboot
```

Isso irá forçar seu computador a utilizar sempre a placa dedicada. Lembrando que o sudo reboot irá reiniciar o computador. Então cuidado para não perder nenhum arquivo não salvo no computador.

---

## 🎯 Resultado Final

Ambiente restaurado com sucesso, pronto para uso em:

* 🤖 Prometheus-VLA
* 🧪 Simulação com Unitree G1

---

## 💡 Dica Extra

Sempre ative o ambiente antes de rodar o projeto:

```bash
conda activate g1
```

---

**Fim do guia ✅**
