import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

# Caminho exato apontando para o seu NOVO dataset da chaleira
caminho_parquet = "meu_dataset/get_the_kettle2/data/chunk-000/file-000.parquet"

print(f"Carregando o dataset na memória (isso pode levar alguns segundos)...")
df = pd.read_parquet(caminho_parquet)

# Descobre todos os episódios disponíveis
if 'episode_index' in df.columns:
    episodios_disponiveis = df['episode_index'].unique()
    print(f"✅ Encontrados {len(episodios_disponiveis)} episódios.")
else:
    episodios_disponiveis = [0]

indice_atual = 0

# ==========================================
# 1. CONFIGURAÇÃO DA RESOLUÇÃO E LAYOUT
# ==========================================
# Aumentamos o tamanho base da figura (largura, altura)
fig, axs = plt.subplots(5, 1, figsize=(14, 9), sharex=True)

# O SEGREDO AQUI: 
# hspace=0.6 (Afasta os gráficos verticalmente)
# bottom=0.12 (Abre espaço livre em baixo para os botões)
# left/right (Usa melhor a largura da sua tela)
plt.subplots_adjust(bottom=0.12, hspace=0.6, top=0.95, left=0.06, right=0.97)

def atualizar_grafico(val=None):
    global indice_atual
    episodio_alvo = episodios_disponiveis[indice_atual]
    
    # Limpa a tela e adiciona as grades de leitura
    for ax in axs:
        ax.clear()
        ax.grid(True, linestyle='--', alpha=0.6) # Deixa a leitura dos números muito melhor

    # Extrai os dados
    df_episodio = df[df['episode_index'] == episodio_alvo]
    estados = np.vstack(df_episodio['observation.state'].values)

    motores     = estados[:, 0:43]
    pressao_esq = estados[:, 43:76]
    pressao_dir = estados[:, 76:109]
    temp_esq    = estados[:, 109:142]
    temp_dir    = estados[:, 142:175]

    # ==========================================
    # 2. PLOTAGEM DINÂMICA (margins x=0 remove espaços laterais)
    # ==========================================
    axs[0].plot(motores)
    axs[0].set_title(f"1. Posição dos 43 Motores | Episódio Alvo: {episodio_alvo} ({indice_atual+1} de {len(episodios_disponiveis)})", fontweight='bold')
    axs[0].margins(x=0)

    axs[1].plot(pressao_esq)
    axs[1].set_title("2. Força Bruta - Mão Esquerda", fontweight='bold')
    axs[1].margins(x=0)

    axs[2].plot(pressao_dir)
    axs[2].set_title("3. Força Bruta - Mão Direita", fontweight='bold')
    axs[2].margins(x=0)

    axs[3].plot(temp_esq)
    axs[3].set_title("4. Temperatura (°C) - Mão Esquerda", fontweight='bold')
    axs[3].margins(x=0)

    axs[4].plot(temp_dir)
    axs[4].set_title("5. Temperatura (°C) - Mão Direita", fontweight='bold')
    axs[4].set_xlabel("Frames do Episódio (Tempo)", fontweight='bold')
    axs[4].margins(x=0)

    plt.draw() # Renderiza a atualização

# ==========================================
# 3. BOTÕES BEM POSICIONADOS
# ==========================================
ax_anterior = plt.axes([0.35, 0.02, 0.12, 0.05])
ax_proximo  = plt.axes([0.53, 0.02, 0.12, 0.05])

btn_anterior = Button(ax_anterior, '<< Anterior')
btn_proximo  = Button(ax_proximo, 'Próximo >>')

def proximo(event):
    global indice_atual
    if indice_atual < len(episodios_disponiveis) - 1:
        indice_atual += 1
        atualizar_grafico()

def anterior(event):
    global indice_atual
    if indice_atual > 0:
        indice_atual -= 1
        atualizar_grafico()

btn_anterior.on_clicked(anterior)
btn_proximo.on_clicked(proximo)

atualizar_grafico()

# ==========================================
# 4. TENTA MAXIMIZAR A JANELA AO ABRIR (Ubuntu / Windows)
# ==========================================
try:
    mng = plt.get_current_fig_manager()
    # Para o backend Qt5Agg (Padrão do Ubuntu)
    mng.window.showMaximized()
except:
    pass

plt.show()