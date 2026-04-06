import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
import os

# Caminho do dataset de teste
caminho_base = "meu_dataset/teste2/data/chunk-000/"
arquivo_parquet = "episode_000.parquet"

# Tenta encontrar o arquivo (pode ser episode_000 ou file-000 dependendo da versão)
caminho_parquet = os.path.join(caminho_base, arquivo_parquet)
if not os.path.exists(caminho_parquet):
    caminho_parquet = os.path.join(caminho_base, "file-000.parquet")

print(f"Carregando o dataset: {caminho_parquet}")
try:
    df = pd.read_parquet(caminho_parquet)
except Exception as e:
    print(f"❌ Erro ao carregar Parquet: {e}")
    exit()

# Descobre todos os episódios disponíveis no arquivo
if 'episode_index' in df.columns:
    episodios_disponiveis = df['episode_index'].unique()
    print(f"✅ Encontrados {len(episodios_disponiveis)} episódios no arquivo.")
else:
    episodios_disponiveis = [0]

indice_atual = 0

# ==========================================
# CONFIGURAÇÃO DO PLAYER DE VÍDEO
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25, top=0.90)

video_frames = None
imagem_plotada = None

def carregar_episodio():
    global video_frames, imagem_plotada
    episodio_alvo = episodios_disponiveis[indice_atual]
    
    df_episodio = df[df['episode_index'] == episodio_alvo].copy()
    
    # 🌟 O SEGREDO: Desempacota o super vetor do LeRobot
    # O LeRobot empilha motores (43) + pressões (66) = 109 primeiros valores.
    # A profundidade (768) começa exatamente no índice 109.
    estados = np.vstack(df_episodio['observation.state'].values)
    
    inicio_depth = 109
    fim_depth = inicio_depth + 768
    
    dados_brutos = estados[:, inicio_depth : fim_depth]
    num_frames = dados_brutos.shape[0]
    
    # Redimensiona os 768 números para (Frames, Altura 24, Largura 32)
    video_frames = dados_brutos.reshape((num_frames, 24, 32))
    
    ax.clear()
    # vmin/vmax ajustados para a escala de 0 a 3 metros do seu simulador/RealSense
    imagem_plotada = ax.imshow(video_frames[0], cmap='plasma', interpolation='nearest', vmin=0.0, vmax=3.0)
    
    ax.set_title(f"Visão Espacial da IA (Matriz 32x24)\nEpisódio: {episodio_alvo} | Frames: {num_frames}", fontweight='bold')
    ax.axis('off')
    
    # Barra de cores (Colorbar) para referência de metros
    if not hasattr(carregar_episodio, "cbar_criada"):
        cbar = plt.colorbar(imagem_plotada, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Distância em Metros', rotation=270, labelpad=15)
        carregar_episodio.cbar_criada = True

    slider_tempo.valmax = num_frames - 1
    slider_tempo.set_val(0)
    plt.draw()

def atualizar_frame(val):
    if video_frames is not None and imagem_plotada is not None:
        frame_idx = int(slider_tempo.val)
        imagem_plotada.set_data(video_frames[frame_idx])
        fig.canvas.draw_idle()

# ==========================================
# CONTROLES INTERATIVOS
# ==========================================
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider_tempo = Slider(ax_slider, 'Tempo', 0, 100, valinit=0, valstep=1)
slider_tempo.on_changed(atualizar_frame)

ax_anterior = plt.axes([0.35, 0.02, 0.12, 0.05])
ax_proximo  = plt.axes([0.53, 0.02, 0.12, 0.05])

btn_anterior = Button(ax_anterior, '<< Ep. Anterior')
btn_proximo  = Button(ax_proximo, 'Próximo Ep. >>')

def proximo(event):
    global indice_atual
    if indice_atual < len(episodios_disponiveis) - 1:
        indice_atual += 1
        carregar_episodio()

def anterior(event):
    global indice_atual
    if indice_atual > 0:
        indice_atual -= 1
        carregar_episodio()

btn_anterior.on_clicked(anterior)
btn_proximo.on_clicked(proximo)

# Tenta maximizar a janela automaticamente
try:
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
except:
    pass

carregar_episodio()
plt.show()