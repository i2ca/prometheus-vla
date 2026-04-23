<h1>Documentação Técnica: Modelo ACT-D (Action Chunking Transformer with Depth & Tactile)</h1>

<h2>1. Visão Geral</h2>
<p>O <strong>ACT-D</strong> é uma evolução do modelo Action Chunking Transformer (ACT), adaptado para o ecossistema <strong>LeRobot</strong>. Ele expande a percepção robótica ao fundir nativamente dados de <strong>Visão 3D (Profundidade)</strong> e <strong>Feedback Tátil (Pressão)</strong> com as informações visuais RGB e proprioceptivas padrão.</p>

<h2>2. Arquitetura e Fluxo de Dados</h2>

<h3>2.1. Percepção Espacial 3D</h3>
<ul>
    <li><strong>Reversão e Projeção:</strong> O mapa de profundidade é convertido em metros reais e projetado em um espaço cartesiano 3D usando parâmetros intrínsecos da câmera (fx, fy, cx, cy).</li>
    <li><strong>Amostragem:</strong> A nuvem de pontos é filtrada (removendo ruídos a menos de 5cm) e amostrada para exatamente 1024 pontos para garantir eficiência de processamento.</li>
    <li><strong>PointNet Encoder:</strong> Uma rede PointNet extrai características geométricas globais, transformando a nuvem de pontos em um vetor de features de alta dimensão.</li>
</ul>

<h3>2.2. Percepção Tátil</h3>
<ul>
    <li><strong>Fusão de Pressão:</strong> Os dados de sensores táteis (ex: 66 dimensões combinando mão esquerda e direita) são processados por um MLP (Linear + ReLU) para alinhar sua dimensionalidade com o Transformer.</li>
</ul>

<h3>2.3. Mecanismo de Fusão Nativa</h3>
<p>Diferente de abordagens que apenas concatenam vetores, o ACT-D utiliza <strong>Fusão Aditiva</strong>. As características 3D e táteis são somadas diretamente ao token de estado (propriocepção) do robô antes de entrarem no Encoder do Transformer:</p>
<pre><code>state_token = state_token + features_3d + features_pressure</code></pre>

<h2>3. Referência dos Componentes do Sistema</h2>

<ul>
    <li><strong>modeling_act.py:</strong> Implementação central da rede neural, contendo a lógica de integração nativa do PointNet e do projetor de pressão no fluxo do ACT.</li>
    <li><strong>configuration_act.py:</strong> Define os hiperparâmetros do modelo, incluindo flags para ativar visão 3D (<code>use_depth_3d</code>) e tato (<code>use_pressure</code>).</li>
    <li><strong>depth_encoder.py:</strong> Contém a lógica matemática para converter mapas de profundidade em nuvens de pontos e a arquitetura da PointNet.</li>
    <li><strong>run_train.py:</strong> Script de treinamento avançado com suporte a datasets de validação e rastreamento de métricas de variância.</li>
    <li><strong>act_d_injector.py:</strong> Módulo de "injeção" que permite aplicar as melhorias do ACT-D via Monkey Patch em instâncias existentes do LeRobot, sem modificar o código fonte original.</li>
    <li><strong>utils.py:</strong> Utilitário <code>VarianceMeter</code> para monitorar a estabilidade do treinamento através do desvio padrão da perda (loss).</li>
</ul>

<h2>4. Processo de Treinamento e Validação</h2>
<ul>
    <li><strong>Dataloading Customizado:</strong> O sistema utiliza um patch no <code>LeRobotDataset</code> para gerenciar mapeamentos de índices globais e relativos, facilitando o uso de subconjuntos de dados.</li>
    <li><strong>Validação em Tempo Real:</strong> Durante o treino, o sistema executa loops de validação periódicos, calculando a média e o desvio padrão da performance no dataset de teste.</li>
    <li><strong>Cálculo de Frames:</strong> O utilitário <code>calculate_frames.py</code> permite analisar a configuração YAML para prever passos por época e o tamanho total do dataset antes do início do processo.</li>
</ul>

<h2>5. Notas de Implementação</h2>
<p>O modelo foi projetado para ignorar o canal de profundidade nas backbones ResNet padrão, tratando-o exclusivamente através do pipeline geométrico para evitar distorções de características visuais. A integração tátil suporta dimensões customizáveis para diferentes configurações de sensores de pressão.</p>