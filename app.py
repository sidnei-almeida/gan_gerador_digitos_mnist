import streamlit as st
import os
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from PIL import Image
import tensorflow as tf
import re
import requests
import tempfile

# Configuração da página
st.set_page_config(
    page_title="Gerador de Dígitos MNIST - GAN",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para tema escuro elegante (baseado no app Keras)
st.markdown("""
<style>
    /* Importar fontes elegantes */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Variáveis de cores premium (paleta fria) */
    :root {
        --primary-color: #46B3E6;
        --secondary-color: #7A7ADB;
        --accent-color: #5CE1E6;
        --dark-bg: #0E1117;
        --card-bg: #1E1E1E;
        --text-primary: #FAFAFA;
        --text-secondary: #B0B0B0;
        --gradient-primary: linear-gradient(135deg, #46B3E6 0%, #5CE1E6 100%);
        --gradient-secondary: linear-gradient(135deg, #7A7ADB 0%, #9A9AFF 100%);
        --gradient-dark: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        --shadow-soft: 0 8px 32px rgba(0, 0, 0, 0.3);
        --shadow-hover: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    
    /* Estilo global */
    .stApp {
        background: var(--dark-bg);
        color: var(--text-primary);
    }
    
    /* Ajustar o container principal para usar toda a largura */
    .main .block-container {
        max-width: none !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* SISTEMA DE TAMANHOS PADRONIZADO */
    
    /* Títulos principais */
    .main-header {
        font-size: 1.4rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Títulos de seção (H2) */
    h2 {
        font-size: 1.1rem !important;
        margin-bottom: 0.8rem !important;
        margin-top: 1.2rem !important;
    }
    
    /* Títulos de subseção (H3) */
    h3 {
        font-size: 0.95rem !important;
        margin-bottom: 0.6rem !important;
        margin-top: 0.8rem !important;
    }
    
    /* Títulos de subsubseção (H4) */
    h4 {
        font-size: 0.85rem !important;
        margin-bottom: 0.5rem !important;
        margin-top: 0.6rem !important;
    }
    
    /* Texto geral */
    p, li, span, div {
        font-size: 0.75rem !important;
        line-height: 1.4 !important;
    }
    
    /* Cards e containers */
    .metric-card, .data-card, .info-box {
        padding: 0.6rem !important;
        margin: 0.3rem 0 !important;
        border-radius: 8px !important;
    }
    
    /* Sidebar - ajustar tamanhos */
    .css-1d391kg {
        font-size: 0.75rem !important;
    }
    
    /* Elementos da sidebar */
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 {
        font-size: 0.9rem !important;
    }
    
    .css-1d391kg p, .css-1d391kg li, .css-1d391kg span {
        font-size: 0.7rem !important;
    }
    
    /* Botões */
    .stButton > button {
        font-size: 0.75rem !important;
        padding: 0.4rem 0.8rem !important;
    }
    
    /* Inputs e textareas */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        font-size: 0.75rem !important;
        padding: 0.4rem !important;
    }
    
    /* Tabelas */
    .stDataFrame {
        font-size: 0.7rem !important;
    }
    
    /* Gráficos - ajustar tamanho dos títulos */
    .plotly .gtitle {
        font-size: 0.9rem !important;
    }
    
    .plotly .xtitle, .plotly .ytitle {
        font-size: 0.75rem !important;
    }
    
    .plotly .legend {
        font-size: 0.7rem !important;
    }
    
    /* Header principal */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 0 4px 8px rgba(70, 179, 230, 0.3);
    }
    
    /* Cards de métricas premium */
    .metric-card {
        background: var(--gradient-dark);
        padding: 1rem;
        border-radius: 12px;
        color: var(--text-primary);
        text-align: center;
        margin: 0.3rem 0;
        border: 1px solid rgba(70, 179, 230, 0.2);
        box-shadow: var(--shadow-soft);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-primary);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-hover);
        border-color: var(--primary-color);
    }
    
    /* Cards de informação */
    .info-box {
        background: var(--card-bg);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
        box-shadow: var(--shadow-soft);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(70, 179, 230, 0.1) 0%, rgba(92, 225, 230, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
        box-shadow: var(--shadow-soft);
        border: 1px solid rgba(70, 179, 230, 0.3);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(122, 122, 219, 0.1) 0%, rgba(154, 154, 255, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--secondary-color);
        margin: 1rem 0;
        box-shadow: var(--shadow-soft);
        border: 1px solid rgba(122, 122, 219, 0.3);
    }
    
    /* Botões premium */
    .stButton > button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-soft);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-hover);
    }
    
    /* Sidebar elegante */
    .css-1d391kg {
        background: var(--card-bg);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Títulos elegantes */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    /* Texto secundário */
    .text-secondary {
        color: var(--text-secondary);
    }
    
    /* Cards de dados */
    .data-card {
        background: var(--card-bg);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: var(--shadow-soft);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Animações suaves */
    .fade-in {
        animation: fadeIn 0.6s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Scrollbar personalizada */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--dark-bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-color);
    }
</style>
""", unsafe_allow_html=True)


# Utilidades
GITHUB_USER = "sidnei-almeida"
GITHUB_REPO = "gan_gerador_digitos_mnist"
GITHUB_BRANCH = "main"
BASE_RAW = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/refs/heads/{GITHUB_BRANCH}"
BASE_GITHUB = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}"


@st.cache_resource
def load_generator_model():
    possible_paths = [
        os.path.join("modelos", "generator_model.keras"),
        os.path.join("notebooks", "generator_model.keras")
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return tf.keras.models.load_model(path, compile=False)
            except Exception as e:
                st.warning(f"Falha ao carregar modelo em {path}: {e}")
    # Fallback: baixar do GitHub Raw
    remote_paths = [
        f"{BASE_GITHUB}/raw/refs/heads/{GITHUB_BRANCH}/modelos/generator_model.keras",
        f"{BASE_RAW}/notebooks/generator_model.keras",
    ]
    for url in remote_paths:
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200 and resp.content:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmpf:
                    tmpf.write(resp.content)
                    tmp_path = tmpf.name
                model = tf.keras.models.load_model(tmp_path, compile=False)
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                return model
        except Exception as e:
            st.warning(f"Falha ao baixar modelo de {url}: {e}")
    return None


def list_checkpoint_files():
    ckpt_dir = os.path.join("training_checkpoints")
    if os.path.isdir(ckpt_dir):
        files = sorted(os.listdir(ckpt_dir))
        ckpt_state_path = os.path.join(ckpt_dir, "checkpoint")
        ckpt_state = None
        if os.path.exists(ckpt_state_path):
            try:
                with open(ckpt_state_path, "r") as f:
                    ckpt_state = f.read()
            except Exception as e:
                ckpt_state = f"Erro ao ler checkpoint: {e}"
        return files, ckpt_state
    # Fallback remoto: inferir nomes conhecidos a partir do repositório atual
    expected = ["checkpoint"] + [
        f"ckpt-{i}-{i}.index" for i in range(1, 12)
    ] + [
        f"ckpt-{i}-{i}.data-00000-of-00001" for i in range(1, 12)
    ]
    files = expected
    # Tentar baixar o arquivo 'checkpoint' para exibir conteúdo
    ckpt_state = None
    url = f"{BASE_RAW}/training_checkpoints/checkpoint"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            ckpt_state = r.text
    except Exception:
        ckpt_state = None
    return files, ckpt_state


def list_images():
    img_dir = os.path.join("imagens")
    if os.path.isdir(img_dir):
        files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        files.sort(key=lambda x: int(''.join([c for c in x if c.isdigit()]) or 0))
        return [os.path.join(img_dir, f) for f in files]
    # Fallback remoto: usar nomes conhecidos presentes no repositório
    expected_names = [f"image_for_ckpt_{i}.png" for i in range(0, 12)]
    return [f"{BASE_GITHUB}/blob/{GITHUB_BRANCH}/imagens/{name}?raw=true" for name in expected_names]


def generate_digits(generator, num_images=16, latent_dim=100, seed=None):
    if generator is None:
        return None
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(0, 1, (num_images, latent_dim))
    preds = generator.predict(noise, verbose=0)
    # Normaliza para [0,1]
    preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
    return preds


def grid_image_from_predictions(preds, grid_rows=4, grid_cols=4):
    if preds is None:
        return None
    h, w = preds.shape[1], preds.shape[2]
    channels = preds.shape[3] if preds.ndim == 4 else 1
    if channels == 1:
        canvas = np.zeros((grid_rows * h, grid_cols * w), dtype=np.float32)
    else:
        canvas = np.zeros((grid_rows * h, grid_cols * w, channels), dtype=np.float32)
    idx = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            if idx >= preds.shape[0]:
                break
            img = preds[idx]
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img[:, :, 0]
            if channels == 1 and img.ndim == 2:
                canvas[r*h:(r+1)*h, c*w:(c+1)*w] = img
            else:
                canvas[r*h:(r+1)*h, c*w:(c+1)*w, ...] = img
            idx += 1
    canvas = (canvas * 255).astype(np.uint8)
    return Image.fromarray(canvas)


def show_status(model_loaded, images, ckpt_files):
    # Status do Modelo
    model_status = "✅ Carregado" if model_loaded else "❌ Erro"
    model_color = "#46B3E6" if model_loaded else "#E66B6B"
    
    st.markdown(f"""
    <div style="background: rgba(70, 179, 230, 0.1); padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid {model_color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #FAFAFA; font-weight: 600;">🧠 Modelo GAN</span>
            <span style="color: {model_color}; font-weight: 700;">{model_status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Status das Imagens (checkpoints)
    images_status = "✅ Carregado" if len(images) > 0 else "❌ Erro"
    images_color = "#46B3E6" if len(images) > 0 else "#E66B6B"
    
    st.markdown(f"""
    <div style="background: rgba(70, 179, 230, 0.1); padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid {images_color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #FAFAFA; font-weight: 600;">🖼️ Imagens de Checkpoints ({len(images)})</span>
            <span style="color: {images_color}; font-weight: 700;">{images_status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Status dos Checkpoints
    ckpt_status = "✅ Carregado" if len(ckpt_files) > 0 else "❌ Erro"
    ckpt_color = "#46B3E6" if len(ckpt_files) > 0 else "#E66B6B"
    
    st.markdown(f"""
    <div style="background: rgba(70, 179, 230, 0.1); padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid {ckpt_color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #FAFAFA; font-weight: 600;">📦 Checkpoints ({len(ckpt_files)})</span>
            <span style="color: {ckpt_color}; font-weight: 700;">{ckpt_status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Status da Geração
    generation_status = "✅ Ativo" if model_loaded else "❌ Inativo"
    generation_color = "#46B3E6" if model_loaded else "#E66B6B"
    
    st.markdown(f"""
    <div style="background: rgba(70, 179, 230, 0.1); padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid {generation_color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #FAFAFA; font-weight: 600;">🔮 Geração</span>
            <span style="color: {generation_color}; font-weight: 700;">{generation_status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_project_info(generator, images, ckpt_files):
    """Mostra informações do projeto"""
    
    # Versão do App
    st.markdown("""
    <div style="background: rgba(70, 179, 230, 0.1); padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid #46B3E6;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #FAFAFA; font-weight: 600;">📱 Versão</span>
            <span style="color: #46B3E6; font-weight: 700;">v1.0.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Total de Imagens de Checkpoints
    st.markdown(f"""
    <div style="background: rgba(70, 179, 230, 0.1); padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid #46B3E6;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #FAFAFA; font-weight: 600;">🖼️ Imagens de Checkpoints</span>
            <span style="color: #46B3E6; font-weight: 700;">{len(images)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Total de Checkpoints
    st.markdown(f"""
    <div style="background: rgba(70, 179, 230, 0.1); padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid #46B3E6;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #FAFAFA; font-weight: 600;">📦 Checkpoints</span>
            <span style="color: #46B3E6; font-weight: 700;">{len(ckpt_files)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Status do Modelo
    model_status = "Ativo" if generator is not None else "Indisponível"
    model_color = "#46B3E6" if generator is not None else "#E66B6B"
    
    st.markdown(f"""
    <div style="background: rgba(70, 179, 230, 0.1); padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid {model_color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #FAFAFA; font-weight: 600;">🧠 Modelo</span>
            <span style="color: {model_color}; font-weight: 700;">{model_status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Última Atualização
    st.markdown("""
    <div style="background: rgba(70, 179, 230, 0.1); padding: 0.8rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid #46B3E6;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #FAFAFA; font-weight: 600;">🕒 Atualizado</span>
            <span style="color: #46B3E6; font-weight: 700;">Hoje</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def page_home(training_images):
    st.markdown('<h1 class="main-header fade-in">🧠 GAN - Gerador de Dígitos MNIST</h1>', unsafe_allow_html=True)
    st.markdown('<p class="text-secondary" style="text-align: center; font-size: 1.2rem; margin-bottom: 3rem;">Rede Neural GAN para Geração de Dígitos Manuscritos</p>', unsafe_allow_html=True)
    
    # Cards de métricas premium - OS PRÓPRIOS CARDS SÃO AS BARRAS
    col1, col2, col3 = st.columns(3)

    total_train = 42000
    total_test = 28000
    total_all = total_train + total_test
    train_pct = (total_train / total_all) * 100
    test_pct = (total_test / total_all) * 100

    with col1:
        # Amostras de Treino
        st.markdown(f'''
        <div class="fade-in" style="background: linear-gradient(90deg, #46B3E6 {train_pct:.1f}%, rgba(70, 179, 230, 0.1) {train_pct:.1f}%); border-radius: 6px; padding: 0.6rem; margin: 0.3rem 0; border: 1px solid rgba(70, 179, 230, 0.3); box-shadow: 0 3px 12px rgba(0, 0, 0, 0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                <span style="color: #FFFFFF; font-weight: 600; font-size: 0.75rem; text-shadow: 0 1px 3px rgba(0,0,0,0.8);">📊 Amostras de Treino</span>
                <span style="color: #FFFFFF; font-weight: 700; font-size: 0.8rem; text-shadow: 0 1px 3px rgba(0,0,0,0.8);">{total_train:,}</span>
            </div>
            <div style="background: rgba(255, 255, 255, 0.3); border-radius: 3px; height: 2px; margin: 0.25rem 0;">
                <div style="background: rgba(255, 255, 255, 0.6); height: 100%; width: 100%; border-radius: 3px;"></div>
            </div>
            <p style="color: #FFFFFF; margin-top: 0.25rem; font-size: 0.65rem; margin: 0; text-shadow: 0 1px 3px rgba(0,0,0,0.8);">Proporção no dataset: {train_pct:.1f}%</p>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        # Amostras de Teste
        st.markdown(f'''
        <div class="fade-in" style="background: linear-gradient(90deg, #5CE1E6 {test_pct:.1f}%, rgba(92, 225, 230, 0.1) {test_pct:.1f}%); border-radius: 6px; padding: 0.6rem; margin: 0.3rem 0; border: 1px solid rgba(92, 225, 230, 0.3); box-shadow: 0 3px 12px rgba(0, 0, 0, 0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                <span style="color: #FFFFFF; font-weight: 600; font-size: 0.75rem; text-shadow: 0 1px 3px rgba(0,0,0,0.8);">🧪 Amostras de Teste</span>
                <span style="color: #FFFFFF; font-weight: 700; font-size: 0.8rem; text-shadow: 0 1px 3px rgba(0,0,0,0.8);">{total_test:,}</span>
            </div>
            <div style="background: rgba(255, 255, 255, 0.3); border-radius: 3px; height: 2px; margin: 0.25rem 0;">
                <div style="background: rgba(255, 255, 255, 0.6); height: 100%; width: 100%; border-radius: 3px;"></div>
            </div>
            <p style="color: #FFFFFF; margin-top: 0.25rem; font-size: 0.65rem; margin: 0; text-shadow: 0 1px 3px rgba(0,0,0,0.8);">Proporção no dataset: {test_pct:.1f}%</p>
        </div>
        ''', unsafe_allow_html=True)

    with col3:
        # Quantidade de checkpoints detectados
        ckpt_files, _ = list_checkpoint_files()
        ckpt_count = len([f for f in ckpt_files if f.startswith('ckpt-')])
        st.markdown(f'''
        <div class="fade-in" style="background: linear-gradient(90deg, #7A7ADB {min(100, ckpt_count*5)}%, rgba(122, 122, 219, 0.1) {min(100, ckpt_count*5)}%); border-radius: 6px; padding: 0.6rem; margin: 0.3rem 0; border: 1px solid rgba(122, 122, 219, 0.3); box-shadow: 0 3px 12px rgba(0, 0, 0, 0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                <span style="color: #FFFFFF; font-weight: 600; font-size: 0.75rem; text-shadow: 0 1px 3px rgba(0,0,0,0.8);">📦 Checkpoints</span>
                <span style="color: #FFFFFF; font-weight: 700; font-size: 0.8rem; text-shadow: 0 1px 3px rgba(0,0,0,0.8);">{ckpt_count}</span>
            </div>
            <div style="background: rgba(255, 255, 255, 0.3); border-radius: 3px; height: 2px; margin: 0.25rem 0;">
                <div style="background: rgba(255, 255, 255, 0.6); height: 100%; width: 100%; border-radius: 3px;"></div>
            </div>
            <p style="color: #FFFFFF; margin-top: 0.25rem; font-size: 0.65rem; margin: 0; text-shadow: 0 1px 3px rgba(0,0,0,0.8);">Amostras geradas ao longo do treino</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Descrição do projeto com design premium
    st.markdown('<h2 style="color: #46B3E6; font-family: \'Inter\', sans-serif; font-weight: 600; margin-bottom: 0.8rem; font-size: 1.1rem;">📝 Sobre o Projeto</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p style="font-size: 0.75rem; line-height: 1.4; margin: 0;">
            Este projeto implementa um <strong>Gerador Adversarial (GAN)</strong> para criar dígitos manuscritos 
            realistas baseados no dataset MNIST. O modelo aprende a gerar imagens de dígitos de 0 a 9 
            através de treinamento adversário entre um gerador e um discriminador.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Características principais com cards premium
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="data-card">
            <h3 style="color: #46B3E6; font-family: 'Inter', sans-serif; font-weight: 600; margin-bottom: 0.6rem; font-size: 0.95rem;">🔧 Arquitetura GAN</h3>
            <ul style="color: #FAFAFA; line-height: 1.4; margin: 0; font-size: 0.75rem;">
                <li><strong>Gerador</strong>: Rede neural que cria imagens</li>
                <li><strong>Discriminador</strong>: Rede que distingue real/fake</li>
                <li><strong>Treinamento</strong>: Processo adversário competitivo</li>
                <li><strong>Entrada</strong>: Ruído aleatório (latent space)</li>
                <li><strong>Saída</strong>: Imagens 28x28 em tons de cinza</li>
                <li><strong>Dataset</strong>: MNIST (60.000 dígitos manuscritos)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="data-card">
            <h3 style="color: #46B3E6; font-family: 'Inter', sans-serif; font-weight: 600; margin-bottom: 0.6rem; font-size: 0.95rem;">📊 Funcionalidades</h3>
            <ul style="color: #FAFAFA; line-height: 1.4; margin: 0; font-size: 0.75rem;">
                <li><strong>Geração Interativa</strong>: Crie dígitos em tempo real</li>
                <li><strong>Análise de Qualidade</strong>: Métricas de diversidade</li>
                <li><strong>Evolução Visual</strong>: Acompanhe o treinamento</li>
                <li><strong>Travessia Latente</strong>: Explore o espaço latente</li>
                <li><strong>Checkpoints</strong>: Visualize progresso do modelo</li>
                <li><strong>Configuração</strong>: Ajuste parâmetros de geração</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Exemplos de evolução visual
    if training_images:
        st.markdown('<h2 style="color: #46B3E6; font-family: \'Inter\', sans-serif; font-weight: 600; margin: 1.2rem 0 0.8rem 0; font-size: 1.1rem;">🔭 Evolução do Treinamento</h2>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, img_path in enumerate(training_images[:8]):
            with cols[i % 4]:
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)


def page_model(generator):
    st.markdown("## 🤖 Modelo Gerador")
    if generator is None:
        st.warning("Modelo não encontrado em `modelos/generator_model.keras` ou `notebooks/generator_model.keras`.")
        return
    try:
        generator_summary = []
        generator.summary(print_fn=lambda x: generator_summary.append(x))
        st.code("\n".join(generator_summary))

        # Visual premium de dimensões de saída e parâmetros
        try:
            output_shape = generator.output_shape
            params = generator.count_params()
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-card"><div style="display:flex;justify-content:space-between;align-items:center;">
                    <span>🖼️ Saída</span><span style=\"color:#46B3E6;font-weight:800;\">{output_shape}</span>
                </div></div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card"><div style="display:flex;justify-content:space-between;align-items:center;">
                    <span>🔢 Parâmetros</span><span style=\"color:#46B3E6;font-weight:800;\">{params:,}</span>
                </div></div>
                """, unsafe_allow_html=True)
        except Exception:
            pass
    except Exception:
        st.info("Resumo indisponível.")


def page_checkpoints(ckpt_files, ckpt_state_text):
    st.markdown("## 📦 Checkpoints de Treinamento")
    if ckpt_state_text:
        st.markdown("### Estado do Checkpoint")
        st.code(ckpt_state_text)
    if not ckpt_files:
        st.info("Nenhum arquivo encontrado em `training_checkpoints/`.")
        return
    df = {"Arquivo": ckpt_files}
    st.dataframe(df, use_container_width=True)
    # Extração robusta de epoch/step do nome
    epochs = []
    order = []
    for f in ckpt_files:
        if f.startswith("ckpt-") and f.endswith(".index"):
            try:
                parts = f.replace('.index','').split('-')
                ep = int(parts[1])
                stp = int(parts[2]) if len(parts) > 2 else None
                epochs.append(ep)
                order.append((f, ep, stp))
            except Exception:
                pass
    if epochs:
        order.sort(key=lambda x: x[1])
        xs = list(range(1, len(order)+1))
        ys = [o[1] for o in order]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', line=dict(color='#46B3E6', width=3), name='Época'))
        fig.update_layout(title='Progresso por checkpoints', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#F2F6FA', xaxis_title='Ordem', yaxis_title='Época')
        st.plotly_chart(fig, use_container_width=True)


def page_images(training_images):
    st.markdown("## 🖼️ Imagens por Época")
    if not training_images:
        st.info("Coloque imagens em `imagens/` no formato image_for_ckpt_*.png.")
        return
    # Seleção de intervalo 50 em 50 (usaremos filtro por sufixo numérico)
    numbers = []
    for p in training_images:
        base = os.path.basename(p)
        digits = ''.join([c for c in base if c.isdigit()])
        if digits:
            try:
                numbers.append((p, int(digits)))
            except Exception:
                pass
    numbers.sort(key=lambda x: x[1])
    if not numbers:
        st.warning("Não foi possível identificar números de época nos nomes dos arquivos.")
        return
    min_ep = min(n for _, n in numbers)
    max_ep = max(n for _, n in numbers)
    default_step = 50 if (max_ep - min_ep) >= 50 else max(1, (max_ep - min_ep)//3 or 1)
    step = st.slider("Intervalo (épocas)", 1, max(2, max_ep - min_ep if (max_ep - min_ep)>0 else 2), default_step, 1)
    filtered = [p for p, n in numbers if ((n - min_ep) % step == 0) or n in (min_ep, min_ep+1)]
    if not filtered:
        filtered = [p for p, _ in numbers]
    cols = st.columns(4)
    for i, img_path in enumerate(filtered):
        with cols[i % 4]:
            st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)


def page_generation(generator):
    st.markdown('<h2 style="color: #46B3E6; font-family: \'Inter\', sans-serif; font-weight: 600; margin-bottom: 2rem;">🎨 Geração Interativa</h2>', unsafe_allow_html=True)
    
    if generator is None:
        st.markdown("""
        <div class="warning-box">
            <p style="font-size: 0.75rem; line-height: 1.4; margin: 0;">
                <strong>⚠️ Modelo não encontrado</strong><br>
                Carregue um modelo gerador válido em <code>modelos/generator_model.keras</code> ou <code>notebooks/generator_model.keras</code> para usar esta funcionalidade.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Interface de geração premium
    st.markdown('<h3 style="color: #46B3E6; font-family: \'Inter\', sans-serif; font-weight: 600; margin-bottom: 1rem;">⚙️ Configurações de Geração</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="data-card">
            <h4 style="color: #46B3E6; font-family: 'Inter', sans-serif; font-weight: 600; margin-bottom: 0.6rem;">🎲 Parâmetros</h4>
        </div>
        """, unsafe_allow_html=True)
        latent_dim = st.number_input("Dimensão do ruído (latent)", min_value=16, max_value=256, value=100, step=4, help="Tamanho do vetor de ruído de entrada")
        seed = st.number_input("Seed (opcional)", min_value=0, max_value=10_000, value=0, step=1, help="Para reproduzir resultados")
    
    with col2:
        st.markdown("""
        <div class="data-card">
            <h4 style="color: #46B3E6; font-family: 'Inter', sans-serif; font-weight: 600; margin-bottom: 0.6rem;">📐 Layout</h4>
        </div>
        """, unsafe_allow_html=True)
        grid = st.selectbox("Grade", options=["4x4", "5x5", "6x6"], index=0, help="Número de dígitos por linha/coluna")
        grid_map = {"4x4": (4,4), "5x5": (5,5), "6x6": (6,6)}
        rows, cols_ = grid_map[grid]
        count = rows * cols_
    
    with col3:
        st.markdown("""
        <div class="data-card">
            <h4 style="color: #46B3E6; font-family: 'Inter', sans-serif; font-weight: 600; margin-bottom: 0.6rem;">🎯 Ação</h4>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Gerar Dígitos", type="primary", use_container_width=True):
            with st.spinner("Gerando dígitos..."):
                preds = generate_digits(generator, num_images=count, latent_dim=latent_dim, seed=seed)
                img = grid_image_from_predictions(preds, grid_rows=rows, grid_cols=cols_)
                if img is not None:
                    st.session_state.generated_image = img
                    st.session_state.generated_predictions = preds
                    st.session_state.generation_params = {
                        'latent_dim': latent_dim,
                        'seed': seed,
                        'grid': grid
                    }
                else:
                    st.error("Falha na geração.")
    
    # Exibir resultado da geração
    if 'generated_image' in st.session_state:
        st.markdown("---")
        st.markdown('<h3 style="color: #46B3E6; font-family: \'Inter\', sans-serif; font-weight: 600; margin-bottom: 1rem;">🖼️ Resultado da Geração</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(st.session_state.generated_image, caption=f"Grade {st.session_state.generation_params['grid']} - Latent {st.session_state.generation_params['latent_dim']}", use_container_width=True)
        
        with col2:
            # Métricas de qualidade premium
            if 'generated_predictions' in st.session_state:
                preds = st.session_state.generated_predictions
                
                # Diversidade: média da distância L2 entre pares amostrados
                flat = preds.reshape(preds.shape[0], -1)
                sample_idx = np.linspace(0, flat.shape[0]-1, num=min(16, flat.shape[0]), dtype=int)
                subset = flat[sample_idx]
                dists = []
                for i in range(len(subset)):
                    for j in range(i+1, len(subset)):
                        dists.append(np.linalg.norm(subset[i]-subset[j]))
                diversity = float(np.mean(dists)) if dists else 0.0
                
                # Esparsidade média (% de pixels > 0.5)
                sparsity = float((flat > 0.5).mean())
                
                # Simetria (diferença média entre metades esquerda/direita)
                imgs = preds if preds.ndim == 4 else preds[..., np.newaxis]
                left = imgs[..., :imgs.shape[2]//2, :]
                right = np.flip(imgs[..., imgs.shape[2]//2:, :], axis=2)
                symmetry = float(np.mean(np.abs(left - right)))
                
                st.markdown("""
                <div class="data-card">
                    <h4 style="color: #46B3E6; font-family: 'Inter', sans-serif; font-weight: 600; margin-bottom: 0.6rem;">📊 Métricas de Qualidade</h4>
                </div>
                """, unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span>🔀 Diversidade</span><span style="color:#46B3E6;font-weight:800;">{diversity:.3f}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span>🧩 Esparsidade</span><span style="color:#46B3E6;font-weight:800;">{sparsity:.1%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span>🪞 Assimetria</span><span style="color:#46B3E6;font-weight:800;">{symmetry:.3f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Travessia de Latente
    st.markdown("---")
    st.markdown('<h3 style="color: #46B3E6; font-family: \'Inter\', sans-serif; font-weight: 600; margin-bottom: 1rem;">🔍 Travessia do Espaço Latente</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p style="font-size: 0.75rem; line-height: 1.4; margin: 0;">
            <strong>Exploração do Espaço Latente:</strong> Varie uma dimensão específica do vetor de ruído 
            para ver como ela afeta a geração. Isso revela como o modelo organiza as características visuais.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_a, col_b = st.columns([3, 1])
    
    with col_b:
        st.markdown("""
        <div class="data-card">
            <h4 style="color: #46B3E6; font-family: 'Inter', sans-serif; font-weight: 600; margin-bottom: 0.6rem;">🎛️ Controles</h4>
        </div>
        """, unsafe_allow_html=True)
        latent_dim_trav = st.number_input("Latent dim", min_value=16, max_value=256, value=100, step=4, key='trav_latent')
        dim_idx = st.number_input("Dimensão a variar", min_value=0, max_value=255, value=0, step=1, key='trav_idx')
        steps = st.slider("Passos", 5, 15, 9, 1)
        base_seed = st.number_input("Seed base", min_value=0, max_value=10000, value=42, step=1, key='trav_seed')
        
        if st.button("🔍 Gerar Travessia", type="secondary", use_container_width=True):
            with st.spinner("Gerando travessia..."):
                np.random.seed(base_seed)
                base = np.random.normal(0, 1, (steps, latent_dim_trav))
                vals = np.linspace(-2.0, 2.0, steps)
                dim_idx = int(min(dim_idx, latent_dim_trav-1))
                for i,v in enumerate(vals):
                    base[i, dim_idx] = v
                preds = generator.predict(base, verbose=0)
                preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
                strip = grid_image_from_predictions(preds, grid_rows=1, grid_cols=steps)
                st.session_state.latent_traversal = strip
                st.session_state.traversal_params = {
                    'dim_idx': dim_idx,
                    'steps': steps,
                    'seed': base_seed
                }
    
    with col_a:
        if 'latent_traversal' in st.session_state:
            st.image(st.session_state.latent_traversal, 
                   caption=f"Travessia na dimensão {st.session_state.traversal_params['dim_idx']} (seed: {st.session_state.traversal_params['seed']})", 
                   use_container_width=True)
        else:
            st.markdown("""
            <div class="data-card" style="text-align: center; padding: 2rem;">
                <p style="color: #B0B0B0; font-size: 0.75rem; margin: 0;">Configure os parâmetros e clique em "Gerar Travessia" para explorar o espaço latente</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Histograma de intensidades
    if 'generated_predictions' in st.session_state:
        st.markdown("---")
        st.markdown('<h3 style="color: #46B3E6; font-family: \'Inter\', sans-serif; font-weight: 600; margin-bottom: 1rem;">📈 Análise de Distribuição</h3>', unsafe_allow_html=True)
        
        preds = st.session_state.generated_predictions
        flat = preds.reshape(preds.shape[0], -1)
        
        fig = px.histogram(flat.flatten(), nbins=30, title='Distribuição de Intensidades dos Pixels')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)', 
            font_color='#FAFAFA',
            title_font_size=14,
            xaxis_title="Intensidade (0-1)",
            yaxis_title="Frequência"
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    # Carregamentos básicos
    generator = load_generator_model()
    images = list_images()
    ckpt_files, ckpt_state_text = list_checkpoint_files()

    # Menu de navegação premium com streamlit-option-menu
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #46B3E6; font-family: 'Inter', sans-serif; font-weight: 700;">🎯 Navegação</h2>
        </div>
        """, unsafe_allow_html=True)
        
        selected = option_menu(
            menu_title=None,
            options=["Início", "Modelo", "Checkpoints", "Imagens", "Geração"],
            icons=["house", "cpu", "boxes", "image", "magic"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#46B3E6", "font-size": "20px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#1E1E1E",
                    "color": "#B0B0B0",
                    "font-family": "'Inter', sans-serif",
                    "font-weight": "500",
                },
                "nav-link-selected": {
                    "background-color": "rgba(70, 179, 230, 0.1)",
                    "color": "#46B3E6",
                    "border-left": "4px solid #46B3E6",
                    "border-radius": "8px",
                },
            }
        )
        
        # Separador
        st.markdown("---")
        
        # Status do Sistema
        st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h3 style="color: #46B3E6; font-family: 'Inter', sans-serif; font-weight: 600; margin-bottom: 1rem;">📊 Status do Sistema</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Status dos Componentes
        show_status(generator is not None, images, [f for f in ckpt_files if f.startswith('ckpt-')])
        
        # Separador
        st.markdown("---")
        
        # Informações do Projeto
        st.markdown("""
        <div style="margin-bottom: 1.5rem;">
            <h3 style="color: #46B3E6; font-family: 'Inter', sans-serif; font-weight: 600; margin-bottom: 1rem;">ℹ️ Informações</h3>
        </div>
        """, unsafe_allow_html=True)
        
        show_project_info(generator, images, ckpt_files)

    if selected == "Início":
        page_home(images)
    elif selected == "Modelo":
        page_model(generator)
    elif selected == "Checkpoints":
        page_checkpoints(ckpt_files, ckpt_state_text)
    elif selected == "Imagens":
        page_images(images)
    elif selected == "Geração":
        page_generation(generator)


if __name__ == "__main__":
    main()


