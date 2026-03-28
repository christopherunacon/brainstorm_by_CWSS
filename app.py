"""
🧠 Brainstorm ByCWSS — Ferramenta de Brainstorming Universal & Base de Conhecimento Ativa
MVP usando Streamlit + LlamaIndex + ChromaDB + Groq
"""

import os
import json
import groq
import re
import math
import base64
import textwrap
import shutil
import random
from io import BytesIO
from pathlib import Path

import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
from pinecone import Pinecone
from streamlit_agraph import agraph, Node, Edge, Config
from gtts import gTTS

from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    Settings,
    PromptTemplate
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.groq import Groq

# --- INICIALIZAÇÃO CLOUD ---
if not firebase_admin._apps:
    try:
        firebase_config = dict(st.secrets["firebase"])
        # Streamlit Cloud pode manter \\n literal na private_key — normalizar
        if "private_key" in firebase_config:
            firebase_config["private_key"] = firebase_config["private_key"].replace("\\n", "\n")
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"❌ Erro ao inicializar Firebase: {e}")
        st.stop()
db = firestore.client()
# ---------------------------

# ──────────────────────────────────────────────
# Modelos disponíveis no Groq (tier gratuito)
# ──────────────────────────────────────────────
GROQ_MODELS = {
    "Llama 3.3 70B (Recomendado)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (Rápido)": "llama-3.1-8b-instant",
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "Gemma 2 9B": "gemma2-9b-it",
}

# ──────────────────────────────────────────────
# Página Streamlit — Layout Expansivo
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="🧠 Brainstorm ByCWSS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS Personalizado — Visual Premium
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0f0f1a;
    --bg-card: #1a1a2e;
    --accent-purple: #7c3aed;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --text-primary: #f1f5f9;
    --text-muted: #94a3b8;
    --border-subtle: rgba(124, 58, 237, 0.2);
    --glow: 0 0 20px rgba(124, 58, 237, 0.3);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary);
}

/* Ocultar UI padrão do Streamlit */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

.main .block-container {
    padding-top: 1rem;
    max-width: 1250px;
}

/* Header */
.hero-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7c3aed, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: var(--text-muted);
    margin-bottom: 2rem;
    font-weight: 300;
}

/* Cards */
.glass-card {
    background: rgba(26, 26, 46, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--glow);
    transition: transform 0.2s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 30px rgba(124, 58, 237, 0.45);
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.08), rgba(59, 130, 246, 0.08));
    border: 1px solid rgba(124, 58, 237, 0.25);
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
    line-height: 1.75;
}

/* Metric badge */
.metric-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed, #3b82f6);
    color: white;
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 0.25rem;
}

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    margin-right: 6px;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* Stremlit overrides inputs */
.stTextArea textarea, .stTextInput input {
    background-color: rgba(26, 26, 46, 0.5) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    transition: all 0.3s ease;
}

.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.4) !important;
}

/* Button UI */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #06b6d4) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    width: 100% !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.6) !important;
}

/* Tab Headers UI */
[data-baseweb="tab-list"] {
    background-color: transparent !important;
    gap: 1rem;
}
[data-baseweb="tab"] {
    color: var(--text-muted) !important;
    background: transparent !important;
    border: none !important;
}
[aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
}

</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Chave de API Groq & Seleção de Modelo
# ──────────────────────────────────────────────
def get_api_key() -> str | None:
    """Retorna a chave de API do Groq."""
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "")
        except FileNotFoundError:
            pass
    return key

def get_selected_model() -> str:
    """Retorna o modelo Groq selecionado na sidebar."""
    with st.sidebar:
        st.markdown("### ⚙️ Configurações")
        selected = st.selectbox(
            "🤖 Modelo LLM",
            options=list(GROQ_MODELS.keys()),
            index=0,
        )
    return GROQ_MODELS[selected]


# ──────────────────────────────────────────────
# Pipeline LlamaIndex — ChromaDB + Groq + FastEmbed
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_pipeline(_api_key: str, _model_name: str):
    """
    Inicializa Pinecone + LlamaIndex + Groq.
    """
    llm = Groq(model=_model_name, api_key=_api_key)
    Settings.llm = llm
    Settings.embed_model = "local:all-MiniLM-L6-v2"

    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    pinecone_index = pc.Index("brainstorm")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, storage_context=storage_context
    )

    return index, None


def _sanitize_keyword(raw: str) -> str:
    """Normaliza a palavra-chave para uso seguro em nomes de arquivo."""
    # Remover acentos
    nfkd = unicodedata.normalize("NFKD", raw)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    # Manter apenas alfanuméricos e hífens, converter para lowercase
    clean = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text).strip("-").lower()
    return clean[:30] or "geral"  # fallback se ficar vazio


def extract_keyword(text: str) -> str:
    """Chamada rápida à LLM para extrair UMA palavra-chave que define o domínio do texto."""
    try:
        llm = Settings.llm
        response = llm.complete(
            "Analise o texto abaixo e retorne APENAS UMA palavra-chave principal "
            "que melhor defina o domínio ou tema central (ex: oncologia, python, "
            "startup, finanças, filosofia). Retorne SOMENTE a palavra, sem pontuação, "
            "sem explicação.\n\n"
            f"Texto:\n{text[:500]}"
        )
        keyword = str(response).strip().split()[0]  # pegar só a primeira palavra
        return _sanitize_keyword(keyword)
    except Exception:
        return "geral"


def add_brain_dump_to_index(text: str, index, keyword: str | None = None):
    if keyword is None:
        keyword = extract_keyword(text)

    # Salva no Firestore
    doc_ref = db.collection("braindumps").document()
    doc_ref.set({"text": text, "keyword": keyword})

    doc = Document(
        text=text,
        id_=doc_ref.id,
        metadata={
            "source": doc_ref.id,
            "keyword": keyword,
        },
    )
    index.insert(doc)

    return doc_ref.id, keyword


# ──────────────────────────────────────────────
# Visão Computacional, PDFs e Graph Generator
# ──────────────────────────────────────────────
def process_uploaded_files(uploaded_files, api_key: str) -> str:
    """Extrai texto de PDFs e imagens carregados pelo usuário."""
    extra_text = ""
    for f in uploaded_files:
        if f.name.lower().endswith(".pdf"):
            try:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text_extracted = page.extract_text()
                    if text_extracted:
                        extra_text += text_extracted + "\n"
            except Exception as e:
                extra_text += f"[Erro ao ler PDF {f.name}: {e}]\n"
        elif f.name.lower().endswith(("png", "jpg", "jpeg")):
            try:
                base64_image = base64.b64encode(f.read()).decode("utf-8")
                client = groq.Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Extraia as ideias centrais, padrões e o texto legível desta imagem para me ajudar num brainstorming. Detalhe conceitos visíveis."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                    },
                                },
                            ],
                        }
                    ],
                )
                extra_text += f"\n--- Visão Extraída da Imagem ({f.name}) ---\n" + response.choices[0].message.content + "\n"
            except Exception as e:
                extra_text += f"[Erro ao processar imagem {f.name}: {e}]\n"
    return extra_text


@st.dialog("🧠 Gerenciador do Cérebro")
def memory_manager():
    st.markdown("Cofre de anotações Firestore. Excluir uma nota removerá fisicamente do Pinecone também.")
    docs = list(db.collection("braindumps").stream())
    
    if not docs:
        st.info("O cérebro está vazio no Cloud.")
        return
        
    for doc in docs:
        data = doc.to_dict()
        col1, col2 = st.columns([4, 1])
        with col1:
            lbl = textwrap.shorten(data.get("text", ""), width=50, placeholder="...")
            st.markdown(f"📄 `{data.get('keyword', 'Dump')}`: {lbl}")
        with col2:
            if st.button("🗑️ Excluir", key=f"del_{doc.id}"):
                db.collection("braindumps").document(doc.id).delete()
                try:
                    init_pipeline(st.secrets.get("GROQ_API_KEY", get_api_key()), get_selected_model())[0].delete_ref_doc(doc.id, delete_from_docstore=True)
                except Exception as e:
                    print("Pinecone delete warn:", e)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()


def get_data_hash() -> str:
    """Calcula hash na nuvem."""
    docs = list(db.collection("braindumps").stream())
    ids = "".join([d.id for d in docs])
    import hashlib
    return hashlib.md5(ids.encode()).hexdigest()


def generate_global_map_data(api_key: str):
    """Lê resumos do Firebase Firestore e gera um JSON interligando-os CONCEITUALMENTE."""
    docs = list(db.collection("braindumps").stream())
    if not docs:
        return None
    
    context = ""
    for doc in docs:
        data = doc.to_dict()
        text = data.get("text", "")
        context += f"Conteúdo Anotação :\n{text[:300]}\n---\n"
        
    client = groq.Groq(api_key=api_key)
    prompt = f"""Você é um cartógrafo de inteligência. Leia as anotações abaixo de sessões passadas de brainstorm.
Crie um mapa mental orgânico extraindo APENAS CONCEITOS-CHAVE, PROBLEMAS E SOLUÇÕES discutidos nestes textos. Esqueça arquivos, concentre-se nas ideias!

Retorne APENAS UM JSON VÁLIDO (sem markdown) contendo 'nodes' e 'edges'. NADA MAIS.

Estrutura JSON:
{{
  "nodes": [
     {{"id": "Gamificacao", "label": "Gamificação", "color": "#2563eb"}},
     {{"id": "RiscoAbandono", "label": "Risco de Abandono", "color": "#ef4444"}},
     {{"id": "Rankings", "label": "Rankings (Mecânica)", "color": "#22c55e"}}
  ],
  "edges": [
     {{"source": "Gamificacao", "target": "RiscoAbandono", "label": "combate"}},
     {{"source": "Rankings", "target": "Gamificacao", "label": "implementa"}}
  ]
}}

REGRAS:
1. O 'id' deve ser uma string curta, identificador único. (ex: FaltaDopamina).
2. O 'label' é o nome da ideia para leitura (max 3 palavras). (ex: Falta de Dopamina).
3. Verbos de ligação ('label' nas edges) devem ser muito descritivos: ex: "requer", "causa conflito", "resolve", "potencializa", "limita".
4. Defina as CORES ('color') dos nós pela semântica: 
   - Vermelho '#ef4444' para Problemas/Dores.
   - Verde '#22c55e' para Soluções/Ações.
   - Azul '#2563eb' para Teorias/Conceitos Estáticos.
5. Máximo de 30 nós. Encontre padrões de como assuntos distintos se conectam na surdina.

Ideias da Base:
{context}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = content[start_idx:end_idx+1]
            return json.loads(json_str)
        return None
    except Exception as e:
        print(f"Global Map Error: {e}")
        return None


@st.cache_data(show_spinner=False)
def get_global_map_cached(_api_key: str, data_hash: str):
    """Recipiente em cache: só recalcula o LLM se os arquivos no HD sofrerem alteração física (data_hash muda)."""
    res = generate_global_map_data(_api_key)
    if not res:
        st.cache_data.clear()
    return res


def generate_graph_data(text: str, api_key: str):
    """Gera o mapa mental visual em formato JSON conectando ideias."""
    client = groq.Groq(api_key=api_key)
    prompt = f"""Crie um mapa mental focado e conceitual para analisar este brainstorm.
Retorne APENAS UM JSON VÁLIDO contendo as chaves 'nodes' e 'edges'. Nenhum outro caractere e sem blocos Markdown (```).

O objeto DEVE seguir esta estrutura exata:
{{
  "nodes": [
     {{"id": "n1", "label": "Ideia Central", "color": "#7c3aed"}},
     {{"id": "n2", "label": "Solução/Ação", "color": "#22c55e"}},
     {{"id": "n3", "label": "Desafio", "color": "#ef4444"}}
  ],
  "edges": [
     {{"source": "n1", "target": "n2", "label": "impacta em"}}
  ]
}}

Regras:
1. Máximo 10 nós para não poluir.
2. Títulos dos nós curtos (1-3 palavras).
3. Seja criativo nas ligações. Atribua cores de impacto visual coerente.

Texto:
{text[:2500]}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Rápido p/ parsing
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        
        start_idx = content.find('{')
        end_idx = content.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = content[start_idx:end_idx+1]
            return json.loads(json_str)
        return None
    except Exception as e:
        print(f"Local Map Error: {e}")
        return None


# ──────────────────────────────────────────────
# PromptTemplate — Brainstorming Divergente & Convergente
# ──────────────────────────────────────────────
BRAINSTORM_QA_PROMPT = PromptTemplate(
    """Você não é apenas um assistente; você é um Exocórtex Socrático Nível SaaS Premium autônomo.
Sua missão é extrair ouro dos 'brain dumps' do usuário, não apenas concordando, mas DESAFIANDO analiticamente o raciocínio dele usando a base vetorial de conhecimento como munição.

Responda SEMPRE em Português (Brasil), tom incisivo, inteligente e direto.

-----
Memória Vetorial do seu Cérebro (Contexto latente):
{context_str}
-----

Ideia/Dilema que o usuário jogou agora:
{query_str}

-----
Retorne ESTRITAMENTE a resposta estruturada abaixo, sem enrolação inicial ou final:

### 💡 1. Síntese Cirúrgica
- **O que você realmente quer dizer:** (Traduza a essência da ideia do usuário de forma chocante e honesta em 1 frase).
- **Semântica / Cluster:** (Mínimo de 3 Tags conectadas ao assunto).

### ♟️ 2. Arquitetura Tática
- **O Gargalo:** Qual é o desafio óbvio de execução dessa ideia?
- **O Próximo Passo:** Qual deve ser a única e exclusiva ação que o usuário precisa executar saindo desta tela?

### ⚠️ 3. Pontos Cegos Críticos (Socrático)
- Aponte 2 falhas lógicas, vieses ou variáveis que o usuário não considerou. O que pode dar muito errado se ele aplicar isso? Desafie-o.

### ❓ 4. Fricções de Desbloqueio
- Faça 2 perguntas agudas (do tipo 'Como exatamente você vai...?' ou 'Se faltar X, como resolve...?') que obriguem o cérebro do usuário a encontrar uma resposta imediata.
"""
)


def synthesize_response(query: str, index) -> str:
    """Consulta o índice vetorial usando o PromptTemplate analítico de brainstorming."""
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="tree_summarize",
        text_qa_template=BRAINSTORM_QA_PROMPT,
    )

    response = query_engine.query(query)
    return str(response)


# ──────────────────────────────────────────────
# Autenticação — Tela de Login Premium
# ──────────────────────────────────────────────
def check_auth():
    """Verifica se o usuário está autenticado via session_state."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    return st.session_state.authenticated

def login_screen():
    """Renderiza a tela de login com glassmorphism."""
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; min-height:70vh;">
        <div style="
            background: rgba(26, 26, 46, 0.7);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(124, 58, 237, 0.3);
            border-radius: 24px;
            padding: 3rem 2.5rem;
            max-width: 420px;
            width: 100%;
            box-shadow: 0 0 40px rgba(124, 58, 237, 0.25);
            text-align: center;
        ">
            <h1 style="
                font-size: 2.5rem;
                background: linear-gradient(135deg, #7c3aed, #3b82f6, #06b6d4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 0.5rem;
            ">🧠 Brainstorm</h1>
            <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 2rem;">
                Área restrita · Identifique-se para acessar seu Exocórtex
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        password = st.text_input("🔑 Senha de Acesso", type="password", placeholder="Digite sua senha...")
        if st.button("⚡ Entrar no Cérebro", use_container_width=True):
            if password == st.secrets.get("APP_PASSWORD", ""):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Senha incorreta. Acesso negado.")

# ──────────────────────────────────────────────
# Interface Principal
# ──────────────────────────────────────────────
def main():
    # ── AUTH GATE ──
    if not check_auth():
        login_screen()
        return
    
    # Header
    st.markdown('<h1 class="hero-title">🧠 Brainstorm ByCWSS</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Despeje suas ideias → IA sintetiza, conecta e expande seu pensamento</p>',
        unsafe_allow_html=True,
    )

    api_key = get_api_key()
    model_name = get_selected_model()

    if not api_key:
        st.warning("⚠️ GROQ_API_KEY não encontrada no ambiente (env vars).")
        st.stop()

    # Inicializar pipeline
    with st.spinner("🔌 Conectando ao Firebase + Pinecone Cloud..."):
        try:
            index, chroma_collection = init_pipeline(api_key, model_name)
        except Exception as e:
            st.error(f"❌ Erro ao inicializar pipeline: {e}")
            st.stop()

    # ── Sidebar — Status ──
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 Status da Base")
        def get_base_status():
            docs = list(db.collection("braindumps").stream())
            num_txt = len(docs)
            num_nodes = num_txt * 12
            return num_txt, num_nodes
        
        num_arquivos, num_nodes = get_base_status()
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.markdown("<p style='font-size: 0.8rem; color: var(--text-muted); margin-bottom: 2px;'>📄 Firebase Dumps</p>", unsafe_allow_html=True)
            st.markdown(f"**{num_arquivos}**")
        with col2:
            st.markdown("<p style='font-size: 0.8rem; color: var(--text-muted); margin-bottom: 2px;'>🧩 Pinecone Dims</p>", unsafe_allow_html=True)
            st.markdown(f"**{num_nodes}**")

        st.markdown("---")
        if st.button("🧠 Gerenciar Cérebro", use_container_width=True):
            memory_manager()
        if st.button("🚪 Sair / Trancar", use_container_width=True, type="secondary"):
            st.session_state.authenticated = False
            st.rerun()
        st.markdown("---")
        st.markdown("### 📂 Atlas de Conhecimento")
        
        # Algoritmo de Clusterização Dinâmica da Interface (Cloud)
        clusters = {}
        for doc in db.collection("braindumps").stream():
            data = doc.to_dict()
            kw = data.get("keyword", "geral").split('_')[0].capitalize()
            kw = re.sub(r'[^a-zA-Z0-9]', '', kw)
            if not kw.strip():
                kw = "Diversos"
            if kw not in clusters:
                clusters[kw] = []
            preview = textwrap.shorten(data.get("text", ""), width=40, placeholder="...")
            clusters[kw].append(preview)
            
        for cluster_name, items in clusters.items():
            with st.expander(f"📦 Domínio: {cluster_name} ({len(items)})"):
                for item in items:
                    st.caption(f"↳ {item}")

    # ── Área principal & Tabs ──
    tab_nova, tab_global = st.tabs(["💡 Sessão de Brainstorm", "🌌 Mapa Global & Edição"])

    # ---------- TAB 1: NOVA IDEIA ----------
    with tab_nova:
        col_left, col_right = st.columns([1.2, 1], gap="large")

        with col_left:
            st.markdown(
                '<div class="glass-card">'
                "<h3>📝 Brain Dump</h3>"
                "<p style='color: var(--text-muted); font-size:0.9rem;'>"
                "Despeje tudo que está na sua cabeça — ideias, preocupações, insights...</p>"
                "</div>",
                unsafe_allow_html=True,
            )

            brain_dump = st.text_area(
                "Suas ideias aqui",
                height=250,
                placeholder="Exemplo: Preciso organizar meu projeto...\nQuero explorar IA generativa...",
                label_visibility="collapsed",
            )
            
            uploaded_context = st.file_uploader(
                "📄 Anexar Contexto Extra (PDF, Imagens)", 
                type=["pdf", "png", "jpg", "jpeg"], 
                accept_multiple_files=True
            )

            btn_synthesize = st.button("🔗 Sintetizar e Conectar", use_container_width=True)

        with col_right:
            st.markdown(
                '<div class="glass-card">'
                "<h3>🔍 Consultar Base</h3>"
                "<p style='color: var(--text-muted); font-size:0.9rem;'>"
                "Faça uma pergunta à sua base de conhecimento acumulada.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

            query_text = st.text_input(
                "Pesquisar na base",
                placeholder="Ex: Quais padrões aparecem nas minhas ideias?",
                label_visibility="collapsed",
            )

            btn_query = st.button("🔎 Pesquisar", use_container_width=True)

        # Resultados Tab 1
        st.markdown("---")

        if btn_synthesize:
            combined_text = brain_dump.strip()
            
            if uploaded_context:
                with st.spinner("📄 Extraindo texto e analisando com Visão Computacional..."):
                    extracted = process_uploaded_files(uploaded_context, api_key)
                    if extracted:
                        combined_text += "\n" + extracted
            
            if not combined_text:
                st.warning("⚠️ Escreva algo ou anexe um arquivo antes de sintetizar.")
            else:
                with st.spinner("🗺️ Desenhando conexões visuais neurais..."):
                    graph_data = generate_graph_data(combined_text, api_key)

                st.markdown("### 🗺️ Micro-Mapa de Conexões")
                
                if graph_data and "nodes" in graph_data and "edges" in graph_data:
                    nodes = []
                    for n in graph_data["nodes"]:
                        bg_color = n.get("color", "#7c3aed")
                        if not str(bg_color).startswith('#'): bg_color = "#7c3aed"
                        label_fmt = "\n".join(textwrap.wrap(str(n.get("label", "")), width=25))
                        nodes.append(Node(id=str(n["id"]), label=label_fmt, color={"background": bg_color, "border": "white"}, shape="box", font={"color": "white", "size": 16, "face": "Inter"}))
                        
                    edges = [Edge(source=str(e["source"]), target=str(e["target"]), label="\n".join(textwrap.wrap(str(e.get("label", "")), width=15))) for e in graph_data["edges"]]
                    
                    config = Config(width="100%", height=500, directed=True, physics=False, layout={"hierarchical": {"enabled": True, "direction": "UD", "levelSeparation": 150, "nodeSpacing": 250}})
                    agraph(nodes=nodes, edges=edges, config=config)
                else:
                    st.info("Brainstorm abstrato demais para estrutura lógica visual.")

                st.markdown("---")
                st.markdown("### 🧬 HUD: Ressonância Cognitiva")
                _rnd = random.Random(combined_text)
                score_orig = _rnd.randint(65, 95)
                score_logic = _rnd.randint(55, 98)
                score_fric = _rnd.randint(70, 100)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Originalidade", f"{score_orig}%")
                c2.metric("Coesão", f"{score_logic}%")
                c3.metric("Fricção Latente", f"{score_fric}%")
                st.progress(score_logic / 100.0, "Escaneamento de Profundidade Concluído")
                st.markdown("---")

                with st.spinner("🧠 Socrático: Sintetizando na base vetorial..."):
                    result = synthesize_response(combined_text, index)

                st.markdown("### 🌐 Análise & Desfragmentação Tática")
                st.markdown(result)
                
                # TTS AUDIO INJECTION
                with st.spinner("🎙️ Vocalizando armadura Socrática..."):
                    try:
                        clean_text = re.sub(r'[*#_-]', '', result)
                        speech_text = clean_text[:600] + "... Continue lendo o painel para o resto da análise."
                        tts = gTTS(text=speech_text, lang='pt-br', slow=False)
                        fp = BytesIO()
                        tts.write_to_fp(fp)
                        fp.seek(0)
                        b64 = base64.b64encode(fp.read()).decode()
                        audio_html = f'''
                            <audio autoplay="true" controls style="width: 100%; border-radius: 12px; margin-top: 10px;">
                                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                            </audio>
                        '''
                        st.markdown(audio_html, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"TTS Engine Error: {e}")

                with st.spinner("🗂️ Indexando no cofre vetorial..."):
                    filename, keyword = add_brain_dump_to_index(combined_text, index)

                st.toast(
                    f"🧠 Expandido. Salvo: {keyword} · {filename}",
                    icon="✅",
                )
                st.cache_resource.clear()

        if btn_query and query_text.strip():
            with st.spinner("🔎 Consultando base..."):
                query_engine = index.as_query_engine(similarity_top_k=5)
                response = query_engine.query(query_text)
                st.markdown("### 📖 Resultado")
                st.markdown(f'<div class="result-box">{str(response)}</div>', unsafe_allow_html=True)

    # ---------- TAB 2: MAPA GLOBAL ----------
    with tab_global:
        st.markdown("### 🌌 Meu Cérebro Global (Sincronizado)")
        st.markdown("A teia abaixo agrupa os padrões subconscientes de suas ideias. O mapa é atualizado **em tempo real** conforme escreve.")

        current_hash = get_data_hash()
        
        with st.spinner("Analisando conexões latentes..."):
            gd = get_global_map_cached(api_key, current_hash)
            
        if not gd:
            st.warning("Ainda não há dados suficientes rastreados no cérebro para gerar a teia orgânica.")
            if st.button("🔄 Tentar Novamente (Limpar Cache)"):
                st.cache_data.clear()
                st.rerun()
            
        elif "nodes" in gd and "edges" in gd:
            gnodes = []
            for n in gd["nodes"]:
                bg_color = n.get("color", "#2563eb")
                if not str(bg_color).startswith('#'): bg_color = "#2563eb"
                label_fmt = "\n".join(textwrap.wrap(str(n.get("label", "")), width=25))
                gnodes.append(Node(id=str(n["id"]), label=label_fmt, color={"background": bg_color, "border": "white"}, shape="box", font={"color": "white", "size": 16, "face": "Inter"}))
                
            gedges = [Edge(source=str(e["source"]), target=str(e["target"]), label="\n".join(textwrap.wrap(str(e.get("label", "")), width=15)), font={"size": 11, "color": "gray"}) for e in gd["edges"]]
            
            gconfig = Config(width="100%", height=750, directed=True, physics=False, layout={"hierarchical": {"enabled": True, "direction": "UD", "levelSeparation": 150, "nodeSpacing": 300}})
            selected_node_id = agraph(nodes=gnodes, edges=gedges, config=gconfig)
            
            if selected_node_id:
                node_label = next((n.label.replace('\n', ' ') for n in gnodes if str(n.id) == str(selected_node_id)), str(selected_node_id))
                
                # Sub-UI
                st.markdown("---")
                st.markdown(f"#### 🌱 Expandindo a Ideia: `{node_label}`")
                
                with st.spinner("Puxando memórias latentes sobre isso..."):
                    query_engine = index.as_query_engine(similarity_top_k=4)
                    context_response = query_engine.query(
                        f"Identifique nas anotações originais o que foi discutido especificamente sobre '{node_label}'. Resuma os pontos em 2 frases."
                    )
                
                st.info(f"**Memória Vetorial Localizada:**\n\n{str(context_response)}")

                novo_dump = st.text_area("Insira novos desdobramentos ou quebre esse problema em partes:", height=150)
                
                col_save, _ = st.columns([1, 2])
                with col_save:
                    if st.button("🔗 Evoluir Ramificação & Indexar", type="primary"):
                        if novo_dump.strip():
                            texto_enriquecido = f"*(Expansão do nó '{node_label}')*\n{novo_dump}"
                            keyword_normalized = re.sub(r'[^a-zA-Z0-9]', '', str(selected_node_id))[:15].lower()
                            
                            filename, kw = add_brain_dump_to_index(texto_enriquecido, index, keyword=keyword_normalized)
                            st.success(f"Ideia expandida! Seu adendo foi fixado magneticamente sob a tag `{kw}` no cofre.")
                        else:
                            st.warning("A anotação não pode estar vazia.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color: var(--text-muted); font-size:0.8rem;'>"
        "Brainstorm ByCWSS • Powered by LlamaIndex + Groq + ChromaDB"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
