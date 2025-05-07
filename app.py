import streamlit as st
from st_audiorec import st_audiorec 
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import google.generativeai as genai
from PIL import Image
import io
import speech_recognition as sr
from gtts import gTTS
import base64
import time
import os
import av # PyAV para processamento de frames com streamlit-webrtc

# --- Configurações Iniciais ---
st.set_page_config(page_title="Assistente Interativo com IA", layout="wide")

# Carregar a API Key do Google
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except KeyError:
    st.error("Chave da API do Google não encontrada. Configure-a em .streamlit/secrets.toml")
    st.stop()
except Exception as e:
    st.error(f"Erro ao configurar a API do Google: {e}")
    st.stop()

# Inicializar o modelo Gemini
# Para multimodal (texto e imagem)
model_vision = genai.GenerativeModel('gemini-pro-vision')
# Para chat/texto apenas
model_text = genai.GenerativeModel('gemini-pro')

# --- Estado da Sessão ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_image_bytes' not in st.session_state:
    st.session_state.current_image_bytes = None
if 'audio_input_text' not in st.session_state:
    st.session_state.audio_input_text = ""
if 'processing_audio' not in st.session_state:
    st.session_state.processing_audio = False


# --- Funções Auxiliares ---
def autoplay_audio_from_bytes(audio_bytes, format="mp3"):
    """Reproduz áudio a partir de bytes no Streamlit."""
    b64 = base64.b64encode(audio_bytes).decode()
    md = f"""
        <audio autoplay="true">
        <source src="data:audio/{format};base64,{b64}" type="audio/{format}">
        Seu navegador não suporta o elemento de áudio.
        </audio>
        """
    st.markdown(md, unsafe_allow_html=True)

def text_to_speech(text):
    """Converte texto em áudio (bytes) usando gTTS."""
    try:
        tts = gTTS(text=text, lang='pt-br') # Linguagem português do Brasil
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp.read()
    except Exception as e:
        st.error(f"Erro no Text-to-Speech: {e}")
        return None

def recognize_speech_from_mic(recognizer, microphone):
    """Captura e reconhece fala do microfone."""
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")
    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    with microphone as source:
        st.info("Ajustando para o ruído ambiente... Fale em instantes.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        st.info("Ouvindo... Fale agora!")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10) # Timeout de 5s se não houver fala, limite de 10s por frase
        except sr.WaitTimeoutError:
            st.warning("Nenhuma fala detectada em 5 segundos.")
            return None, "Nenhuma fala detectada."

    st.info("Processando sua fala...")
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }
    try:
        response["transcription"] = recognizer.recognize_google(audio, language="pt-BR")
    except sr.RequestError as e:
        response["success"] = False
        response["error"] = f"API indisponível/erro de conexão; {e}"
    except sr.UnknownValueError:
        response["error"] = "Não foi possível entender o áudio"

    return response, None if response["success"] else response["error"]


# --- Componente da Câmera ---
class VisionTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_bytes = None

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_image() # Converte para PIL Image
        st.session_state.current_image_bytes = self._image_to_bytes(img)
        return frame # Retorna o frame original para exibição

    def _image_to_bytes(self, image: Image.Image, format="JPEG"):
        buffered = io.BytesIO()
        image.save(buffered, format=format)
        return buffered.getvalue()

# Configuração RTC para tentar forçar o envio de frames (pode não ser necessário com transform)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- Interface do Usuário (Layout) ---
st.title("🤖 Assistente Interativo com IA (Visão e Voz)")
st.markdown("Use sua câmera e microfone para interagir com a IA.")

col1, col2 = st.columns(2)

with col1:
    st.header("👁️ Visão da IA")
    webrtc_ctx = webrtc_streamer(
        key="camera",
        mode=WebRtcMode.SENDRECV, # Precisa receber para exibir, envia para processar
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=VisionTransformer,
        media_stream_constraints={"video": True, "audio": False}, # Apenas vídeo aqui
        async_processing=True,
    )
    if webrtc_ctx.state.playing:
        st.info("Câmera ativa. A imagem atual está sendo capturada.")
    else:
        st.info("Clique em 'Start' para ativar a câmera.")

    if st.session_state.current_image_bytes:
        st.image(st.session_state.current_image_bytes, caption="Imagem Capturada Recentemente", use_column_width=True)

with col2:
    st.header("💬 Conversa com a IA")

    # Exibir histórico do chat
    chat_container = st.container(height=400) # Usando um container com altura fixa
    with chat_container:
        for entry in st.session_state.chat_history:
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])
                if "audio" in entry and entry["audio"]:
                    st.audio(entry["audio"], format="audio/mp3")

    # Entrada de áudio
    st.subheader("🎤 Fale com a IA")
    if not st.session_state.processing_audio:
        if st.button("Clique para Falar (e aguarde)"):
            st.session_state.processing_audio = True
            st.rerun() # Força o rerun para entrar no bloco de processamento

    if st.session_state.processing_audio:
        with st.spinner("Gravando e Processando Áudio..."):
            recognizer = sr.Recognizer()
            microphone = sr.Microphone()
            response, error_msg = recognize_speech_from_mic(recognizer, microphone)

            if error_msg:
                st.warning(f"Erro ao gravar/reconhecer áudio: {error_msg}")
                st.session_state.audio_input_text = ""
            elif response and response["transcription"]:
                st.session_state.audio_input_text = response["transcription"]
                st.success(f"Você disse: {st.session_state.audio_input_text}")
            else:
                st.session_state.audio_input_text = "" # Limpa se não houve transcrição válida

            st.session_state.processing_audio = False # Reseta o estado
            # Não precisa de rerun aqui, o fluxo continua

    # Campo para prompt de texto (seja digitado ou preenchido pelo áudio)
    prompt_text = st.text_input("Ou digite sua pergunta/comando aqui:", value=st.session_state.audio_input_text, key="text_prompt_input")

    if st.button("Enviar para IA", key="send_to_ia"):
        if not prompt_text:
            st.warning("Por favor, forneça uma pergunta ou comando.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt_text})

            with st.spinner("IA pensando..."):
                try:
                    # Prepara as partes da mensagem para o modelo multimodal
                    message_parts = [prompt_text]
                    image_for_model = None

                    if st.session_state.current_image_bytes:
                        try:
                            img_pil = Image.open(io.BytesIO(st.session_state.current_image_bytes))
                            # O SDK espera um objeto PIL Image diretamente para 'gemini-pro-vision'
                            image_for_model = img_pil
                            message_parts.append(image_for_model) # Adiciona a imagem se disponível
                            st.info("Enviando texto e imagem para a IA.")
                        except Exception as e:
                            st.warning(f"Não foi possível processar a imagem capturada: {e}")
                            st.info("Enviando apenas texto para a IA.")
                    else:
                        st.info("Nenhuma imagem da câmera para enviar. Enviando apenas texto.")

                    # Se tiver imagem, usa o modelo vision, senão, o modelo text
                    if image_for_model:
                        # O modelo vision espera a imagem como parte da lista de conteúdos
                        # E o texto deve ser o primeiro elemento da lista
                        # Ex: response = model_vision.generate_content([prompt_text, image_pil_object])
                        full_prompt = [prompt_text, image_for_model] if image_for_model else [prompt_text]
                        response_ia = model_vision.generate_content(full_prompt, stream=True)
                        response_ia.resolve() # Resolve o stream para obter o texto completo
                    else:
                        # Para o modelo de texto, podemos usar o histórico para manter o contexto
                        # No entanto, para simplificar a lógica de imagem, vamos fazer chamadas únicas por enquanto.
                        # Para um chatbot de texto contínuo, você faria:
                        # chat_session = model_text.start_chat(history=...)
                        # response_ia = chat_session.send_message(prompt_text, stream=True)
                        # response_ia.resolve()
                        response_ia = model_text.generate_content(prompt_text, stream=True)
                        response_ia.resolve()

                    ai_text_response = response_ia.text
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_text_response})

                    # Converter resposta da IA para áudio e adicionar ao histórico
                    audio_response_bytes = text_to_speech(ai_text_response)
                    if audio_response_bytes:
                        st.session_state.chat_history[-1]["audio"] = audio_response_bytes
                        autoplay_audio_from_bytes(audio_response_bytes)

                    # Limpar o campo de prompt de texto e o texto do áudio
                    st.session_state.audio_input_text = ""
                    # Precisamos de uma forma de limpar o text_input. Usar st.empty() ou um novo key pode funcionar.
                    # A maneira mais simples é forçar um rerun após o processamento.
                    st.rerun()

                except Exception as e:
                    st.error(f"Erro ao contatar a IA: {e}")
                    # Adicionar mensagem de erro ao chat para feedback
                    st.session_state.chat_history.append({"role": "assistant", "content": f"Desculpe, ocorreu um erro: {e}"})


# --- Rodapé ---
st.markdown("---")
st.markdown("Desenvolvido como um exemplo de integração multimodal com Streamlit e Google AI.")
