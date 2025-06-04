import streamlit as st
import time
from datetime import datetime
import os
from dotenv import load_dotenv
import asyncio
import tempfile
import base64
import uuid

from workflows.workflow import process_query
from state.state import Query
from tools.embedding_service import initialize_mock_data, embed_streamlit_data
from services.tts_service import tts_service
from services.stt_service import stt_service

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="Financial Assistant",
    page_icon="💰",
    layout="wide"
)

def autoplay_audio(audio_base64: str, audio_id: str):
    """Create an HTML audio player with autoplay and stop functionality."""
    md = f"""
        <div id="audio-container-{audio_id}">
            <audio id="audio-{audio_id}" autoplay="true">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            <button onclick="document.getElementById('audio-{audio_id}').pause(); document.getElementById('audio-{audio_id}').currentTime = 0;">Stop Speech</button>
        </div>
        <script>
            // Stop all other audio players when this one starts
            document.getElementById('audio-{audio_id}').addEventListener('play', function() {{
                const allAudios = document.getElementsByTagName('audio');
                for(let audio of allAudios) {{
                    if(audio.id !== 'audio-{audio_id}') {{
                        audio.pause();
                        audio.currentTime = 0;
                    }}
                }}
            }});
        </script>
        """
    st.markdown(md, unsafe_allow_html=True)

def main():
    # Sidebar configuration
    st.sidebar.title("Settings")
    
    # Enable/disable voice
    enable_voice = st.sidebar.checkbox("Enable Voice Output", value=True)
    
    # Enable/disable speech input
    enable_speech_input = st.sidebar.button("🎤 Toggle Speech Input")
    
    # Initialize speech input state
    if "speech_input_enabled" not in st.session_state:
        st.session_state.speech_input_enabled = False
    
    # Toggle speech input
    if enable_speech_input:
        st.session_state.speech_input_enabled = not st.session_state.speech_input_enabled
    
    # Show speech input status
    st.sidebar.write(f"Speech Input: {'Enabled' if st.session_state.speech_input_enabled else 'Disabled'}")
    
    # Data Management Section
    st.sidebar.title("Data Management")
    
    # Initialize mock data
    if st.sidebar.button("Initialize Mock Data"):
        with st.spinner("Initializing mock data..."):
            result = initialize_mock_data()
            st.sidebar.success(result)
    
    # File upload
    st.sidebar.title("Upload Portfolio Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=['csv', 'json', 'txt', 'md'],
        help="Upload your portfolio data in CSV, JSON, or text format"
    )
    
    if uploaded_file:
        content_type = st.sidebar.selectbox(
            "Content Type",
            ["portfolio", "news", "research", "other"],
            help="Select the type of content being uploaded"
        )
        
        if st.sidebar.button("Embed Uploaded Data"):
            with st.spinner("Embedding data..."):
                result = embed_streamlit_data(uploaded_file, content_type)
                st.sidebar.success(result)
    
    # Main chat interface
    st.title("🤖 Financial Assistant")
    st.markdown("Ask me anything about financial news, your portfolio, or general financial concepts!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and enable_voice and "audio_id" in message:
                try:
                    # Convert assistant's response to speech
                    audio_base64 = tts_service.text_to_speech(message["content"])
                    autoplay_audio(audio_base64, message["audio_id"])
                except Exception as e:
                    st.error(f"Error generating speech: {str(e)}")
    
    # Speech input
    if st.session_state.speech_input_enabled:
        st.markdown("### 🎤 Voice Input")
        audio_bytes = st.audio_input(
            label="Click to record your question",
            key="voice_input"
        )
        
        if audio_bytes:
            # Save the audio bytes to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_audio.write(audio_bytes.getbuffer())
                temp_audio_path = temp_audio.name
            
            try:
                with st.spinner("Transcribing your speech..."):
                    # Use Google Cloud Speech-to-Text for transcription
                    if stt_service.speech_client:
                        with open(temp_audio_path, 'rb') as audio_file:
                            transcribed_text = stt_service.transcribe_audio(audio_file)
                    else:
                        st.warning("Speech-to-Text service not available. Please check your Google Cloud credentials.")
                        transcribed_text = None
                    
                    if transcribed_text:
                        st.text_area("Transcribed text:", transcribed_text, height=100)
                        # Process the transcribed text automatically
                        prompt = transcribed_text
                        # Process the transcribed text as a normal prompt
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        
                        # Process query
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                query = Query(
                                    text=prompt,
                                    user_id="user_1",
                                    timestamp=datetime.now().timestamp()
                                )
                                
                                result = asyncio.run(process_query(query.text, query.user_id))
                                response_text = result.get("final_result", "I apologize, but I couldn't process your request.")
                                st.markdown(response_text)
                                
                                if enable_voice and response_text:
                                    try:
                                        audio_base64 = tts_service.text_to_speech(str(response_text))
                                        audio_id = str(uuid.uuid4())
                                        autoplay_audio(audio_base64, audio_id)
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": response_text,
                                            "audio_id": audio_id
                                        })
                                    except Exception as e:
                                        st.error(f"Error generating speech: {str(e)}")
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": response_text
                                        })
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
    
    # Text input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Create query object
                query = Query(
                    text=prompt,
                    user_id="user_1",
                    timestamp=datetime.now().timestamp()
                )
                
                # Process query through workflow
                result = asyncio.run(process_query(query.text, query.user_id))
                
                # Display response
                response_text = result.get("final_result", "I apologize, but I couldn't process your request.")
                st.markdown(response_text)
                
                # Generate and play voice if enabled
                if enable_voice and response_text:
                    try:
                        audio_base64 = tts_service.text_to_speech(str(response_text))
                        audio_id = str(uuid.uuid4())
                        autoplay_audio(audio_base64, audio_id)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "audio_id": audio_id
                        })
                    except Exception as e:
                        st.error(f"Error generating speech: {str(e)}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text
                        })

if __name__ == "__main__":
    main() 