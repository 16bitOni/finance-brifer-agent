import os
from google.cloud import speech
import tempfile
import json
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class STTService:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        self.speech_client = None
        self._initialize()

    def _initialize(self):
        """Initialize Google Cloud Speech-to-Text service."""
        # Check for GOOGLE_APPLICATION_CREDENTIALS (file path)
        credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if credentials_json:
            try:
                # Validate JSON content before writing
                json.loads(credentials_json)  # Check if valid JSON
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_credentials:
                    temp_credentials.write(credentials_json)
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_credentials.name
                self.speech_client = speech.SpeechClient()
                logger.info("Google Cloud Speech-to-Text initialized using JSON content.")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in GOOGLE_APPLICATION_CREDENTIALS_JSON: {str(e)}")
                self.speech_client = None
            except Exception as e:
                logger.error(f"Error initializing Google Cloud Speech-to-Text: {str(e)}")
                self.speech_client = None
            finally:
                # Clean up temp file
                if 'temp_credentials' in locals() and os.path.exists(temp_credentials.name):
                    os.unlink(temp_credentials.name)
        else:
            logger.warning("Google Cloud credentials not found. Speech-to-text functionality will be disabled.")
            self.speech_client = None

    def transcribe_audio(self, audio_file):
        """Transcribe audio using Google Cloud Speech-to-Text."""
        if not self.speech_client:
            logger.warning("Speech client not initialized. Cannot transcribe audio.")
            return None
        
        try:
            # Read the audio file
            content = audio_file.read()
            
            # Configure the audio settings
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=48000,  # Updated to match st.audio_input's sample rate
                language_code="en-US",
                enable_automatic_punctuation=True
            )
            
            # Perform the transcription
            response = self.speech_client.recognize(config=config, audio=audio)
            
            # Combine all transcriptions
            transcript = ""
            for result in response.results:
                transcript += result.alternatives[0].transcript + " "
            
            logger.info("Successfully transcribed audio")
            return transcript.strip()
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return None

# Create singleton instance
stt_service = STTService()