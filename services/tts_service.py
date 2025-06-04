from google.cloud import texttospeech
import base64
import os
import logging
from google.oauth2 import service_account
import json
import tempfile
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class TTSService:
    """Service for text-to-speech conversion using Google Cloud TTS."""
    
    def __init__(self):
        """Initialize the TTS service."""
        try:
            # Load environment variables from .env file
            load_dotenv()
            
            # Get credentials from environment variable
            credentials_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
            if not credentials_json:
                raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable not set")
            
            try:
                # Clean the JSON string
                credentials_json = credentials_json.strip()
                
                # Log the problematic area
                try:
                    json.loads(credentials_json)
                except json.JSONDecodeError as e:
                    # Get the context around the error
                    start = max(0, e.pos - 50)
                    end = min(len(credentials_json), e.pos + 50)
                    context = credentials_json[start:end]
                    logger.error(f"JSON Error at position {e.pos}:")
                    logger.error(f"Context: ...{context}...")
                    logger.error(f"Error: {str(e)}")
                    
                    # Try to identify common issues
                    if "Expecting property name" in str(e):
                        logger.error("This usually means there's a missing quote or comma in your JSON")
                        logger.error("Check for:")
                        logger.error("1. Missing quotes around property names")
                        logger.error("2. Single quotes instead of double quotes")
                        logger.error("3. Missing commas between properties")
                        logger.error("4. Extra commas at the end of objects")
                    raise
                
                # Create a temporary file with the credentials
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_credentials:
                    temp_credentials.write(credentials_json)
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_credentials.name
                
                # Initialize client with default credentials
                self.client = texttospeech.TextToSpeechClient()
                logger.info("Successfully initialized Google Cloud TTS client")
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in GOOGLE_APPLICATION_CREDENTIALS_JSON: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Error initializing TTS client: {str(e)}")
                raise
            finally:
                # Clean up temp file
                if 'temp_credentials' in locals() and os.path.exists(temp_credentials.name):
                    os.unlink(temp_credentials.name)
                    
        except Exception as e:
            logger.error(f"Error initializing TTS client: {e}")
            raise
    
    def text_to_speech(self, text: str) -> str:
        """
        Convert text to speech using Google Cloud TTS.
        
        Args:
            text (str): The text to convert to speech
            
        Returns:
            str: Base64 encoded audio data
        """
        try:
            # Set the text input to be synthesized
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Build the voice request
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Neural2-F",  # Using a neural voice for better quality
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
            
            # Select the type of audio file
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,  # Normal speed
                pitch=0.0  # Default pitch
            )
            
            # Perform the text-to-speech request
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # Convert the audio content to base64
            audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')
            
            logger.info("Successfully converted text to speech")
            return audio_base64
            
        except Exception as e:
            logger.error(f"Error converting text to speech: {e}")
            raise

# Create singleton instance
tts_service = TTSService() 