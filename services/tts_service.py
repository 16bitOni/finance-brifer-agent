from google.cloud import texttospeech
import base64
import os
import logging

logger = logging.getLogger(__name__)

class TTSService:
    """Service for text-to-speech conversion using Google Cloud TTS."""
    
    def __init__(self):
        """Initialize the TTS service."""
        try:
            self.client = texttospeech.TextToSpeechClient()
            logger.info("Successfully initialized Google Cloud TTS client")
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