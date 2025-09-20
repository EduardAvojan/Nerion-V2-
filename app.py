#!/usr/bin/env python
"""
Main entry point for Nerion.
Launches the chat interface with voice or text-only mode.
"""
import argparse
from app.nerion_chat import main
from app.chat.state import ChatState, VoiceSettings

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Nerion AI Assistant')
    parser.add_argument('--text-only', action='store_true', help='Disable speech input, use text input only')
    parser.add_argument('--no-speech-output', action='store_true', help='Disable speech output')
    args = parser.parse_args()
    
    # Inject a custom state if needed
    if args.text_only or args.no_speech_output:
        state = ChatState()
        if args.text_only:
            # Configure for text input only
            print("Speech input disabled, using text input only.")
            state.voice = VoiceSettings(enabled=False)  # Disable speech input
        
        if args.no_speech_output:
            # Disable speech output
            print("Speech output disabled.")
            state.set_mute(True)  # Mute speech output
        
        # Run with custom state
        main(custom_state=state)
    else:
        # Run with default settings
        main()
