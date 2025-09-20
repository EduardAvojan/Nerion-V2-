from __future__ import annotations
from typing import List, Dict
import os
from app.parent.driver import ParentLLM

class DeepSeekLocalLLM(ParentLLM):
    """Implementation of ParentLLM using a local DeepSeek model via Ollama.
    This class communicates with the locally installed DeepSeek R1 14B model.
    """
    def __init__(self, model: str = None):
        self.model = model or os.environ.get("NERION_LLM_MODEL", "deepseek-r1:14b")
        
    def complete(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to the local DeepSeek model and return the JSON response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            str: JSON string containing the ParentDecision
        """
        try:
            # Import here to avoid startup dependency
            from langchain_ollama import ChatOllama
            
            # Create Ollama chat instance with the DeepSeek model
            llm = ChatOllama(
                model=self.model,
                temperature=0.1,  # Low temperature for more deterministic planning
                format="json"     # Ensure JSON output
            )
            
            # Convert messages to the format expected by ChatOllama
            ollama_messages = []
            for msg in messages:
                role = msg.get('role', '').lower()
                if role == 'system':
                    ollama_messages.append({"role": "system", "content": msg.get('content', '')})
                elif role == 'user':
                    ollama_messages.append({"role": "human", "content": msg.get('content', '')})
                elif role == 'assistant':
                    ollama_messages.append({"role": "assistant", "content": msg.get('content', '')})
            
            # Get response from DeepSeek model
            response = llm.invoke(ollama_messages)
            
            # Extract and return the content string
            if hasattr(response, 'content'):
                return response.content
            return str(response)
            
        except Exception as e:
            print(f"[DeepSeekLocalLLM] Error: {e}")
            # Return a fallback plan that asks for clarification
            return (
                '{"intent":"error","plan":[{"action":"ask_user","tool":null,'
                '"args":{},"summary":"LLM unavailable; please clarify"}],"final_response":null,'
                '"confidence":0.0,"requires_network":false,"notes":"LLM error occurred"}'
            )
