#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import traceback

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

# Import the assistant class
try:
    from app.ai.assistant import ETAIQAssistantService
    from app.ai.schemas import AssistantRequest
    
    # Create instance
    assistant = ETAIQAssistantService()
    
    # Test dataset summary
    request = AssistantRequest(message="Summarize dataset", conversation_id="test-convo-1")
    response = assistant.handle_message(request)
    
    print("AI Assistant response:")
    print(response.response)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    print(traceback.format_exc())
