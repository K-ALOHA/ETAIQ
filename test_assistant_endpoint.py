#!/usr/bin/env python3
import httpx
import json

url = "http://127.0.0.1:8000/api/v1/assistant/chat"
payload = {
    "message": "Summarize dataset",
    "conversation_id": "test-convo-001"
}
headers = {
    "Content-Type": "application/json"
}

response = httpx.post(url, json=payload, headers=headers)
print(f"Status code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
