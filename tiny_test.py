#!/usr/bin/env python3
import httpx
response = httpx.get("http://127.0.0.1:8000/api/v1/dataset")
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
