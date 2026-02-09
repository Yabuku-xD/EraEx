import os
import requests
import json
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

class SemanticEnhancer:
    def __init__(self):
        self.api_key = os.getenv('GLM_API_KEY')
        self.base_url = "https://api.z.ai/api/coding/paas/v4/chat/completions"
        self.model = "glm-4.7"
        self.enabled = bool(self.api_key)
        
        if self.enabled:
            print("Semantic Enhancer: Online (GLM-4.7)")
        else:
            print("Semantic Enhancer: Offline (no API key)")
    
    def analyze_query(self, user_query: str) -> Dict:
        if not self.enabled:
            return {'keywords': [user_query], 'mood': None, 'genre': None}
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a music search assistant. Return only valid JSON."},
                    {"role": "user", "content": f'Analyze: "{user_query}". Return JSON: {{"keywords": ["terms"], "mood": "tone or null", "genre": "genre or null"}}'}
                ],
                "max_tokens": 200
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                msg = result['choices'][0]['message']
                content = msg.get('content') or msg.get('reasoning_content', '')
                content = content.strip()
                
                if '{' in content and '}' in content:
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    content = content[start:end]
                
                if '```' in content:
                    parts = content.split('```')
                    content = parts[1] if len(parts) > 1 else parts[0]
                    if content.startswith('json'):
                        content = content[4:].strip()
                
                return json.loads(content)
            else:
                print(f"GLM {response.status_code}")
                return {'keywords': [user_query], 'mood': None, 'genre': None}
                
        except Exception as e:
            print(f"GLM fallback: {e}")
            return {'keywords': [user_query], 'mood': None, 'genre': None}
    
    def get_enhanced_query(self, user_query: str) -> str:
        result = self.analyze_query(user_query)
        parts = result.get('keywords', [user_query])
        if result.get('mood'):
            parts.append(result['mood'])
        if result.get('genre'):
            parts.append(result['genre'])
        return ' '.join(str(p) for p in parts)