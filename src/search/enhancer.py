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
                    {"role": "system", "content": "You are a music expert. Analyze the user's query to extract search terms, mood, and genre. Return ONLY valid JSON."},
                    {"role": "user", "content": 'Analyze: "sad midnight drive". Return JSON structure.'},
                    {"role": "assistant", "content": '{"keywords": ["night", "drive", "lonely"], "mood": "Melancholic", "genre": "Synthwave"}'},
                    {"role": "user", "content": f'Analyze: "{user_query}". Return JSON structure.'}
                ],

                "max_tokens": 1024
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                msg = result['choices'][0]['message']
                
                # GLM-4.7 puts reasoning in 'reasoning_content' and final answer in 'content'
                # If 'content' is empty, it might be in 'reasoning_content' or truncated
                content = msg.get('content', '')
                reasoning = msg.get('reasoning_content', '')
                
                # If content is empty/short but reasoning has JSON, use reasoning
                target = content if len(content) > 10 else (reasoning if len(reasoning) > 10 else content)
                
                # Debug logging
                try:
                    with open('glm_debug.log', 'w', encoding='utf-8') as f:
                        f.write(f"--- CONTENT ---\n{content}\n\n--- REASONING ---\n{reasoning}")
                except:
                    pass

                # Extract JSON from Markdown code blocks if present
                clean_content = target.strip()
                if '```' in clean_content:
                    parts = clean_content.split('```')
                    for p in parts:
                        if p.strip().startswith('json'):
                            clean_content = p.strip()[4:].strip()
                            break
                        elif p.strip().startswith('{'):
                            clean_content = p.strip()
                            break
                
                # Find first { and last }
                if '{' in clean_content and '}' in clean_content:
                    start = clean_content.find('{')
                    end = clean_content.rfind('}') + 1
                    clean_content = clean_content[start:end]
                
                return json.loads(clean_content)
                
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