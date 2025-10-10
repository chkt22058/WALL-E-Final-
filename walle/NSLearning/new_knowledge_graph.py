import os
import openai
import json
import time
import copy
import re
import inspect
import textwrap
from typing import List, Callable, Tuple, Dict, Optional

# OpenAI APIキーの設定 (環境変数から取得することを推奨)
try:
    client = openai.OpenAI()
    print("[Global] OpenAI client initialized successfully.")
except Exception as e:
    print(f"[Global] Error initializing OpenAI client: {e}")
    print("Please ensure your OPENAI_API_KEY environment variable is set correctly.")
    client = None 

class KnowledgeGraph:

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.client = client 

        if self.client is None:
            print("[KG] Warning: OpenAI client is not initialized. LLM calls will fail.")

    def generate_KnowledgeGraph(self, transitions_data: list):
       
        if self.client is None:
            print("[KG] Error: OpenAI client is not available. Cannot generate action.")

        # 変換された遷移データをJSON文字列に変換（インデント付きで可読性高く）
        transitions_json_str = json.dumps(transitions_data, indent=4, ensure_ascii=False)
            
        # プロンプト ===================================================================
        system_prompt = textwrap.dedent("""
    You are a helpful assistant with inductive reasoning. Given the history trajectory,
    including action and observation, you need to reflect on the action execution results
    and identify and extract prerequisite or feasibility constraints, that is, discover
    when an action or item creation requires the presence of certain materials, resources, or other items.

    We define the Knowledge Graph as:
    {
        "V": "the set of entities (e.g., items, materials, location-specific objects, or abstract concepts)",
        "E": "the set of directed edges, each capturing a relationship or prerequisite among entities"
    }

    An edge takes the form:
    (u, v, label),
    where u and v are entities in V, and label indicates how u relates to v
    (for example, 'requires', 'consumes', 'collects', etc.).

    You should ONLY respond in the following format:
    [
    {'u':'entity_u', 'v':'entity_v', 'label':{'relation':'...', 'quantity':'...'}},
    {'u':'entity_u', 'v':'entity_v', 'label':{'relation':'...', 'quantity':'...'}},
    ...
    ]
    example:
    [
    {'u':'wooden_sword', 'v':'table', 'label':{'relation':'requires', 'quantity':None}},
    {'u':'table', 'v':'wood', 'label':{'relation':'consumes', 'quantity':'2'}}
    ]
    Instructions:
    - Ensure the response can be parsed by Python 'json.loads', e.g.: no trailing commas, **no single quotes**, etc.
        """)
        
        header = textwrap.dedent("""
            I will give you an array of transitions:
        """)


        user_prompt = header + transitions_json_str

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # プロンプトのログ残す用
        # ========================================================
        log_text = "--- System Prompt ---\n"
        log_text += system_prompt + "\n\n"
        
        for msg in messages[1:]:
            log_text += f"--- {msg['role'].capitalize()} ---\n"
            log_text += msg['content'] + "\n\n"
        # ========================================================

        with open("./CodeRule/input/KG_llm_prompt.txt", "w", encoding="utf-8") as f:
            f.write(log_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=2000,
                temperature=0,
            )

            generate_response = response.choices[0].message.content
            knowledge_graph_data = json.loads(generate_response) 
            return knowledge_graph_data

        except openai.APITimeoutError:
            print("[KG] API request timed out.")
          
        except openai.APIConnectionError as e:
            print(f"[KG] API connection error: {e}")

        except openai.RateLimitError:
            print("[KG] OpenAI API rate limit exceeded. Waiting 5 seconds...")
            time.sleep(5)
            return self.generate_KnowledgeGraph(transitions_data)

        except openai.APIStatusError as e:
            print(f"[KG] OpenAI API status error: {e.status_code} - {e.response}")

        except json.JSONDecodeError:
            print(f"[KG] JSON decode error from LLM response: {content}")
    
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    

    def save(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    