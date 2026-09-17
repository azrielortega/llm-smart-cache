import os

from core.cache_logic import SmartCache

from dotenv import load_dotenv

from openai import OpenAI

def get_response(user_input):
    cached_answer = cache.query(user_input)
    if cached_answer:
        return cached_answer

    print("Calling LLM API")
    new_answer = f"AI response to: {user_input}" 
    
    cache.update(user_input, new_answer)
    return new_answer


load_dotenv()

AI_KEY = os.getenv("OPENROUTER_KEY")

if not AI_KEY:
    raise ValueError("API KEY Not founnd!")

client = OpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key=AI_KEY)



# cache = SmartCache(max_distance=0.3)

# print(get_response("How to bake a cake?")) # Miss
# print(get_response("How do I bake a cake?")) # HIT (Semantic similarity works!)