import os 
import time
from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions

load_dotenv()

client = genai.Client(http_options=HttpOptions(api_version="v1"))

prompt = "Why is the sky blue?"

strt_time = time.time()
response = client.models.generate_content(
    model="gemini-2.0-flash", contents=prompt
)
end_time = time.time()

print(f"Time taken: {end_time - strt_time:.4f} seconds")
print(response.usage_metadata.prompt_token_count)