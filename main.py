import os
import time 
from dotenv import load_dotenv
from google import genai
from google.genai.types import EmbedContentConfig

load_dotenv()

client = genai.Client()

start_time = time.time()

response = client.models.embed_content(
    model="text-embedding-005",
    contents=["text to be embedded", "idk how is this working", "hello i am you"],
    config=EmbedContentConfig(
        task_type="CLASSIFICATION",
        output_dimensionality=768,
    ),
)

end_time = time.time()
print(f"Time taken: {end_time - start_time:.4f} seconds")
print(response.embeddings[1].values)
