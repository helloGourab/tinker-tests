import asyncio
from google import genai
from google.genai.types import EmbedContentConfig
from google.genai.types import HttpOptions

# Init client with HTTP options to get token usage
client = genai.Client(http_options=HttpOptions(api_version="v1"))

# Load the dataframe (modify this path or use your own df)
df_10k = pd.read_csv("path_to_your_csv.csv").head(10000)  # only first 10k rows
urls = df_10k['url'].tolist()

# Constants
BATCH_SIZE = 250
MAX_TOKENS_PER_BATCH = 20000

# Helper: get token count for a single input
def get_token_count(text: str) -> int:
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=text
    )
    return response.usage_metadata.prompt_token_count

# Helper: yield batches with token safety
def get_safe_batches(urls):
    batch = []
    total_tokens = 0
    for url in urls:
        token_count = get_token_count(url)
        if (len(batch) >= BATCH_SIZE) or (total_tokens + token_count > MAX_TOKENS_PER_BATCH):
            yield batch
            batch = []
            total_tokens = 0
        batch.append(url)
        total_tokens += token_count
    if batch:
        yield batch

# Async embed one batch
async def embed_batch(batch):
    return client.models.embed_content(
        model="text-embedding-005",
        contents=batch,
        config=EmbedContentConfig(
            task_type="CLASSIFICATION",
            output_dimensionality=768,
        ),
    ).embeddings

# Main embedding loop
async def embed_all(urls):
    all_embeddings = []
    for i, batch in enumerate(get_safe_batches(urls), 1):
        print(f"Embedding batch {i} with {len(batch)} URLs...")
        embeddings = await embed_batch(batch)
        all_embeddings.extend([e.values for e in embeddings])
    return all_embeddings


embeddings = asyncio.run(embed_all(urls))
print(f"Total embedded: {len(embeddings)}")
