## Hey can anyone please verify if i am doing something wrong or the vertex api is actually that slow

- ok so i locally tested both the api i am using with sample small data
- here is the embedding api use and output

```python
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
```

```python
    # output
    Time taken: 12.6191 seconds
    ...embedding vector
```

- and here is the token count api

```python
    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    prompt = "Why is the sky blue?"

    strt_time = time.time()
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    )
    end_time = time.time()

    print(f"Time taken: {end_time - strt_time:.4f} seconds")
    print(response.usage_metadata.prompt_token_count)
```

```python
    # output
    Time taken: 3.5657 seconds
    6
```

- but when i use these combined to get some embeddings on this data:

```python
    "http://www.saman-sna.cf"
    "https://libreriabaruqeros45.libreria15.repl.co"
    "https://www.endometriosis-uk.org"
    "https://www.mca.org.mt"
    "http://www.xgdsgj.com" # ...only 100 rows of this kinda strings for testing

```

- using this code on google colab

```python
    import asyncio
    from google import genai
    from google.genai.types import EmbedContentConfig
    from google.genai.types import HttpOptions

    # Init client with HTTP options to get token usage
    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    # Load the dataframe (modify this path or use your own df)
    df_100_rows = pd.read_csv("data.csv").head(100)  # only first 100 rows
    urls = df_100_rows['url'].tolist()

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
```

- more than 15 mins passed and no output yet ... so i terminated it but i wonder how is the `vertex` ai of google which is pretty much their enterprise ai solution be so slow even for only `100 small strings` ... so does that mean i am doing something wrong here ?
