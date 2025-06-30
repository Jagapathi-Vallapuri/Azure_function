from huggingface_hub import InferenceClient
import os
import logging

# In Azure Functions, environment variables are loaded from Application Settings
HF_API_TOKEN_1 = os.getenv("HF_API_TOKEN_1")
HF_API_TOKEN_2 = os.getenv("HF_API_TOKEN_2")  # Optional, for redundancy

MODEL_ID = "NeuML/pubmedbert-base-embeddings"

def get_text_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a list of text strings using the Hugging Face Inference Client.
    Tries multiple API tokens if available.
    """
    api_tokens = [HF_API_TOKEN_1]
    if HF_API_TOKEN_2:
        api_tokens.append(HF_API_TOKEN_2)

    for token in api_tokens:
        if not token:
            logging.warning("Hugging Face API token not found. Skipping embedding generation.")
            continue

        try:
            # Initialize the Hugging Face Inference Client with the current token
            client = InferenceClient(api_key=token)

            # Generate embeddings for the input texts
            embeddings = []
            for text in texts:
                embedding = client.feature_extraction(model=MODEL_ID, text=text)
                embeddings.append(embedding.tolist())
            return embeddings
        except Exception as e:
            logging.error(f"Error generating embeddings with token ending in ...{token[-5:]}: {e}")
            continue

    raise Exception("Failed to generate embeddings after trying all available Hugging Face API tokens.")