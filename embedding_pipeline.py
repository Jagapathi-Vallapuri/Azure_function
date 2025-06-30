import os
import json
import csv
import chromadb
from chromadb.config import Settings
from embedding_utils import get_text_embeddings

def process_embeddings(extracted_data_root, chroma_persist_dir):
    """
    For each PDF folder in extracted_data_root:
      - Reads text chunks from <pdf>_chunks.json
      - Reads image captions from images/<pdf>_captions.csv
      - Generates embeddings for each text chunk and image caption
      - Stores results in persistent ChromaDB collections in file share
    """
    # Set up persistent ChromaDB client
    client = chromadb.Client(Settings(persist_directory=chroma_persist_dir))
    text_collection = client.get_or_create_collection("textEmbeddings")
    image_collection = client.get_or_create_collection("imageEmbeddings")

    for folder in os.listdir(extracted_data_root):
        folder_path = os.path.join(extracted_data_root, folder)
        if not os.path.isdir(folder_path):
            continue
        # Text chunks
        text_chunks_path = os.path.join(folder_path, f"{folder}_chunks.json")
        if os.path.exists(text_chunks_path):
            with open(text_chunks_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            ids = [f"{folder}_chunk_{idx+1}" for idx in range(len(chunks))]
            metadatas = [{"pdf_id": str(folder)} for _ in chunks]
            if chunks:
                embeddings = get_text_embeddings(chunks)
                text_collection.add(
                    ids=ids,
                    embeddings=embeddings, # type: ignore[reportArgumentType]
                    metadatas=metadatas, # type: ignore[reportArgumentType]
                    documents=chunks  # type: ignore[reportArgumentType]
                )
        # Image captions
        image_captions_path = os.path.join(folder_path, "images", f"{folder}_captions.csv")
        if os.path.exists(image_captions_path):
            with open(image_captions_path, 'r', encoding='utf-8') as f:
                reader = list(csv.DictReader(f))
            if reader:
                captions = [row["caption"] for row in reader]
                ids = [f"{folder}_{row['image_name']}" for row in reader]
                metadatas = [{"pdf_id": str(folder)} for _ in reader]
                embeddings = get_text_embeddings(captions)
                image_collection.add(
                    ids=ids,
                    embeddings=embeddings, # type: ignore[reportArgumentType]
                    metadatas=metadatas, # type: ignore[reportArgumentType]
                    documents=captions  # type: ignore[reportArgumentType]
                )
    client.persist()  # type: ignore

# Example usage:
# process_embeddings("/path/to/extracted_data", "/mnt/fileshare/chromadb")

