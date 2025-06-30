import os
import json
import csv
from io import StringIO
from pymongo import MongoClient
from azure.storage.blob import BlobServiceClient
from embedding_utils import get_text_embeddings
import logging

# No need to configure logging here, the Azure Function host does it.

def process_single_pdf_embeddings(
    blob_service_client: BlobServiceClient,
    container_name: str,
    pdf_blob_prefix: str,
    mongo_uri: str
):
    """
    Generates and stores embeddings for a SINGLE PDF's extracted data.
    This function is idempotent and expects a pre-authenticated blob client.

    Args:
        blob_service_client: An authenticated Azure BlobServiceClient instance.
        container_name: The name of the blob container (e.g., 'container1').
        pdf_blob_prefix: The prefix for the specific PDF's data (e.g., 'extracted_data/39593126').
        mongo_uri: The connection string for MongoDB.
    """
    logging.info(f"Starting embedding generation for prefix: {pdf_blob_prefix}")

    try:
        container_client = blob_service_client.get_container_client(container_name)
        mongo_client = MongoClient(mongo_uri)
        db = mongo_client["vector_database"]
        text_coll = db["textEmbeddings"]
        img_coll = db["imageEmbeddings"]

        # Extract the unique PDF identifier from the prefix
        pdf_id = os.path.basename(pdf_blob_prefix)

        # --- Process Text Embeddings ---
        chunks_blob_name = f"{pdf_blob_prefix}/{pdf_id}_chunks.json"
        logging.info(f"Processing text chunks from: {chunks_blob_name}")
        
        blob_client = container_client.get_blob_client(chunks_blob_name)
        if not blob_client.exists():
            logging.warning(f"Text chunks blob not found, skipping: {chunks_blob_name}")
        else:
            data = blob_client.download_blob().readall()
            chunks = json.loads(data)
            
            # Validate the structure of chunks
            try:
                if not isinstance(chunks, list):
                    raise ValueError("Expected 'chunks' to be a list of strings.")
                if not all(isinstance(chunk, str) for chunk in chunks):
                    raise ValueError("Each chunk must be a string.")
            except ValueError as ve:
                logging.error(f"Invalid structure for text chunks in {chunks_blob_name}: {ve}")
                raise

            ids_to_check = [f"{pdf_id}_chunk_{i+1}" for i in range(len(chunks))]
            existing_ids = {doc['_id'] for doc in text_coll.find({'_id': {'$in': ids_to_check}}, {'_id': 1})}
            
            new_chunks = [chunk for i, chunk in enumerate(chunks) if ids_to_check[i] not in existing_ids]
            
            if not new_chunks:
                logging.info(f"All text chunks for {pdf_id} are already embedded.")
            else:
                logging.info(f"Generating embeddings for {len(new_chunks)} new text chunks.")
                embeddings = get_text_embeddings(new_chunks) # Directly use strings as text chunks
                documents = [
                    {
                        "_id": f"{pdf_id}_chunk_{i+1}", 
                        "embedding": embedding, 
                        "metadata": {"pdf_id": pdf_id, "source": chunks_blob_name}, 
                        "text": chunk
                    } 
                    for i, (embedding, chunk) in enumerate(zip(embeddings, new_chunks))
                ]
                text_coll.insert_many(documents)
                logging.info(f"Successfully inserted {len(documents)} new text embeddings for {pdf_id}.")

        # --- Process Image Embeddings ---
        captions_blob_name = f"{pdf_blob_prefix}/images/{pdf_id}_captions.csv"
        logging.info(f"Processing image captions from: {captions_blob_name}")

        blob_client = container_client.get_blob_client(captions_blob_name)
        if not blob_client.exists():
            logging.warning(f"Image captions blob not found, skipping: {captions_blob_name}")
        else:
            data = blob_client.download_blob().readall().decode('utf-8')
            reader = list(csv.DictReader(StringIO(data)))

            # IDEMPOTENCY CHECK
            ids_to_check = [f"{pdf_id}_{row['image_name']}" for row in reader]
            existing_ids = {doc['_id'] for doc in img_coll.find({'_id': {'$in': ids_to_check}}, {'_id': 1})}

            new_captions_data = [row for row in reader if f"{pdf_id}_{row['image_name']}" not in existing_ids]

            if not new_captions_data:
                logging.info(f"All image captions for {pdf_id} are already embedded.")
            else:
                logging.info(f"Generating embeddings for {len(new_captions_data)} new image captions.")
                captions = [row['caption'] for row in new_captions_data]
                embeddings = get_text_embeddings(captions)
                documents = [
                    {
                        "_id": f"{pdf_id}_{row['image_name']}",
                        "embedding": embedding,
                        "metadata": {"pdf_id": pdf_id, "source": captions_blob_name},
                        "text": caption
                    }
                    for row, embedding, caption in zip(new_captions_data, embeddings, captions)
                ]
                img_coll.insert_many(documents)
                logging.info(f"Successfully inserted {len(documents)} new image embeddings for {pdf_id}.")

        logging.info(f"Completed embedding generation for prefix: {pdf_blob_prefix}")

    except Exception as e:
        logging.error(f"An error occurred during embedding generation for {pdf_blob_prefix}: {e}")
        # Re-raise the exception so the calling function (e.g., queue trigger) knows it failed and can retry.
        raise