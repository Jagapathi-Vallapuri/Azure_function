import azure.functions as func
import logging
import os
import tempfile
import time
import json
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from text_extractor import extract_text
from image_extractor import extract_images_and_caption
from table_extractor import extract_tables
from pymongo import MongoClient

app = func.FunctionApp()

def get_blob_service_client():
    """
    Creates a BlobServiceClient with intelligent authentication:
    - Local development: Uses connection string (no Azure CLI required)
    - Azure production: Uses managed identity
    """
    connection_string = os.environ.get("storage1rag_STORAGE")

    if connection_string:
        try:
            logging.info("Using connection string for blob service authentication")
            return BlobServiceClient.from_connection_string(connection_string)
        except Exception as e:
            logging.error(f"Connection string authentication failed: {e}")
            raise Exception("Failed to authenticate using connection string. Check the value of 'storage1rag_STORAGE'.")

    # Only attempt managed identity if running in Azure
    if os.environ.get("WEBSITE_SITE_NAME"):
        try:
            logging.info("Using managed identity for blob service authentication")
            account_url = "https://storage1rag.blob.core.windows.net"
            credential = DefaultAzureCredential()
            return BlobServiceClient(account_url=account_url, credential=credential)
        except Exception as e:
            logging.error(f"Managed identity authentication failed: {e}")

    logging.error("No valid authentication method available. Ensure connection string is configured for local development.")
    raise Exception("AuthenticationError: Unable to authenticate with Azure Blob Storage. Check environment variables and managed identity configuration.")

def upload_folder_to_blob(blob_service_client, local_folder_path, container_name, blob_folder_prefix):
    """
    Uploads all files from a local folder to a blob container with the specified prefix.
    """
    try:
        container_client = blob_service_client.get_container_client(container_name)
        
        for root, dirs, files in os.walk(local_folder_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_folder_path)
                blob_name = f"{blob_folder_prefix}/{relative_path}".replace("\\", "/")
                
                with open(local_file_path, "rb") as data:
                    container_client.upload_blob(name=blob_name, data=data, overwrite=True)
                    logging.info(f"Uploaded {local_file_path} to {blob_name}")
                    
    except Exception as e:
        logging.error(f"Error uploading folder to blob: {e}")
        raise

@app.blob_trigger(arg_name="myblob", path="container1/pdfs/{name}",
                  connection="storage1rag_STORAGE") 
def process_pdf_function(myblob: func.InputStream):
    """
    Azure Function triggered when a PDF file is uploaded to the pdfs directory.
    Extracts text, images, and tables, then uploads results to extracted_data folder.
    """
    try:
        blob_name = myblob.name or "unknown.pdf"
        logging.info(f"Processing PDF: {blob_name}")

        if not blob_name.lower().endswith('.pdf'):
            logging.warning(f"Skipping non-PDF file: {blob_name}")
            return

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_filename = os.path.basename(blob_name)
            temp_pdf_path = os.path.join(temp_dir, pdf_filename)

            with open(temp_pdf_path, 'wb') as temp_file:
                temp_file.write(myblob.read())

            output_dir = os.path.join(temp_dir, "extracted_data")
            os.makedirs(output_dir, exist_ok=True)

            extract_text(temp_pdf_path, output_dir)
            extract_images_and_caption(temp_pdf_path, output_dir)
            extract_tables(temp_pdf_path, output_dir)

            blob_service_client = get_blob_service_client()

            pdf_name_without_ext = os.path.splitext(pdf_filename)[0]
            blob_folder_prefix = f"extracted_data/{pdf_name_without_ext}"

            upload_folder_to_blob(
                blob_service_client=blob_service_client,
                local_folder_path=output_dir,
                container_name="container1",
                blob_folder_prefix=blob_folder_prefix
            )

            logging.info(f"Successfully processed and uploaded PDF: {pdf_filename}")

            mongo_uri = os.getenv("MONGO_URI")
            if mongo_uri:
                from embedding_pipeline_blob import process_single_pdf_embeddings
                blob_prefix = f"extracted_data/{pdf_name_without_ext}"
                process_single_pdf_embeddings(
                    blob_service_client=get_blob_service_client(),
                    container_name="container1",
                    pdf_blob_prefix=blob_prefix,
                    mongo_uri=mongo_uri
                )

    except Exception as e:
        logging.error(f"Error processing PDF {myblob.name}: {str(e)}")
        raise

@app.route(route="generateEmbeddings", auth_level=func.AuthLevel.FUNCTION)
def generate_embeddings(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP-triggered function to generate embeddings for new folders under extracted_data.
    """
    try:
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            return func.HttpResponse(
                "Missing parameter: provide 'MONGO_URI' in environment variables.",
                status_code=400
            )

        from embedding_pipeline_blob import process_single_pdf_embeddings
        connection_string = os.getenv("storage1rag_STORAGE")
        if not connection_string:
            return func.HttpResponse(
                "Missing parameter: provide 'storage1rag_STORAGE' in environment variables.",
                status_code=400
            )

        blob_service_client = get_blob_service_client()
        container_name = "container1"
        blob_prefix = "extracted_data"

        process_single_pdf_embeddings(
            blob_service_client=blob_service_client,
            container_name=container_name,
            pdf_blob_prefix=blob_prefix,
            mongo_uri=mongo_uri
        )

        logging.info(f"Successfully processed embeddings for all folders under {blob_prefix}")
        return func.HttpResponse(
            "Processed embeddings for all folders in blob storage using MongoDB.",
            status_code=200
        )
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        return func.HttpResponse(f"Error: {e}", status_code=500)

@app.route(route="health", auth_level=func.AuthLevel.ANONYMOUS)
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """
    Health check endpoint to ensure the function app is running.
    """
    try:
        return func.HttpResponse("Function app is running.", status_code=200)
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return func.HttpResponse(f"Error: {e}", status_code=500)
