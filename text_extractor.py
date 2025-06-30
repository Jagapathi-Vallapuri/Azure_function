import fitz  # PyMuPDF
import os
import logging
import re
import json

def chunk_text(text, chunk_size=256, overlap=32):
    """
    Splits text into overlapping chunks of approximately chunk_size words.
    Each chunk overlaps the previous by 'overlap' words.
    Returns a list of text chunks.
    """
    words = text.split()
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        if not chunk:
            break
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks

def preprocess_text(text):
    """
    Cleans text by removing non-printable characters and extra whitespace.
    """
    text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text(pdf_path, output_dir, chunk_size=512, overlap=32):
    """
    Extracts text from a PDF, chunks it with overlap, preprocesses each chunk, and saves the preprocessed chunks as a JSON file.
    No chunks are filtered out.
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    text_output_path = os.path.join(output_dir, f"{pdf_name}.txt")
    chunks_output_path = os.path.join(output_dir, f"{pdf_name}_chunks.json")

    try:
        doc = fitz.open(pdf_path)
        text = ""
        # Ignore first and last page
        for i in range(1, doc.page_count - 1):
            page = doc.load_page(i)
            page_text = page.get_text("text")  # type: ignore[attr-defined]
            text += f"\n--- Page {i+1} ---\n"
            text += page_text.strip() + "\n"
        with open(text_output_path, "w", encoding="utf-8") as f:
            f.write(text)
        # Chunk the text (ignoring page markers)
        text_for_chunking = re.sub(r'\n--- Page \d+ ---\n', ' ', text)
        chunks = chunk_text(text_for_chunking, chunk_size=chunk_size, overlap=overlap)
        preprocessed_chunks = [preprocess_text(chunk) for chunk in chunks if chunk.strip()]
        with open(chunks_output_path, "w", encoding="utf-8") as f:
            json.dump(preprocessed_chunks, f, ensure_ascii=False, indent=2)
        logging.info(f"  - Successfully extracted, chunked, and preprocessed text from {pdf_name}")
    except Exception as e:
        logging.error(f"  - Error extracting/chunking/preprocessing text from {pdf_name}: {e}")