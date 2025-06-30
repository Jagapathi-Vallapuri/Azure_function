import fitz  # PyMuPDF
import os
import logging
import csv
import time
from image_caption import generate_image_caption

def extract_images_and_caption(pdf_path, output_dir, min_width=128, min_height=128, delay=0.2):
    """
    Extracts images from a PDF (ignoring first and last page), saves them, captions them, and writes a CSV.
    Adds a delay between API calls to avoid overloading the Gemini API.
    Images and CSV are saved in output_dir/images/.
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    images_output_dir = os.path.join(output_dir, "images")
    os.makedirs(images_output_dir, exist_ok=True)
    csv_path = os.path.join(images_output_dir, f"{pdf_name}_captions.csv")
    captions = []
    try:
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        for page_num in range(1, page_count - 1):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                if width < min_width or height < min_height:
                    continue  # Skip small images
                image_filename = f"page_{page_num+1}_img_{img_index+1}.{image_ext}"
                image_path = os.path.join(images_output_dir, image_filename)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                # Caption the image
                try:
                    caption = generate_image_caption(image_path)
                except Exception as e:
                    caption = f"Error: {e}"
                captions.append({"image_name": image_filename, "caption": caption})
                time.sleep(delay)  
        
        with open(csv_path, "w", newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["image_name", "caption"])
            writer.writeheader()
            writer.writerows(captions)
        logging.info(f"  - Successfully extracted and captioned images from {pdf_name}")
    except Exception as e:
        logging.error(f"  - Error extracting/captioning images from {pdf_name}: {e}")
        raise
