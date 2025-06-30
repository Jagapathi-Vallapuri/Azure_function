import pdfplumber
import os
import logging
import pandas as pd

def extract_tables(pdf_path, output_dir):
    """
    Extracts tables from a PDF and saves them as CSV files.
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    tables_output_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_output_dir, exist_ok=True)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if tables:
                    for j, table in enumerate(tables):
                        df = pd.DataFrame(table[1:], columns=table[0])
                        table_filename = os.path.join(tables_output_dir, f"page_{i+1}_table_{j+1}.csv")
                        df.to_csv(table_filename, index=False)
        logging.info(f"  - Successfully extracted tables from {pdf_name}")
    except Exception as e:
        logging.error(f"  - Error extracting tables from {pdf_name}: {e}")
        raise
