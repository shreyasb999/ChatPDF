# main.py

from config.settings import PDF_FILE, OUTPUT_PATH
from processors.pdf_processor import load_and_partition_pdf, separate_chunks
from processors.image_utils import get_images_base64, display_base64_image
from utils.helpers import print_chunk_types

def main():
    print("📝 Partitioning PDF...")
    chunks = load_and_partition_pdf(PDF_FILE, OUTPUT_PATH)

    print("✅ PDF partitioned. Showing chunk types...")
    print_chunk_types(chunks)

    print("🔎 Separating tables and text...")
    tables, texts = separate_chunks(chunks)
    print(f"Tables found: {len(tables)}")
    print(f"Text chunks found: {len(texts)}")

    print("🖼️ Extracting base64 images from chunks...")
    images = get_images_base64(chunks)
    print(f"Images found: {len(images)}")

    if images:
        print("📷 Displaying first image...")
        display_base64_image(images[0])

if __name__ == "__main__":
    main()
