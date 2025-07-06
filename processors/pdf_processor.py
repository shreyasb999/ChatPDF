# processors/pdf_processor.py

from unstructured.partition.pdf import partition_pdf
from typing import List, Tuple, Any

def load_and_partition_pdf(file_path: str) -> List[Any]:
    """
    Partition the PDF into a list of unstructured 'chunks' (text, tables, images, etc.).
    """
    return partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )

def separate_chunks(chunks: List[Any]) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Split chunks into (text_blocks, table_blocks, image_blocks).
    """
    text_blocks = []
    table_blocks = []
    image_blocks = []

    for chunk in chunks:
        chunk_type = type(chunk).__name__.lower()
        if "table" in chunk_type:
            table_blocks.append(chunk)
        elif "image" in chunk_type:
            image_blocks.append(chunk)
        else:
            # assume everything else is text-like
            text_blocks.append(chunk)

    return text_blocks, table_blocks, image_blocks
