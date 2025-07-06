# utils/helpers.py

def get_text_from_block(block) -> str:
    """
    Safely extract text from a chunk (handles multiple possible attributes).
    """
    if hasattr(block, "text"):
        return block.text
    if hasattr(block, "get_text"):
        return block.get_text()
    return str(block)
