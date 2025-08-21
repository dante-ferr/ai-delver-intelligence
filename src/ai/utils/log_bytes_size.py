import logging


def log_bytes_size(data_bytes: bytes, context_label: str):
    try:
        size_in_bytes = len(data_bytes)
        size_in_kb = size_in_bytes / 1024
        size_in_mb = size_in_kb / 1024

        logging.info(
            f"[{context_label}] Data size: "
            f"{size_in_bytes} Bytes | "
            f"{size_in_kb:.2f} KB | "
            f"{size_in_mb:.2f} MB"
        )
    except Exception as e:
        logging.error(f"[{context_label}] Could not measure bytes size: {e}")
