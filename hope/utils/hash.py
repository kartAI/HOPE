import hashlib


def md5_hash(text: str) -> str:
    """Generate MD5 hash for the given text."""
    md5 = hashlib.md5()
    md5.update(text.encode("utf-8"))
    return md5.hexdigest()
