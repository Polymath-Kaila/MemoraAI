
import re
from typing import List

def chunk_text(text: str, max_chars: int = 1200) -> List[str]:
    # naive chunking on sentence boundaries
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 > max_chars:
            if cur:
                chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks

def approx_token_count(text: str) -> int:
    # rough estimate ~ 4 chars per token
    return max(1, int(len(text) / 4))
