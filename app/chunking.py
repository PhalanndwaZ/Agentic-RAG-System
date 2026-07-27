SEPARATORS = ["\n\n", ". ", " "]  # paragraph, sentence, word — in order of preference


def _pack(pieces: list[str], separator: str, chunk_size: int, remaining_separators: list[str]) -> list[str]:
    chunks = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{separator}{piece}" if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(piece) <= chunk_size:
                current = piece
            else:
                # piece is still too big — recurse with the next separator down
                chunks.extend(_recursive_split(piece, remaining_separators, chunk_size))
                current = ""
    if current:
        chunks.append(current)
    return chunks


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]  # last resort

    separator, *rest = separators
    pieces = text.split(separator)
    return _pack(pieces, separator, chunk_size, rest)


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]
        overlapped.append(f"{tail} {chunks[i]}")
    return overlapped


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    raw_chunks = _recursive_split(text.strip(), SEPARATORS, chunk_size)
    raw_chunks = [c.strip() for c in raw_chunks if c.strip()]
    return _add_overlap(raw_chunks, chunk_overlap)

