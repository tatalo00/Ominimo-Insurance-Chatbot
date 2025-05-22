import fitz  # PyMuPDF
from langchain.schema import Document
from pathlib import Path
import hashlib
import re
import nltk

nltk.download("punkt_tab")

# --- CONFIG ---
SENTENCES_PER_CHUNK = 3
MIN_CHUNK_LENGTH = 100  # characters
HEADER_PATTERN = re.compile(r"^(\d{1,2}(\.\d+)*\s*[\.\-\)]?\s+.*|[A-ZÁÉŐÚÜ\s]{5,})$")

FOOTER_HEADER_NOISE = [
    "Teljes körű OMINIMO CASCO biztosítási feltételek",
    "Függelék",
    "signal.hu",
    "Kötelező gépjármű-felelősségbiztosítás | Általános szerződési feltételek"
]

def clean_text(text: str) -> str:
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        if any(noise.lower() in line.lower() for noise in FOOTER_HEADER_NOISE):
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines).strip()

def get_headers_from_blocks(blocks):
    headers = []
    for b in blocks:
        line = b[4].strip()
        if HEADER_PATTERN.match(line):
            headers.append((b[1], line))  # use y-coordinate and text
    headers.sort()  # from top to bottom
    return headers

def find_nearest_header(headers, y_position):
    """
    Finds the nearest preceding header for a given y_position.
    """
    current_header = "N/A"
    for y, text in headers:
        if y > y_position:
            break
        current_header = text
    return current_header

def semantic_sentence_chunk(text: str, source: str, page: int, headers=None):
    from nltk.tokenize import sent_tokenize

    def find_nearest_header(y_position, headers):
        current_header = "N/A"
        for y, h in headers:
            if y > y_position:
                break
            current_header = h
        return current_header

    sentences = sent_tokenize(text)
    chunks = []
    seen_hashes = set()
    headers = headers or []

    for i in range(0, len(sentences), SENTENCES_PER_CHUNK):
        chunk = " ".join(sentences[i:i + SENTENCES_PER_CHUNK]).strip()
        if len(chunk) >= MIN_CHUNK_LENGTH:
            chunk_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()
            if chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)

                # Approximate y_position from first sentence if needed
                header = headers[0][1] if headers else "N/A"

                chunks.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": source,
                        "page": page,
                        "header": header
                    }
                ))
    return chunks

def extract_block_chunks(pdf_path: Path):
    doc = fitz.open(pdf_path)
    all_docs = []
    seen_hashes = set()

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))

        headers = get_headers_from_blocks(blocks)

        buffer = ""
        last_y = None
        current_y = 0

        for b in blocks:
            text = clean_text(b[4].strip())
            y = b[1]

            if last_y is not None and abs(y - last_y) > 20:
                if len(buffer.strip()) >= MIN_CHUNK_LENGTH:
                    chunk_hash = hashlib.md5(buffer.encode("utf-8")).hexdigest()
                    if chunk_hash not in seen_hashes:
                        seen_hashes.add(chunk_hash)
                        header = find_nearest_header(headers, current_y)
                        all_docs.append(Document(
                            page_content=buffer.strip(),
                            metadata={
                                "source": pdf_path.stem,
                                "page": page_num,
                                "header": header
                            }
                        ))
                    buffer = ""
            buffer += " " + text
            last_y = y

        if len(buffer.strip()) >= MIN_CHUNK_LENGTH:
            chunk_hash = hashlib.md5(buffer.encode("utf-8")).hexdigest()
            if chunk_hash not in seen_hashes:
                header = find_nearest_header(headers, current_y)
                seen_hashes.add(chunk_hash)
                all_docs.append(Document(
                    page_content=buffer.strip(),
                    metadata={
                        "source": pdf_path.stem,
                        "page": page_num,
                        "header": header
                    }
                ))
    #Save chunks in .txt file
    with open(f"chunks_output{pdf_path.name[:-4]}", "w", encoding="utf-8") as f:
        for i, doc in enumerate(all_docs, 1):
            f.write(f"--- Chunk #{i} ---\n")
            f.write(doc.page_content.strip() + "\n")
            f.write(f"[source: {doc.metadata['source']}, page: {doc.metadata['page']}]\n\n")

    return all_docs

def extract_and_chunk(pdf_path: Path):
    if "mtpl_coverage" in pdf_path.stem.lower():
        return extract_block_chunks(pdf_path)

    doc = fitz.open(pdf_path)
    all_docs = []
    seen_hashes = set()

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))
        full_text = clean_text("\n".join(b[4] for b in blocks))

        # NEW: Extract headers from page
        headers = get_headers_from_blocks(blocks)

        # NEW: Pass headers to chunking function
        chunks = semantic_sentence_chunk(full_text, pdf_path.stem, page_num, headers)
        for chunk in chunks:
            chunk_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()
            if chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)
                all_docs.append(chunk)

    with open(f"chunks_output_{pdf_path.stem}.txt", "w", encoding="utf-8") as f:
        for i, doc in enumerate(all_docs, 1):
            f.write(f"--- Chunk #{i} ---\n")
            f.write(doc.page_content.strip() + "\n")
            f.write(f"[source: {doc.metadata['source']}, page: {doc.metadata['page']}, header: {doc.metadata.get('header', 'N/A')}]\n\n")

    return all_docs


