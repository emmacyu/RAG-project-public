"""PDF chunking for the vector store.

Kept free of app.config and database imports so it can be unit-tested and evaluated offline.

What it fixes compared with the original PyPDFLoader + 800-character splitter:
  1. Page numbers are 1-based (PyPDFLoader's `page` metadata is 0-based, so citations were off by one).
  2. Repeated page headers/footers are stripped, and the PDF's hard line wraps are joined back into
     paragraphs, so the splitter no longer cuts sentences at every visual line break.
  3. Chunk length is measured in the embedding model's own tokens and kept under its input limit,
     so no chunk is silently truncated before embedding.
  4. Every chunk gets a deterministic ID, so re-uploading a file overwrites its chunks instead of
     duplicating them.
  5. When the document has recognisable section headings, chunks follow its sections instead of its
     pages: a chunk never mixes two sections, may span a page break, and starts with its heading path
     (for example "[B. Enterprise-wide model risk management > B.3 Use of models]"). Documents without
     reliable headings fall back to page-by-page chunking.
"""
import hashlib
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# all-MiniLM-L6-v2 embeds at most 256 tokens (including [CLS] and [SEP]); anything longer is cut off.
# Leave headroom so the overlap and the special tokens never push a chunk past the limit.
DEFAULT_CHUNK_TOKENS = 200
DEFAULT_OVERLAP_TOKENS = 40

# A line this close to the widest lines on the page ran into the right margin, i.e. it is a wrap.
_WRAP_RATIO = 0.8
# A line that shows up on at least this share of pages near the top or bottom is a header/footer.
_REPEAT_RATIO = 0.5
_EDGE_LINES = 3


def _normalise_for_repeat(line: str) -> str:
    """Treat 'Page 3' and 'Page 4' as the same footer line."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _repeated_edge_lines(pages: list[list[str]]) -> set[str]:
    counts: Counter[str] = Counter()
    for lines in pages:
        edges = {_normalise_for_repeat(l) for l in lines[:_EDGE_LINES] + lines[-_EDGE_LINES:] if l.strip()}
        counts.update(edges)
    threshold = max(2, int(len(pages) * _REPEAT_RATIO))
    return {line for line, n in counts.items() if n >= threshold}


def _join_wrapped_lines(lines: list[str], wrap_width: int) -> str:
    """Rebuild paragraphs: a line that reaches the margin continues on the next line.

    Exception: a full line that ends a sentence and is followed by a capitalised line is treated as a
    paragraph end, otherwise the heading that follows ("D.1 Policies ...") would be glued onto it.
    """
    paragraphs: list[str] = []
    current = ""
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            if current:
                paragraphs.append(current)
                current = ""
            continue
        if current:
            joiner = "" if current.endswith("-") else " "
            current = f"{current}{joiner}{line}"
        else:
            current = line
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        sentence_end = line[-1] in ".!?:" and next_line[:1].isupper()
        if len(raw.rstrip()) < wrap_width or sentence_end:
            paragraphs.append(current)
            current = ""
    if current:
        paragraphs.append(current)
    return "\n".join(paragraphs)


def load_pdf_pages(file_path: str) -> list[dict]:
    """Return one cleaned text block per page: [{"page": 1, "page_label": "1", "text": "..."}]."""
    reader = PdfReader(file_path)
    raw_pages = [(page.extract_text() or "").split("\n") for page in reader.pages]

    repeated = _repeated_edge_lines(raw_pages)
    lengths = sorted(len(l.rstrip()) for lines in raw_pages for l in lines if l.strip())
    wrap_width = int(lengths[int(len(lengths) * 0.9)] * _WRAP_RATIO) if lengths else 0

    try:
        labels = list(reader.page_labels)
    except Exception:
        labels = []

    pages = []
    for index, lines in enumerate(raw_pages):
        kept = [l for l in lines if _normalise_for_repeat(l) not in repeated]
        text = _join_wrapped_lines(kept, wrap_width)
        label = labels[index] if index < len(labels) else str(index + 1)
        pages.append({"page": index + 1, "page_label": label, "text": text})
    return pages


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    # Imported lazily: loading the model is slow and only needed when a document is ingested.
    from sentence_transformers import SentenceTransformer

    model_id = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    return SentenceTransformer(model_id, device="cpu")


def _token_limit(model_name: str) -> int:
    return _load_model(model_name).max_seq_length - 2  # room for [CLS] and [SEP]


@lru_cache(maxsize=32)
def _token_splitter(model_name: str, chunk_tokens: int, overlap_tokens: int) -> RecursiveCharacterTextSplitter:
    limit = _token_limit(model_name)
    if chunk_tokens > limit:
        raise ValueError(f"chunk_tokens={chunk_tokens} exceeds {model_name}'s limit of {limit} tokens")
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        _load_model(model_name).tokenizer,
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        separators=["\n", ". ", "; ", ", ", " ", ""],
    )


# ---------------------------------------------------------------------------
# Section structure
# ---------------------------------------------------------------------------

# Major headings: (pattern, how to compute the level from the match)
_MAJOR_HEADINGS = [
    (re.compile(r"^(?:Appendix|Annex|Schedule)\s+[\w.-]+\b"), lambda m: 1),
    (re.compile(r"^([A-Z])((?:\.\d+)+)\s+[A-Z]"), lambda m: 1 + m.group(2).count(".")),  # A.1 Purpose
    (re.compile(r"^(\d+)((?:\.\d+)+)\s+[A-Z]"), lambda m: 1 + m.group(2).count(".")),    # 3.2 Data
    (re.compile(r"^(?:[A-Z]|[IVX]{1,4}|\d{1,2})\.\s+[A-Z]"), lambda m: 1),               # A. Overview
]
_MAX_HEADING_CHARS = 90
_MAX_HEADING_WORDS = 12
# Short label lines ("Model risk", "Model rationale") that introduce the paragraph below them.
_MAX_MINOR_CHARS = 50
_MAX_MINOR_WORDS = 6
_MIN_BODY_AFTER_MINOR = 60
# Fewer major headings than this and the document is treated as unstructured.
MIN_MAJOR_HEADINGS = 3


def _heading_shaped(line: str, max_chars: int, max_words: int) -> bool:
    return (
        0 < len(line) <= max_chars
        and len(line.split()) <= max_words
        and line[-1] not in ".,;:!?"
    )


def major_heading_level(line: str) -> int | None:
    """Level of a numbered/lettered section heading, or None.

    Numbered list items ("1. data input component ...", "1. Model risk is well understood ...") are
    rejected because they are long, start lower-case, or end with punctuation.
    """
    if not _heading_shaped(line, _MAX_HEADING_CHARS, _MAX_HEADING_WORDS):
        return None
    for pattern, level in _MAJOR_HEADINGS:
        match = pattern.match(line)
        if match:
            return level(match)
    return None


def is_minor_heading(line: str, next_line: str | None) -> bool:
    """A short capitalised label directly followed by a real paragraph."""
    return (
        _heading_shaped(line, _MAX_MINOR_CHARS, _MAX_MINOR_WORDS)
        and line[0].isupper()
        and not line[0].isdigit()
        and next_line is not None
        and len(next_line) >= _MIN_BODY_AFTER_MINOR
    )


def build_sections(pages: list[dict]) -> list[dict] | None:
    """Group cleaned page text into sections: [{"path": [...], "paragraphs": [(text, page), ...]}].

    Returns None when the document does not have enough recognisable headings to trust.
    """
    paragraphs = [(line.strip(), p["page"]) for p in pages for line in p["text"].split("\n") if line.strip()]
    major_count = sum(1 for text, _ in paragraphs if major_heading_level(text))
    if major_count < MIN_MAJOR_HEADINGS:
        return None

    sections: list[dict] = []
    major_path: list[str] = []   # numbered/lettered headings, nested by level
    minor: str | None = None     # current label heading under the major path
    body: list[tuple[str, int]] = []

    def flush():
        if body:
            sections.append({"path": major_path + ([minor] if minor else []), "paragraphs": list(body)})
            body.clear()

    for i, (text, page) in enumerate(paragraphs):
        next_text = paragraphs[i + 1][0] if i + 1 < len(paragraphs) else None
        level = major_heading_level(text)
        if level:
            flush()
            major_path[:] = major_path[: level - 1] + [text]
            minor = None
        elif is_minor_heading(text, next_text):
            flush()
            minor = text
        else:
            body.append((text, page))
    flush()
    return sections


def _section_chunks(sections, file_path, file_hash, labels, model_name, chunk_tokens, overlap_tokens, heading_prefix):
    tokenizer = _load_model(model_name).tokenizer
    docs, ids = [], []
    for s_index, section in enumerate(sections):
        path = section["path"]
        prefix = f"[{' > '.join(path)}]\n" if (heading_prefix and path) else ""
        budget = chunk_tokens - len(tokenizer.tokenize(prefix))
        if prefix and budget < 2 * overlap_tokens:  # very deep path: keep only the innermost heading
            prefix = f"[{path[-1]}]\n"
            budget = chunk_tokens - len(tokenizer.tokenize(prefix))
        splitter = _token_splitter(model_name, budget, overlap_tokens)

        body, spans, offset = [], [], 0
        for text, page in section["paragraphs"]:
            body.append(text)
            spans.append((offset, offset + len(text), page))
            offset += len(text) + 1
        body_text = "\n".join(body)

        cursor = 0
        for c_index, piece in enumerate(splitter.split_text(body_text)):
            start = body_text.find(piece, cursor)
            if start < 0:
                start = cursor
            end = start + len(piece)
            cursor = start + 1
            pages = [pg for s, e, pg in spans if s < end and e > start] or [spans[0][2]]
            docs.append(Document(
                page_content=prefix + piece,
                metadata={
                    "source": file_path,
                    "file_hash": file_hash,
                    "chunking": "structure",
                    "section": " > ".join(path),
                    "page": min(pages),
                    "page_end": max(pages),
                    "page_label": labels.get(min(pages), str(min(pages))),
                    "chunk": c_index,
                },
            ))
            ids.append(f"{file_hash}-s{s_index}-c{c_index}")
    return docs, ids


def _page_chunks(pages, file_path, file_hash, model_name, chunk_tokens, overlap_tokens):
    splitter = _token_splitter(model_name, chunk_tokens, overlap_tokens)
    docs, ids = [], []
    for page in pages:
        for i, piece in enumerate(splitter.split_text(page["text"])):
            docs.append(Document(
                page_content=piece,
                metadata={
                    "source": file_path,
                    "file_hash": file_hash,
                    "chunking": "page",
                    "page": page["page"],
                    "page_end": page["page"],
                    "page_label": page["page_label"],
                    "chunk": i,
                },
            ))
            ids.append(f"{file_hash}-p{page['page']}-c{i}")
    return docs, ids


def chunk_pdf(
    file_path: str,
    model_name: str,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    strategy: str = "structure",
    heading_prefix: bool = True,
) -> tuple[list[Document], list[str]]:
    """Split a PDF into embedding-sized chunks. Returns (documents, deterministic ids).

    strategy="structure" follows the document's sections and falls back to "page" when the
    document has no reliable headings; strategy="page" always chunks page by page.
    """
    file_hash = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()[:16]
    pages = load_pdf_pages(file_path)
    if strategy == "structure":
        sections = build_sections(pages)
        if sections:
            labels = {p["page"]: p["page_label"] for p in pages}
            return _section_chunks(sections, file_path, file_hash, labels, model_name,
                                   chunk_tokens, overlap_tokens, heading_prefix)
    elif strategy != "page":
        raise ValueError(f"unknown chunking strategy: {strategy}")
    return _page_chunks(pages, file_path, file_hash, model_name, chunk_tokens, overlap_tokens)
