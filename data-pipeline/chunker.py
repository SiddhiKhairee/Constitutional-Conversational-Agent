"""
Indian Constitution — PDF Parser & Chunker
==========================================
Strategy: chunk by Article boundary, not by fixed token size.
Each chunk = one Article (with its clauses/sub-clauses intact).
Metadata is stored alongside every chunk for precise retrieval.

Usage:
    python chunker.py --pdf path/to/constitution.pdf --out chunks.json
"""

import re
import json
import argparse
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

try:
    import pdfplumber
except ImportError:
    raise ImportError("Run: pip install pdfplumber --break-system-packages")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Chunk dataclass ────────────────────────────────────────────────────────────

@dataclass
class ConstitutionChunk:
    chunk_id: str            # e.g. "PART_III__ARTICLE_21"
    article_number: str      # e.g. "21", "21A", "370"
    article_title: str       # e.g. "Protection of life and personal liberty"
    part: str                # e.g. "III", "IV", "IVA"
    part_title: str          # e.g. "Fundamental Rights"
    text: str                # full text of the article + clauses
    page_start: int
    page_end: int
    amendment: Optional[str] = None  # e.g. "86th Amendment, 2002" if inserted later
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


# ── Regex patterns (FIXED) ─────────────────────────────────────────────────────

# Matches: "21. Protection...", "21A. Right...", or optional "Article 21."
# Handles leading spaces from PDF extraction and terminates at the em-dash (—) or dot.
# ARTICLE_HEADER = re.compile(
#     r"^\s*(?:Article\s+)?(\d+[A-Z]?)\s*[.\-—–]\s*(.+?)(?:[.\-—–]|\n|$)",
#     re.IGNORECASE | re.MULTILINE
# )
# The optimized version of the 1st Regex
ARTICLE_HEADER = re.compile(
    r"^\s*(\d+[A-Z-]*[A-Z]?)\.\s+\d*\[?([^\n]+(?:\n[^\n—.]+)??)(?:\]\.—|\]—|\.—|—)",
    re.MULTILINE
)

# Matches Part headers securely by allowing leading/trailing whitespaces: "  PART III  "
PART_HEADER = re.compile(
    r"^\s*PART\s+([IVXLCDA]+[A-Z]?)\s*$",
    re.IGNORECASE | re.MULTILINE
)

# Amendment insertion note (common in official PDFs)
AMENDMENT_NOTE = re.compile(
    r"(Ins(?:erted)?\.?\s+by\s+.+?Amendment.+?(?:\d{4}))",
    re.IGNORECASE
)

# Schedule headers — so we can stop chunking Articles and start chunking Schedules
SCHEDULE_HEADER = re.compile(
    r"^\s*THE\s+(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH)\s+SCHEDULE",
    re.IGNORECASE | re.MULTILINE
)


# ── PDF extraction ─────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text page by page from the PDF.
    Returns list of {page_num, text} dicts.
    """

    # gotta start from page 33, skip content
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        log.info(f"PDF opened: {len(pdf.pages)} pages")
        for i, page in enumerate(pdf.pages, start=33):
            text = page.extract_text(x_tolerance=2, y_tolerance=3)
            if text:
                # Clean up common PDF artifacts
                text = clean_page_text(text)
                pages.append({"page_num": i + 1, "text": text})
    return pages


def clean_page_text(text: str) -> str:
    """Remove page headers, footers, and OCR artifacts common in Constitution PDFs."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        # Skip page numbers (lone digits)
        if re.match(r"^\d+$", line):
            continue
        # Skip common header/footer patterns
        if re.match(r"^(THE CONSTITUTION OF INDIA|Constitution of India)\s*$", line, re.I):
            continue
        # Skip very short lines that are likely artifacts
        if len(line) < 3:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def merge_pages(pages: list[dict]) -> str:
    """Merge all page texts with a page boundary marker."""
    parts = []
    for p in pages:
        parts.append(f"\n<<PAGE {p['page_num']}>>\n{p['text']}")
    return "\n".join(parts)


# ── Part detection ─────────────────────────────────────────────────────────────

KNOWN_PARTS = {
    "I":    "The Union and its Territory",
    "II":   "Citizenship",
    "III":  "Fundamental Rights",
    "IV":   "Directive Principles of State Policy",
    "IVA":  "Fundamental Duties",
    "V":    "The Union",
    "VI":   "The States",
    "VII":  "States in Part B of the First Schedule (Repealed)",
    "VIII": "The Union Territories",
    "IX":   "The Panchayats",
    "IXA":  "The Municipalities",
    "IXB":  "The Co-operative Societies",
    "X":    "The Scheduled and Tribal Areas",
    "XI":   "Relations between the Union and the States",
    "XII":  "Finance, Property, Contracts and Suits",
    "XIII": "Trade, Commerce and Intercourse within the Territory of India",
    "XIV":  "Services Under the Union and the States",
    "XIVA": "Tribunals",
    "XV":   "Elections",
    "XVI":  "Special Provisions Relating to Certain Classes",
    "XVII": "Official Language",
    "XVIII":"Emergency Provisions",
    "XIX":  "Miscellaneous",
    "XX":   "Amendment of the Constitution",
    "XXI":  "Temporary, Transitional and Special Provisions",
    "XXII": "Short Title, Commencement, Authoritative Text in Hindi and Repeals",
}


def detect_current_part(text_so_far: str) -> tuple[str, str]:
    """Find the most recent Part heading before this point in the document."""
    matches = list(PART_HEADER.finditer(text_so_far))
    if not matches:
        return ("I", KNOWN_PARTS.get("I", ""))
    last = matches[-1]
    part_num = last.group(1).upper()
    part_title = KNOWN_PARTS.get(part_num, "Unknown")
    return (part_num, part_title)


# ── Amendment detection ────────────────────────────────────────────────────────

def detect_amendment(text: str) -> Optional[str]:
    """Check if this article was inserted/substituted by a constitutional amendment."""
    match = AMENDMENT_NOTE.search(text)
    if match:
        return match.group(1).strip()
    return None


# ── Page number extraction from merged text ────────────────────────────────────

def get_page_for_position(merged_text: str, pos: int) -> int:
    """Given a character position in merged text, find which page it's on."""
    page_markers = list(re.finditer(r"<<PAGE (\d+)>>", merged_text))
    current_page = 1
    for m in page_markers:
        if m.start() > pos:
            break
        current_page = int(m.group(1))
    return current_page


# ── Core chunking logic ────────────────────────────────────────────────────────

def parse_article_id(art_num_str: str) -> tuple[int, str]:
    """Converts '279A' into (279, 'A') for exact chronological sorting."""
    match = re.match(r"(\d+)([A-Z]*)", art_num_str)
    if match:
        num = int(match.group(1))
        suffix = match.group(2)
        return (num, suffix)
    return (0, "")

def chunk_by_article(merged_text: str) -> list[ConstitutionChunk]:
    chunks = []
    
    # 1. Extract matches using the tightened typography regex
    article_matches = list(ARTICLE_HEADER.finditer(merged_text))
    log.info(f"Regex matched {len(article_matches)} raw potential headers.")

    schedule_match = SCHEDULE_HEADER.search(merged_text)
    text_end = schedule_match.start() if schedule_match else len(merged_text)

    # Track sequence using our tuple comparison
    last_valid_sequence = (0, "")

    for i, match in enumerate(article_matches):
        article_num = match.group(1).strip()   # e.g., "279A"
        article_title = match.group(2).strip() # e.g., "Goods and Services Tax Council"

        # Parse into sortable tuple
        current_sequence = parse_article_id(article_num)

        # 2. SEQUENCE VALIDATION GUARDRAIL
        # If the detected number goes backward (e.g., hitting a footnote referencing an earlier article)
        # or jumps unrealistically ahead by more than 50 numbers, it's a footnote noise match. Skip it!
        if current_sequence <= last_valid_sequence or (current_sequence[0] - last_valid_sequence[0] > 50 and last_valid_sequence[0] != 0):
            continue  # Rejects footnote noise securely without dropping '279A'

        # Set text boundaries
        start = match.start()
        
        # Look ahead for the next truly valid sequence element to find the end boundary
        end = text_end
        for next_match in article_matches[i + 1:]:
            next_num = next_match.group(1).strip()
            next_sequence = parse_article_id(next_num)
            # Only cut the chunk if the next anchor is chronologically moving forward
            if next_sequence > current_sequence:
                end = next_match.start()
                break

        article_text = merged_text[start:end].strip()
        article_text = re.sub(r"<<PAGE \d+>>\n?", "", article_text).strip()

        # Update tracking index
        last_valid_sequence = current_sequence

        # Metadata extraction & Chunk generation
        part_num, part_title = detect_current_part(merged_text[:start])
        page_start = get_page_for_position(merged_text, start)
        page_end = get_page_for_position(merged_text, end)

        chunks.append(ConstitutionChunk(
            chunk_id=f"PART_{part_num}__ARTICLE_{article_num}",
            article_number=article_num,
            article_title=article_title,
            part=part_num,
            part_title=part_title,
            text=article_text,
            page_start=page_start,
            page_end=page_end,
            amendment=detect_amendment(article_text)
        ))

    return chunks
    """
    Split the Constitution text into per-Article chunks.
    Each Article boundary is detected by the ARTICLE_HEADER regex.
    Text between two Article headers belongs to the first Article.
    """
    chunks = []

    # Find all Article headers and their positions
    article_matches = list(ARTICLE_HEADER.finditer(merged_text))

    if not article_matches:
        log.warning("No Article headers found — check PDF extraction quality.")
        return chunks

    log.info(f"Found {len(article_matches)} article headers")

    # Stop before Schedules
    schedule_match = SCHEDULE_HEADER.search(merged_text)
    text_end = schedule_match.start() if schedule_match else len(merged_text)

    for i, match in enumerate(article_matches):
        article_num = match.group(1).strip()
        article_title = match.group(2).strip()

        # Text for this article = from this header to the next (or end)
        start = match.start()
        end = article_matches[i + 1].start() if i + 1 < len(article_matches) else text_end
        article_text = merged_text[start:end].strip()

        # Remove <<PAGE N>> markers from the text itself
        article_text = re.sub(r"<<PAGE \d+>>\n?", "", article_text).strip()

        # Skip empty or suspiciously short chunks (likely parsing artifacts from index pages)
        if len(article_text.split()) < 12:
            log.debug(f"Skipping line {article_num} — too short to be full article body")
            continue

        # Detect Part from everything before this article
        part_num, part_title = detect_current_part(merged_text[:start])

        # Detect amendment note
        amendment = detect_amendment(article_text)

        # Page numbers
        page_start = get_page_for_position(merged_text, start)
        page_end = get_page_for_position(merged_text, end)

        chunk = ConstitutionChunk(
            chunk_id=f"PART_{part_num}__ARTICLE_{article_num}",
            article_number=article_num,
            article_title=article_title,
            part=part_num,
            part_title=part_title,
            text=article_text,
            page_start=page_start,
            page_end=page_end,
            amendment=amendment,
        )
        chunks.append(chunk)

    return chunks


# ── Schedule chunking (bonus) ──────────────────────────────────────────────────

SCHEDULE_NAMES = {
    "FIRST": "Lists of States and Union Territories",
    "SECOND": "Provisions as to the President, Governors etc.",
    "THIRD": "Forms of Oaths or Affirmations",
    "FOURTH": "Allocation of Seats in the Council of States",
    "FIFTH": "Provisions as to the Administration of Scheduled Areas",
    "SIXTH": "Provisions as to the Administration of Tribal Areas",
    "SEVENTH": "Union List, State List, Concurrent List",
    "EIGHTH": "Languages",
    "NINTH": "Acts and Regulations (outside judicial review)",
    "TENTH": "Anti-defection provisions",
    "ELEVENTH": "Powers of Panchayats",
    "TWELFTH": "Powers of Municipalities",
}

def chunk_schedules(merged_text: str) -> list[ConstitutionChunk]:
    """Chunk the 12 Schedules as separate top-level chunks."""
    chunks = []
    schedule_matches = list(SCHEDULE_HEADER.finditer(merged_text))

    for i, match in enumerate(schedule_matches):
        ordinal = match.group(1).upper()
        title = SCHEDULE_NAMES.get(ordinal, "")
        start = match.start()
        end = schedule_matches[i + 1].start() if i + 1 < len(schedule_matches) else len(merged_text)
        text = re.sub(r"<<PAGE \d+>>\n?", "", merged_text[start:end]).strip()

        chunk = ConstitutionChunk(
            chunk_id=f"SCHEDULE_{ordinal}",
            article_number=f"SCH-{ordinal}",
            article_title=f"The {ordinal.title()} Schedule — {title}",
            part="SCHEDULES",
            part_title="Schedules",
            text=text,
            page_start=get_page_for_position(merged_text, start),
            page_end=get_page_for_position(merged_text, end),
        )
        chunks.append(chunk)

    return chunks


# ── Quality checks ─────────────────────────────────────────────────────────────

def validate_chunks(chunks: list[ConstitutionChunk]) -> None:
    """Log quality stats and warn about potential issues."""
    log.info(f"\n{'='*50}")
    log.info(f"Total chunks: {len(chunks)}")

    word_counts = [c.word_count for c in chunks]
    if word_counts:
        log.info(f"Avg words/chunk: {sum(word_counts) // len(word_counts)}")
        log.info(f"Min words: {min(word_counts)} | Max words: {max(word_counts)}")

    # Warn about very long chunks (might need sub-chunking for embedding)
    long_chunks = [c for c in chunks if c.word_count > 600]
    if long_chunks:
        log.warning(f"{len(long_chunks)} chunks exceed 600 words — consider sub-chunking:")
        for c in long_chunks[:5]:
            log.warning(f"  Article {c.article_number}: {c.word_count} words")

    # Check Part coverage
    parts_seen = set(c.part for c in chunks)
    log.info(f"Parts covered: {sorted(parts_seen)}")

    # Detect likely missing articles (gaps in numbering)
    nums = []
    for c in chunks:
        try:
            nums.append(int(re.sub(r"[A-Z]", "", c.article_number)))
        except ValueError:
            pass
    nums.sort()
    gaps = [nums[i] for i in range(1, len(nums)) if nums[i] - nums[i-1] > 2]
    if gaps:
        log.warning(f"Possible missing articles around: {gaps[:10]}")

    log.info("="*50)


# ── Sub-chunking for long articles ────────────────────────────────────────────

def sub_chunk_long_articles(
    chunks: list[ConstitutionChunk],
    max_words: int = 500
) -> list[ConstitutionChunk]:
    """
    For articles that exceed max_words (e.g. Article 356, lengthy schedules),
    split by clause — lines starting with '(' like (1), (a), (i).
    The parent article metadata is preserved in each sub-chunk.
    """
    final = []
    clause_pattern = re.compile(r"^(\(\d+\)|\([a-z]\)|\([ivxlc]+\))", re.MULTILINE)

    for chunk in chunks:
        if chunk.word_count <= max_words:
            final.append(chunk)
            continue

        # Try to split on clause boundaries
        clause_positions = [m.start() for m in clause_pattern.finditer(chunk.text)]

        if len(clause_positions) < 2:
            # Can't split cleanly — keep as is but log it
            log.debug(f"Article {chunk.article_number} is long but has no clause splits")
            final.append(chunk)
            continue

        # Group clauses into sub-chunks of ~max_words
        sub_texts = []
        current_start = 0
        current_words = 0

        for j, pos in enumerate(clause_positions):
            segment = chunk.text[current_start:pos]
            seg_words = len(segment.split())

            if current_words + seg_words > max_words and current_words > 0:
                sub_texts.append(chunk.text[current_start:pos])
                current_start = pos
                current_words = seg_words
            else:
                current_words += seg_words

        # Last segment
        sub_texts.append(chunk.text[current_start:])

        for k, sub_text in enumerate(sub_texts):
            sub = ConstitutionChunk(
                chunk_id=f"{chunk.chunk_id}__SUB_{k+1}",
                article_number=chunk.article_number,
                article_title=chunk.article_title,
                part=chunk.part,
                part_title=chunk.part_title,
                text=sub_text.strip(),
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                amendment=chunk.amendment,
            )
            final.append(sub)

    log.info(f"After sub-chunking: {len(final)} chunks (was {len(chunks)})")
    return final


# ── Save ──────────────────────────────────────────────────────────────────────

def save_chunks(chunks: list[ConstitutionChunk], out_path: str) -> None:
    data = [asdict(c) for c in chunks]
    Path(out_path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log.info(f"Saved {len(chunks)} chunks → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_and_chunk(pdf_path: str, out_path: str = "chunks.json") -> list[ConstitutionChunk]:
    log.info(f"Loading PDF: {pdf_path}")
    pages = extract_text_from_pdf(pdf_path)

    log.info("Merging pages...")
    merged = merge_pages(pages)

    log.info("Chunking by Article boundary...")
    article_chunks = chunk_by_article(merged)

    log.info("Chunking Schedules...")
    schedule_chunks = chunk_schedules(merged)

    all_chunks = article_chunks + schedule_chunks

    log.info("Sub-chunking long articles...")
    all_chunks = sub_chunk_long_articles(all_chunks, max_words=500)

    validate_chunks(all_chunks)
    save_chunks(all_chunks, out_path)

    return all_chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and chunk the Indian Constitution PDF")
    parser.add_argument("--pdf", required=True, help="Path to constitution.pdf")
    parser.add_argument("--out", default="chunks.json", help="Output JSON path")
    args = parser.parse_args()

    chunks = parse_and_chunk(args.pdf, args.out)
    print(f"\nDone. {len(chunks)} chunks saved to {args.out}")