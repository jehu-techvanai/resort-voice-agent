"""Knowledge base: parse DOCX / PDF / MD / TXT, chunk, embed, keep numbered versions, search the active one.

"Train" in the UI = build a new version from the files in staging and make it active. Older versions stay on disk for rollback.
"""
import json
import re
import shutil
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

import config
import store
from logging_utils import log as _log

KB_DIR = store.DATA_DIR / "kb"
STAGING = KB_DIR / "staging"
STAGING.mkdir(parents=True, exist_ok=True)
ALLOWED = {".docx", ".pdf", ".md", ".txt"}


@dataclass
class Chunk:
    id: str
    title: str
    text: str
    source: str
    # v2 metadata for filtering / explainability
    section: str = ""
    tags: list[str] = None  # type: ignore

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ParseError(Exception):
    pass


# ---------------------------------------------------------------- parsing into blocks
# A block is (kind, text) with kind in {"h", "p"}; page markers are ("page", "3").

_Q_START = re.compile(r"^\s*(q[\.:)]|question[\s:]|\d+[\.)]\s+.*\?\s*$)", re.I)


def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80 or s.endswith((".", ",", ";", ":", "?")):
        return False
    words = s.split()
    if len(words) > 10:
        return False
    caps = sum(1 for w in words if w[:1].isupper() or not w[:1].isalpha())
    return s.isupper() or caps >= max(1, int(len(words) * 0.7))


def _lines_to_blocks(lines: list[str]) -> list[tuple[str, str]]:
    """Merge wrapped lines from PDF/TXT into paragraphs; keep short title-like lines and questions on their own."""
    blocks, buf = [], []

    def flush():
        if buf:
            blocks.append(("p", " ".join(buf).strip()))
            buf.clear()

    for raw in lines:
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.endswith("?") and len(line) < 220 or _Q_START.match(line):
            flush()
            blocks.append(("p", line))
            continue
        if _looks_like_heading(line) and (not buf or buf[-1].endswith((".", "!", "?", ":"))):
            flush()
            blocks.append(("h", line))
            continue
        buf.append(line)
        if line.endswith((".", "!", "?")) and len(" ".join(buf)) > 400:
            flush()
    flush()
    return blocks


def parse_docx(path: Path) -> list[tuple[str, str]]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        d = docx.Document(str(path))
    except Exception as e:
        raise ParseError(f"cannot open DOCX: {e}")
    blocks = []
    for el in d.element.body.iterchildren():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(el, d)
            text = p.text.strip()
            if not text:
                continue
            style = (p.style.name or "").lower() if p.style is not None else ""
            all_bold = bool(p.runs) and all(r.bold for r in p.runs if r.text.strip())
            if style.startswith(("heading", "title")) or (all_bold and len(text) < 80 and not text.endswith("?")):
                blocks.append(("h", text))
            else:
                blocks.append(("p", text))
        elif tag == "tbl":
            t = Table(el, d)
            rows = [[c.text.strip() for c in r.cells] for r in t.rows]
            if not rows:
                continue
            header = rows[0]
            for r in rows[1:] or rows:
                if header and len(header) == len(r) and rows[1:]:
                    cells = [f"{h}: {v}" for h, v in zip(header, r) if v]
                else:
                    cells = [v for v in r if v]
                # de-duplicate merged cells
                seen, uniq = set(), []
                for c in cells:
                    if c not in seen:
                        uniq.append(c)
                        seen.add(c)
                if uniq:
                    blocks.append(("p", "; ".join(uniq) + "."))
    return blocks


def parse_pdf(path: Path) -> list[tuple[str, str]]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        raise ParseError(f"cannot open PDF: {e}")
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ParseError("PDF is password protected")
    blocks, total = [], 0
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        total += len(text.strip())
        blocks.append(("page", str(i)))
        blocks += _lines_to_blocks(text.splitlines())
    if total < 40 * max(1, len(reader.pages)):
        raise ParseError("PDF has almost no extractable text. It looks scanned (images only); OCR is not supported yet. "
                         "Export it as a text PDF or DOCX instead.")
    return blocks


def parse_md(path: Path) -> list[tuple[str, str]]:
    blocks, para = [], []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if re.match(r"^#{1,6}\s", line):
            if para:
                blocks.append(("p", " ".join(para)))
                para = []
            blocks.append(("h", line.lstrip("#").strip()))
        elif not line.strip():
            if para:
                blocks.append(("p", " ".join(para)))
                para = []
        else:
            para.append(line.strip().lstrip("-*").strip())
    if para:
        blocks.append(("p", " ".join(para)))
    return blocks


def parse_txt(path: Path) -> list[tuple[str, str]]:
    return _lines_to_blocks(path.read_text(encoding="utf-8", errors="ignore").splitlines())


PARSERS = {".docx": parse_docx, ".pdf": parse_pdf, ".md": parse_md, ".txt": parse_txt}


# ---------------------------------------------------------------- chunking

def _is_question(text: str) -> bool:
    return (text.endswith("?") and len(text) < 220) or bool(_Q_START.match(text))


def _clean_q(text: str) -> str:
    return re.sub(r"^\s*(q[\.:)]|question[\s:]|\d+[\.)])\s*", "", text, flags=re.I).strip()


def _clean_a(text: str) -> str:
    return re.sub(r"^\s*(a[\.:)]|answer[\s:])\s*", "", text, flags=re.I).strip()


def _windows(words: list[str], size: int, overlap: int = 30) -> list[str]:
    if len(words) <= int(size * 1.4):
        return [" ".join(words)]
    out, step = [], max(1, size - overlap)
    for start in range(0, len(words), step):
        part = words[start:start + size]
        if len(part) < 25 and out:
            out[-1] += " " + " ".join(part)
            break
        out.append(" ".join(part))
        if start + size >= len(words):
            break
    return out


def chunk_blocks(blocks: list[tuple[str, str]], source: str) -> list[Chunk]:
    size = config.CHUNK_WORDS
    stem = Path(source).stem.replace("_", " ")
    n_questions = sum(1 for k, t in blocks if k == "p" and _is_question(t))
    faq_mode = n_questions >= 3

    sections: list[tuple[str, list[str]]] = []   # (title, paragraphs)
    title, paras, heading, page = None, [], None, None

    def close():
        nonlocal title, paras
        if paras:
            sections.append((title or heading or (f"{stem}, page {page}" if page else stem), paras))
        title, paras = None, []

    for kind, text in blocks:
        if kind == "page":
            page = text
            if not faq_mode and not heading:
                close()
            continue
        if kind == "h":
            close()
            heading = text
            continue
        if faq_mode and _is_question(text):
            close()
            title = _clean_q(text)
            paras = [title]
            continue
        paras.append(_clean_a(text) if faq_mode else text)
    close()

    chunks, n = [], 0
    for t, ps in sections:
        body_words = " ".join(ps).split()
        if not body_words:
            continue
        parts = _windows(body_words, size)
        for i, part in enumerate(parts):
            n += 1
            ttl = t if len(parts) == 1 else f"{t} (part {i + 1})"
            chunks.append(Chunk(id=f"{Path(source).stem[:20]}-{n}", title=ttl[:160], text=part, source=source))
    return chunks


def parse_file(path: Path) -> list[Chunk]:
    ext = path.suffix.lower()
    if ext not in PARSERS:
        raise ParseError(f"unsupported file type {ext}; use DOCX, PDF, MD or TXT")
    blocks = PARSERS[ext](path)
    chunks = chunk_blocks(blocks, path.name)
    if not chunks:
        raise ParseError("no text found in the file")
    return chunks


# ---------------------------------------------------------------- versions and search

class KnowledgeBase:
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        if not config.MOCK:
            from sentence_transformers import SentenceTransformer
            try:
                # offline-first: the model is already in the local HF cache, so skip the
                # network HEAD checks at every startup (cuts ~10s from boot time)
                self.model = SentenceTransformer(config.EMBED_MODEL, device=config.DEVICE,
                                                 local_files_only=True)
            except Exception:
                self.model = SentenceTransformer(config.EMBED_MODEL, device=config.DEVICE)
        self.version = None
        self.chunks: list[Chunk] = []
        self.matrix = None
        # v2: BM25 index per active version
        self.bm25 = None
        self._ensure_initial()
        active = store.load_settings().get("active_kb")
        versions = [v["version"] for v in self.list_versions()]
        self.activate(active if active in versions else versions[-1])

    # --- embeddings
    def _embed_docs(self, chunks: list[Chunk]) -> np.ndarray:
        if config.MOCK:
            return np.zeros((len(chunks), 1), dtype=np.float32)
        with self.lock:
            return self.model.encode([f"{c.title}. {c.text}" for c in chunks], normalize_embeddings=True,
                                     batch_size=32, show_progress_bar=False).astype(np.float32)

    # --- v2 BM25
    def _build_bm25(self, chunks: list[Chunk]):
        if not config.V2_RAG or config.MOCK:
            self.bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            _log.warning("rank_bm25 not installed; skipping BM25 index")
            self.bm25 = None
            return
        tokenized = [self._tokenize(c.title + " " + c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]{2,}", text.lower())

    def _bm25_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if not self.bm25 or not query:
            return []
        scores = self.bm25.get_scores(self._tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    # --- v2 query expansion
    def _expand_query(self, query: str) -> str:
        if not config.V2_QUERY_EXPANSION or len(query.split()) >= 6:
            return query
        terms = set(self._tokenize(query))
        synonyms = {}
        if config.V2_SYNONYMS.exists():
            try:
                synonyms = json.loads(config.V2_SYNONYMS.read_text())
            except Exception:
                synonyms = {}
        for t in list(terms):
            for syn in synonyms.get(t, []):
                terms.add(syn.lower())
        # add nearest KB heading terms as expansion context
        for c in self.chunks[:60]:
            for w in self._tokenize(c.title):
                if len(w) > 3:
                    terms.add(w)
        if not terms:
            return query
        # keep original query first, then expansion terms
        expanded = query + ". " + " ".join(sorted(terms))
        return expanded

    # --- v2 hybrid search entry point
    def search(self, query: str, k: int = None, filters: dict | None = None) -> list[tuple[Chunk, float]]:
        k = k or config.TOP_K
        if not query or not self.chunks:
            return []
        if config.MOCK:
            return self._search_mock(query, k)
        effective = self._expand_query(query) if config.V2_QUERY_EXPANSION else query
        if config.V2_RAG:
            return self._search_hybrid(effective, k, filters)
        # fallback to original dense-only path
        return self._search_dense(query, k)

    def _search_dense(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        qv = self.model.encode([self.QUERY_PREFIX + query], normalize_embeddings=True)[0]
        scores = self.matrix @ qv
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]

    def _search_hybrid(self, query: str, k: int, filters: dict | None = None) -> list[tuple[Chunk, float]]:
        # dense
        qv = self.model.encode([self.QUERY_PREFIX + query], normalize_embeddings=True)[0]
        scores = self.matrix @ qv
        dense: list[tuple[Chunk, float]] = []
        order = np.argsort(-scores)
        for i in order:
            c = self.chunks[i]
            if filters and not self._matches_filters(c, filters):
                continue
            dense.append((c, float(scores[i])))
            if len(dense) >= config.V2_TOP_DENSE:
                break
        # bm25
        bm25 = self._bm25_search(query, config.V2_TOP_BM25)
        if filters:
            bm25 = [(c, s) for c, s in bm25 if self._matches_filters(c, filters)][: config.V2_TOP_BM25]
        # RRF
        fused: dict[str, list[float]] = {}
        for rank_i, (c, _) in enumerate(dense):
            fused.setdefault(c.id, []).append(1.0 / (config.V2_RRF_K + rank_i + 1))
        for rank_i, (c, _) in enumerate(bm25):
            fused.setdefault(c.id, []).append(1.0 / (config.V2_RRF_K + rank_i + 1))
        ranked = sorted(fused.items(), key=lambda x: -sum(x[1]))
        out = []
        for cid, _ in ranked:
            c = next((x for x in self.chunks if x.id == cid), None)
            if c is None:
                continue
            out.append((c, 0.0))
            if len(out) >= (config.V2_RERANK_TOP or k):
                break
        # optional re-rank
        if config.V2_RERANK and out:
            out = self._rerank(query, out)
        else:
            # use max(dense, bm25) as the fused score
            dmap = {c.id: s for c, s in dense}
            bmap = {c.id: s for c, s in bm25}
            out = [(c, max(dmap.get(c.id, 0.0), bmap.get(c.id, 0.0))) for c, _ in out]
        return out[:k]

    def _rerank(self, query: str, candidates: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            return candidates
        reranker = getattr(self, "_reranker", None)
        if reranker is None:
            try:
                self._reranker = CrossEncoder(config.EMBED_MODEL, device=config.DEVICE)
            except Exception as e:
                _log.warning("v2 re-ranker init failed: %s", e)
                self._reranker = None
        if not self._reranker:
            return candidates
        pairs = [[query, f"{c.title}. {c.text}"] for c, _ in candidates]
        try:
            rs = self._reranker.predict(pairs)
        except Exception as e:
            _log.warning("v2 rerank predict failed: %s", e)
            return candidates
        return [(c, float(rs[i])) for i, (c, _) in enumerate(candidates)]

    @staticmethod
    def _matches_filters(c: Chunk, filters: dict | None) -> bool:
        if not filters:
            return True
        if "source" in filters and c.source != filters["source"]:
            return False
        if "section" in filters and c.section and c.section != filters["section"]:
            return False
        if "tags" in filters and filters["tags"]:
            if not set(filters["tags"]).intersection(c.tags):
                return False
        return True

    def _search_mock(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        stem = lambda w: w[:-1] if len(w) > 3 and w.endswith("s") else w
        q = {stem(w) for w in re.findall(r"[a-z0-9]+", query.lower())} - _STOP
        if not q:
            return []
        scored = []
        for c in self.chunks:
            words = {stem(w) for w in re.findall(r"[a-z0-9]+", (c.title + " " + c.text).lower())}
            scored.append((c, min(1.0, len(q & words) / min(len(q), 3))))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    # --- staging
    def staged_files(self) -> list[dict]:
        return [{"name": p.name, "size": p.stat().st_size} for p in sorted(STAGING.iterdir()) if p.is_file()]

    def stage(self, filename: str, data: bytes) -> dict:
        name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(filename).name).strip() or "upload"
        if Path(name).suffix.lower() not in ALLOWED:
            raise ParseError(f"{name}: unsupported file type; use DOCX, PDF, MD or TXT")
        if len(data) > 25 * 1024 * 1024:
            raise ParseError(f"{name}: larger than 25 MB")
        (STAGING / name).write_bytes(data)
        return {"name": name, "size": len(data)}

    def unstage(self, name: str) -> bool:
        p = STAGING / Path(name).name
        if p.exists():
            p.unlink()
            return True
        return False

    # --- versions
    def list_versions(self) -> list[dict]:
        out = []
        for d in sorted(KB_DIR.glob("v*")):
            meta = d / "meta.json"
            if d.is_dir() and meta.exists():
                out.append(json.loads(meta.read_text()))
        return out

    def train(self, note: str = "") -> dict:
        files = sorted(p for p in STAGING.iterdir() if p.is_file())
        if not files:
            raise ParseError("no files uploaded; upload at least one DOCX, PDF, MD or TXT file first")
        report, all_chunks = [], []
        for f in files:
            try:
                ch = parse_file(f)
                all_chunks += ch
                report.append({"file": f.name, "chunks": len(ch), "ok": True})
            except ParseError as e:
                report.append({"file": f.name, "chunks": 0, "ok": False, "error": str(e)})
        if not all_chunks:
            raise ParseError("none of the files produced any text: " + "; ".join(r.get("error", "") for r in report))
        t0 = time.perf_counter()
        matrix = self._embed_docs(all_chunks)
        nums = [int(v["version"][1:]) for v in self.list_versions()]
        version = f"v{(max(nums) + 1) if nums else 1:04d}"
        vdir = KB_DIR / version
        (vdir / "sources").mkdir(parents=True)
        for f in files:
            shutil.copy2(f, vdir / "sources" / f.name)
        (vdir / "chunks.json").write_text(json.dumps([asdict(c) for c in all_chunks], indent=1))
        np.save(vdir / "embeddings.npy", matrix)
        meta = {
            "version": version,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files": report,
            "chunks": len(all_chunks),
            "embed_seconds": round(time.perf_counter() - t0, 2),
            "note": note,
        }
        (vdir / "meta.json").write_text(json.dumps(meta, indent=2))
        self.activate(version)
        return meta

    def activate(self, version: str):
        vdir = KB_DIR / version
        if not (vdir / "meta.json").exists():
            raise ParseError(f"version {version} not found")
        chunks = [Chunk(**c) for c in json.loads((vdir / "chunks.json").read_text())]
        matrix = np.load(vdir / "embeddings.npy")
        if not config.MOCK and matrix.shape[1] == 1:   # built in mock mode; re-embed
            matrix = self._embed_docs(chunks)
            np.save(vdir / "embeddings.npy", matrix)
        self.chunks, self.matrix, self.version = chunks, matrix, version
        self._build_bm25(chunks)
        store.save_settings({"active_kb": version})

    def _ensure_initial(self):
        if self.list_versions():
            return
        if not any(STAGING.iterdir()) and config.KNOWLEDGE_FILE.exists():
            shutil.copy2(config.KNOWLEDGE_FILE, STAGING / config.KNOWLEDGE_FILE.name)
        self.train(note="initial sample knowledge base")

    # --- search
    def search(self, query: str, k: int = config.TOP_K) -> list[tuple[Chunk, float]]:
        chunks, matrix = self.chunks, self.matrix
        if not query or not chunks:
            return []
        if config.MOCK:
            stem = lambda w: w[:-1] if len(w) > 3 and w.endswith("s") else w
            q = {stem(w) for w in re.findall(r"[a-z0-9]+", query.lower())} - _STOP
            if not q:
                return []
            scored = []
            for c in chunks:
                words = {stem(w) for w in re.findall(r"[a-z0-9]+", (c.title + " " + c.text).lower())}
                scored.append((c, min(1.0, len(q & words) / min(len(q), 3))))
            scored.sort(key=lambda x: -x[1])
            return scored[:k]
        with self.lock:
            qv = self.model.encode([self.QUERY_PREFIX + query], normalize_embeddings=True)[0]
        scores = matrix @ qv
        order = np.argsort(-scores)[:k]
        return [(chunks[i], float(scores[i])) for i in order]


_STOP = set("a an the is are am do does did i you we it to of for in on at and or what how much can my me your be with this that there any".split())
