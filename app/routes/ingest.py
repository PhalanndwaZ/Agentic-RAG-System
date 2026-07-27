import io
import logging
import os

from fastapi import APIRouter, HTTPException, Request, UploadFile
from pypdf import PdfReader

from app.chunking import chunk_text
from app.config import get_settings
from app.schemas import FolderIngestRequest, FolderIngestResponse, IngestResponse

router = APIRouter()
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")


def _extract_text(filename: str, raw_bytes: bytes) -> str:
    # PDFs need actual parsing to pull text out of pages; anything else
    # (.txt, .md) is treated as plain text and just decoded.
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw_bytes))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = raw_bytes.decode("utf-8", errors="ignore")

    # Postgres text columns reject NUL bytes outright; some PDFs embed them
    # due to encoding quirks in the source document. Strip them here, once,
    # so nothing downstream (chunking, storage) has to worry about it.
    return text.replace("\x00", "")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: Request, file: UploadFile) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename")

    settings = get_settings()
    raw_bytes = await file.read()

    text = _extract_text(file.filename, raw_bytes)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in file")

    chunks = chunk_text(text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced zero chunks")

    embeddings = request.app.state.embedder.embed(chunks)

    store = request.app.state.vectorstore
    document_id = store.add_document(source=file.filename)
    chunks_created = store.add_chunks(document_id, chunks, embeddings)

    return IngestResponse(
        document_id=document_id,
        source=file.filename,
        chunks_created=chunks_created,
    )


@router.post("/ingest-folder", response_model=FolderIngestResponse)
async def ingest_folder(request: Request, body: FolderIngestRequest) -> FolderIngestResponse:
    settings = get_settings()

    if not os.path.isdir(body.folder_path):
        raise HTTPException(status_code=400, detail=f"Not a valid directory: {body.folder_path}")

    embedder = request.app.state.embedder
    store = request.app.state.vectorstore

    files_processed = 0
    total_chunks_created = 0
    skipped: list[str] = []

    for filename in os.listdir(body.folder_path):
        if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
            continue

        file_path = os.path.join(body.folder_path, filename)
        try:
            with open(file_path, "rb") as f:
                raw_bytes = f.read()

            text = _extract_text(filename, raw_bytes)
            if not text.strip():
                skipped.append(filename)
                continue

            chunks = chunk_text(text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
            if not chunks:
                skipped.append(filename)
                continue

            embeddings = embedder.embed(chunks)
            document_id = store.add_document(source=filename)
            chunks_created = store.add_chunks(document_id, chunks, embeddings)

            files_processed += 1
            total_chunks_created += chunks_created
        except Exception as e:
            # One bad file (corrupted PDF, permission error, etc.) shouldn't
            # kill the whole batch — log the real reason, then keep going.
            logger.error(f"Failed to ingest {filename}: {e}")
            skipped.append(filename)

    return FolderIngestResponse(
        files_processed=files_processed,
        total_chunks_created=total_chunks_created,
        skipped=skipped,
    )