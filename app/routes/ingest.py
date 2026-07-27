import io

from fastapi import APIRouter, HTTPException, Request, UploadFile
from pypdf import PdfReader

from app.chunking import chunk_text
from app.config import get_settings
from app.schemas import IngestResponse

router = APIRouter()


def _extract_text(filename: str, raw_bytes: bytes) -> str:
    # PDFs need actual parsing to pull text out of pages; anything else
    # (.txt, .md) is treated as plain text and just decoded.
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw_bytes))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return raw_bytes.decode("utf-8", errors="ignore")


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