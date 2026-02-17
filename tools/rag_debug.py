#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBAPI_ROOT = ROOT / "python" / "webAPI"
if str(WEBAPI_ROOT) not in sys.path:
    sys.path.insert(0, str(WEBAPI_ROOT))



def main() -> int:
    parser = argparse.ArgumentParser(description="Debug single-document RAG pipeline")
    parser.add_argument("--file", required=True, help="Path to source file (.pdf/.docx/.pptx)")
    parser.add_argument("--query", required=True, help="Question for retrieval/answer")
    parser.add_argument("--user", default="rag_debug_user", help="Logical user namespace")
    args = parser.parse_args()

    os.environ.setdefault("RAG_DEBUG", "true")

    from app.models import UserBase
    from app.utils.io_db import DbHelper
    from app.utils.io_file_operation import create_folder_structure, return_user_folder_pdf

    source_path = Path(args.file).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"File not found: {source_path}")

    user = UserBase(id="0", name=args.user, email="rag-debug@local")
    create_folder_structure(user)
    user_pdf_path = Path(return_user_folder_pdf(user))
    target_path = user_pdf_path / source_path.name
    target_path.write_bytes(source_path.read_bytes())

    helper = DbHelper(user)

    print("[1/5] Parsing + chunking")
    chunks = helper.separate_file(str(target_path))
    print(f"chunks: {len(chunks)}")

    print("[2/5] Embeddings init")
    embeddings = helper.get_embeddings()
    print(f"embeddings class: {embeddings.__class__.__name__}")

    print("[3/5] Indexing into local vector DB")
    ok = helper.put_vector_in_db(chunks, embeddings)
    print(f"indexing status: {ok}")

    print("[4/5] Retrieval")
    vectordb = helper.get_vectror_db(embeddings)
    retrieved = vectordb.similarity_search_with_score(args.query, k=int(os.getenv("RAG_RETRIEVAL_K", "5")))
    print(f"retrieved: {len(retrieved)}")
    for i, (doc, score) in enumerate(retrieved[:5]):
        snippet = doc.page_content.replace("\n", " ")[:180]
        print(f"  - #{i} score={score:.4f} source={doc.metadata.get('source')} chunk={doc.metadata.get('chunk')} text={snippet}")

    print("[5/5] LLM answer")
    answer = helper.get_answer(args.query)
    print("answer:")
    print(answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
