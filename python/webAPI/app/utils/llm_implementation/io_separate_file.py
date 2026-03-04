# имена классов должны начинаться с sf_
# и отображать основные характеристики (по усмотрению разработчика)
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import os, re, inspect, hashlib
from pathlib import Path
import fitz
from abc import ABC, abstractmethod
from langchain_core.documents import Document as LangDocument
from tabulate import tabulate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils.rag_artifacts import (
    rag_artifacts,
    serialize_documents,
    combine_page_content,
)


class sf_DataProcessing_keywords_512_chunk_and_Tables:
    """Точка входа парсинга файлов и подготовки чанков для RAG ingestion."""

    def __init__(self, file_path):
        self.file_path = file_path
        self.source = str(Path(file_path).resolve())
        self.doc_id = self._build_doc_id(self.source)
        self.chunk_size = self._env_int("RAG_CHUNK_SIZE_CHARS", 2400)
        self.chunk_overlap = self._env_int("RAG_CHUNK_OVERLAP_CHARS", 300)
        self.table_window_threshold = self._env_int("RAG_TABLE_WINDOW_THRESHOLD", 50)
        self.table_window_size = self._env_int("RAG_TABLE_WINDOW_SIZE", 20)
        self.table_summary_preview_rows = self._env_int("RAG_TABLE_SUMMARY_PREVIEW_ROWS", 3)
        self.max_header_rows = self._env_int("RAG_TABLE_MAX_HEADER_ROWS", 2)
        # Разделитель используется только для текстовых абзацев.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _env_int(self, name: str, default: int) -> int:
        """Читает числовой параметр из env с безопасным fallback."""
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = int(raw)
            return value if value > 0 else default
        except Exception:
            return default

    def _build_doc_id(self, source_path: str) -> str:
        """Строит стабильный doc_id по имени файла и абсолютному пути."""
        stem = re.sub(r"[^0-9A-Za-zА-Яа-я_-]+", "_", Path(source_path).stem).strip("_") or "doc"
        digest = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:8]
        return f"{stem}_{digest}"

    def _bbox_to_rect(self, bbox: tuple[float, float, float, float], pad: float = 5) -> "fitz.Rect":
        """Преобразует bbox таблицы в прямоугольник PyMuPDF с небольшим отступом."""
        x1, y1, x2, y2 = bbox
        rect = fitz.Rect(x1, y1, x2, y2)
        if hasattr(rect, "inflate"):
            return rect.inflate(pad, pad)
        return fitz.Rect(x1 - pad, y1 - pad, x2 + pad, y2 + pad)

    def _normalize_cell(self, value: str) -> str:
        """Нормализует содержимое ячейки таблицы для семантического поиска."""
        text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"\s*([,;:])\s*", r"\1 ", text)
        return text.strip()

    def clean_text(self, text: str) -> str:
        """Первичная очистка текста от служебных символов и маркеров."""
        cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        cleaned = cleaned.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")
        # Удаляем устаревшие маркеры таблиц, чтобы не отправлять их в embedding.
        cleaned = cleaned.replace("[TABLE_START]", "").replace("[TABLE_END]", "")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        return cleaned

    def normalize_text(self, text: str) -> str:
        """Нормализует переносы, дубликаты строк и форматирование абзацев."""
        prepared = self.clean_text(text)
        raw_lines = prepared.split("\n")
        lines: list[str] = []
        for line in raw_lines:
            # Сохраняем структуру абзацев, но убираем избыточные пробелы внутри строк.
            normalized_line = re.sub(r"\s{2,}", " ", line).strip()
            lines.append(normalized_line)

        # Удаляем подряд идущие одинаковые строки (часто повторяются в колонтитулах).
        deduped: list[str] = []
        prev = None
        for line in lines:
            if line and line == prev:
                continue
            deduped.append(line)
            prev = line if line else prev

        # Восстанавливаем разделение заголовков и основного текста.
        rebuilt: list[str] = []
        for idx, line in enumerate(deduped):
            rebuilt.append(line)
            if not line:
                continue
            if idx + 1 < len(deduped):
                nxt = deduped[idx + 1]
                if self._looks_like_heading(line) and nxt and not self._looks_like_heading(nxt):
                    rebuilt.append("")

        text_out = "\n".join(rebuilt)
        text_out = re.sub(r"\n{3,}", "\n\n", text_out)
        return text_out.strip()

    def _looks_like_heading(self, line: str) -> bool:
        """Эвристика для отделения заголовков от основного текста."""
        if len(line) < 3 or len(line) > 120:
            return False
        letters = re.sub(r"[^A-Za-zА-Яа-яЁё]", "", line)
        if not letters:
            return False
        return letters.isupper()

    def _normalize_row_width(self, rows: list[list[str]]) -> list[list[str]]:
        """Приводит все строки таблицы к одинаковой ширине по числу колонок."""
        if not rows:
            return []
        max_cols = max(len(row) for row in rows)
        normalized = []
        for row in rows:
            padded = row + [""] * (max_cols - len(row))
            normalized.append([self._normalize_cell(cell) for cell in padded])
        return normalized

    def _detect_header_depth(self, rows: list[list[str]]) -> int:
        """Определяет глубину шапки таблицы (1..max_header_rows)."""
        if not rows:
            return 1
        max_depth = min(self.max_header_rows, len(rows))
        depth = 1
        for idx in range(max_depth):
            row = rows[idx]
            non_empty = [c for c in row if c.strip()]
            if not non_empty:
                continue
            numeric_like = sum(1 for c in non_empty if re.fullmatch(r"[\d\s.,:/-]+", c))
            # Если строка в основном текстовая, считаем ее частью заголовка.
            if numeric_like <= max(1, len(non_empty) // 2):
                depth = idx + 1
        return max(1, depth)

    def _build_headers(self, header_rows: list[list[str]]) -> list[str]:
        """Нормализует многоуровневую шапку и делает имена колонок уникальными."""
        if not header_rows:
            return ["col_1"]
        cols = len(header_rows[0])
        levels: list[list[str]] = [[] for _ in range(cols)]
        for row in header_rows:
            for col in range(cols):
                value = (row[col] if col < len(row) else "").strip()
                if value:
                    levels[col].append(value)

        headers = []
        for idx, parts in enumerate(levels, start=1):
            name = " > ".join(parts).strip(" >")
            headers.append(name or f"col_{idx}")

        # Устраняем дубли после нормализации шапки.
        seen: dict[str, int] = {}
        unique_headers: list[str] = []
        for header in headers:
            cnt = seen.get(header, 0) + 1
            seen[header] = cnt
            unique_headers.append(header if cnt == 1 else f"{header}_{cnt}")
        return unique_headers

    def _carry_forward_empty_cells(self, rows: list[list[str]]) -> list[list[str]]:
        """Протягивает значения сверху для пустых ячеек (merged cells)."""
        if not rows:
            return []
        carried = [row[:] for row in rows]
        for r_idx in range(1, len(carried)):
            for c_idx in range(len(carried[r_idx])):
                if carried[r_idx][c_idx].strip():
                    continue
                carried[r_idx][c_idx] = carried[r_idx - 1][c_idx]
        return carried

    def _hash_header(self, headers: list[str]) -> str:
        """Возвращает стабильный hash нормализованной шапки таблицы."""
        normalized = "|".join(h.strip().lower() for h in headers)
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]

    def _format_row_semantic(self, headers: list[str], row: list[str], table_title: str, section: str) -> str:
        """Собирает строку таблицы в семантический текст без ASCII-таблиц."""
        pairs = []
        for col, value in zip(headers, row):
            val = self._normalize_cell(value)
            if not val:
                continue
            pairs.append(f"{col}: {val}")
        row_text = "; ".join(pairs) if pairs else "пустая строка"
        return self.normalize_text(
            f"Table: {table_title}\nContext: {section or 'n/a'}\nRow: {row_text}"
        )

    def _build_table_chunks(
        self,
        table_rows: list[list[str]],
        table_title: str,
        section: str,
        table_seq: int,
    ) -> list[dict]:
        """Формирует чанки таблиц: summary + row/window."""
        normalized_rows = self._carry_forward_empty_cells(self._normalize_row_width(table_rows))
        if not normalized_rows:
            return []
        header_depth = self._detect_header_depth(normalized_rows)
        header_rows = normalized_rows[:header_depth]
        data_rows = normalized_rows[header_depth:] if len(normalized_rows) > header_depth else []
        headers = self._build_headers(header_rows)
        data_rows = self._carry_forward_empty_cells(data_rows)
        header_signature = self._hash_header(headers)
        table_id = f"{self.doc_id}_table_{table_seq}_{header_signature[:8]}"
        section_path = section or "root"

        # summary: короткий обзор таблицы с колонками и первыми строками.
        preview_rows = data_rows[: self.table_summary_preview_rows]
        preview_payload = []
        for row in preview_rows:
            preview_payload.append("; ".join(f"{h}: {self._normalize_cell(v)}" for h, v in zip(headers, row) if self._normalize_cell(v)))
        summary_text = self.normalize_text(
            f"Table: {table_title}\n"
            f"Context: {section_path}\n"
            f"Columns: {', '.join(headers)}\n"
            f"Rows total: {len(data_rows)}\n"
            f"Preview:\n" + ("\n".join(preview_payload) if preview_payload else "n/a")
        )
        chunks: list[dict] = [
            {
                "page_content": summary_text,
                "metadata": {
                    "type": "table_summary",
                    "content_type": "table",
                    "table_id": table_id,
                    "table_title": table_title,
                    "header_signature": header_signature,
                    "section_path": section_path,
                    "section": section_path,
                },
            }
        ]

        if len(data_rows) > self.table_window_threshold:
            # Для больших таблиц сохраняем окна строк, чтобы не делать огромный единый чанк.
            window_size = max(1, self.table_window_size)
            for start in range(0, len(data_rows), window_size):
                end = min(start + window_size, len(data_rows))
                lines = []
                for row_idx, row in enumerate(data_rows[start:end], start=start + 1):
                    row_text = self._format_row_semantic(headers, row, table_title, section_path)
                    lines.append(f"Row {row_idx}: {row_text.split('Row: ', 1)[-1]}")
                content = self.normalize_text(
                    f"Table: {table_title}\n"
                    f"Context: {section_path}\n"
                    f"Header: {', '.join(headers)}\n"
                    f"Rows: {start + 1}-{end}\n" + "\n".join(lines)
                )
                chunks.append(
                    {
                        "page_content": content,
                        "metadata": {
                            "type": "table_rows_window",
                            "content_type": "table",
                            "table_id": table_id,
                            "table_title": table_title,
                            "header_signature": header_signature,
                            "row_range": f"{start + 1}-{end}",
                            "section_path": section_path,
                            "section": section_path,
                        },
                    }
                )
        else:
            for row_index, row in enumerate(data_rows, start=1):
                content = self._format_row_semantic(headers, row, table_title, section_path)
                chunks.append(
                    {
                        "page_content": content,
                        "metadata": {
                            "type": "table_row",
                            "content_type": "table",
                            "table_id": table_id,
                            "table_title": table_title,
                            "header_signature": header_signature,
                            "row_index": row_index,
                            "section_path": section_path,
                            "section": section_path,
                        },
                    }
                )
        return chunks

    def _new_metadata(self, chunk_index: int, chunk_type: str, section: str) -> dict:
        """Готовит унифицированный минимум metadata для любого чанка."""
        content_type = "table" if str(chunk_type).startswith("table") else "paragraph"
        if chunk_type == "error":
            content_type = "error"
        return {
            "doc_id": self.doc_id,
            "chunk_index": chunk_index,
            "chunk_id": f"{self.doc_id}_{chunk_index}",
            "source": self.source,
            "type": chunk_type,
            "content_type": content_type,
            "section": section or "root",
        }

    def _parse_docx(self) -> list[dict]:
        """Извлекает абзацы и таблицы из DOCX с сохранением структуры."""
        from docx import Document as DocxDocument
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        doc = DocxDocument(self.file_path)
        blocks: list[dict] = []
        current_section = "root"
        table_seq = 0
        for child in doc.element.body:
            tag = child.tag.split("}")[-1]
            if tag == "p":
                paragraph = Paragraph(child, doc)
                text = paragraph.text or ""
                if not text.strip():
                    continue
                style_name = (paragraph.style.name if paragraph.style is not None else "") or ""
                if style_name.lower().startswith("heading"):
                    current_section = self.normalize_text(text)
                blocks.append(
                    {
                        "kind": "paragraph",
                        "text": text,
                        "section": current_section,
                    }
                )
            elif tag == "tbl":
                table_seq += 1
                table = Table(child, doc)
                rows = []
                for row in table.rows:
                    rows.append([(cell.text or "") for cell in row.cells])
                blocks.append(
                    {
                        "kind": "table",
                        "rows": rows,
                        "section": current_section,
                        "table_title": f"Таблица {table_seq}",
                    }
                )
        return blocks

    def _parse_pdf(self) -> list[dict]:
        """Извлекает абзацы и таблицы из PDF без OCR."""
        import camelot

        blocks: list[dict] = []
        pdf = fitz.open(self.file_path)
        table_seq = 0
        try:
            for page in pdf:
                page_num = page.number + 1
                section = f"page_{page_num}"
                table_rects: list[fitz.Rect] = []
                page_tables = []
                try:
                    page_tables = camelot.read_pdf(
                        self.file_path,
                        pages=str(page_num),
                        flavor="stream",
                        strip_text="\n",
                        split_text=True,
                    )
                    for table in page_tables:
                        table_rects.append(self._bbox_to_rect(table._bbox))
                except Exception as exc:
                    print(f"[PDF] таблицы не извлечены page={page_num}: {exc}")
                    page_tables = []

                # Сохраняем текст вне областей таблиц, чтобы не дублировать контент.
                text_chunks: list[str] = []
                for x0, y0, x1, y1, txt, *_ in page.get_text("blocks"):
                    if not (txt or "").strip():
                        continue
                    rect = fitz.Rect(x0, y0, x1, y1)
                    if any(rect.intersects(tbl_rect) for tbl_rect in table_rects):
                        continue
                    text_chunks.append(txt)
                page_text = self.normalize_text("\n".join(text_chunks))
                if page_text:
                    blocks.append(
                        {
                            "kind": "paragraph",
                            "text": page_text,
                            "section": section,
                        }
                    )

                for table in page_tables:
                    if table.df.empty:
                        continue
                    table_seq += 1
                    rows = table.df.fillna("").astype(str).values.tolist()
                    blocks.append(
                        {
                            "kind": "table",
                            "rows": rows,
                            "section": section,
                            "table_title": f"Таблица {table_seq}",
                        }
                    )
        finally:
            pdf.close()
        return blocks

    def _parse_pptx(self) -> list[dict]:
        """Извлекает текст и таблицы из PPTX."""
        from pptx import Presentation

        blocks: list[dict] = []
        prs = Presentation(self.file_path)
        table_seq = 0
        for slide_idx, slide in enumerate(prs.slides, start=1):
            section = f"slide_{slide_idx}"
            text_lines: list[str] = []
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False) and shape.text:
                    text_lines.append(shape.text)
                if getattr(shape, "has_table", False):
                    table_seq += 1
                    rows = []
                    for row in shape.table.rows:
                        rows.append([cell.text or "" for cell in row.cells])
                    blocks.append(
                        {
                            "kind": "table",
                            "rows": rows,
                            "section": section,
                            "table_title": f"Таблица {table_seq}",
                        }
                    )
            slide_text = self.normalize_text("\n".join(text_lines))
            if slide_text:
                blocks.append(
                    {
                        "kind": "paragraph",
                        "text": slide_text,
                        "section": section,
                    }
                )
        return blocks

    def _parse_blocks(self) -> list[dict]:
        """Определяет формат файла и извлекает структурированные блоки."""
        ext = Path(self.file_path).suffix.lower()
        if ext == ".docx":
            return self._parse_docx()
        if ext == ".pdf":
            return self._parse_pdf()
        if ext == ".pptx":
            return self._parse_pptx()
        raise ValueError(f"Неподдерживаемый формат файла: {ext}")

    def _split_text_chunks(self, text: str) -> list[str]:
        """Разбивает нормализованный текст на чанки ограниченного размера."""
        if not text:
            return []
        return [self.normalize_text(chunk) for chunk in self.text_splitter.split_text(text) if self.normalize_text(chunk)]

    def separate_file(self):
        """Полный пайплайн clean_text -> normalize_text -> chunk_text."""
        diag_enabled = rag_artifacts.has_session("UP")
        # Сохраняем артефакты только когда DEBUG_RAG включен.
        try:
            if diag_enabled:
                # Фиксируем исходный файл и параметры обработки.
                rag_artifacts.copy_file("UP", "input", self.file_path)
                rag_artifacts.write_json(
                    "UP",
                    "meta",
                    "document_meta.json",
                    {
                        "file_path": self.file_path,
                        "parser_class": self.__class__.__name__,
                        "doc_id": self.doc_id,
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                    },
                )

            blocks = self._parse_blocks()
            if diag_enabled:
                rag_artifacts.write_json("UP", "parsed", "documents.json", blocks)
                parsed_text = "\n\n".join(
                    self.normalize_text(item.get("text", ""))
                    for item in blocks
                    if item.get("kind") == "paragraph"
                )
                rag_artifacts.write_text("UP", "parsed", "raw_text.txt", parsed_text)

            final_docs: list[LangDocument] = []
            chunk_index = 0
            table_seq = 0
            for block in blocks:
                kind = block.get("kind")
                section = block.get("section", "root")
                if kind == "paragraph":
                    normalized_text = self.normalize_text(block.get("text", ""))
                    for chunk_text in self._split_text_chunks(normalized_text):
                        metadata = self._new_metadata(chunk_index, "paragraph", section)
                        final_docs.append(LangDocument(page_content=chunk_text, metadata=metadata))
                        chunk_index += 1
                elif kind == "table":
                    table_seq += 1
                    table_chunks = self._build_table_chunks(
                        table_rows=block.get("rows", []),
                        table_title=block.get("table_title", f"Таблица {table_seq}"),
                        section=section,
                        table_seq=table_seq,
                    )
                    for item in table_chunks:
                        metadata = self._new_metadata(chunk_index, item["metadata"].get("type", "table"), section)
                        metadata.update(item["metadata"])
                        final_docs.append(
                            LangDocument(
                                page_content=self.normalize_text(item["page_content"]),
                                metadata=metadata,
                            )
                        )
                        chunk_index += 1

            if len(final_docs) == 0:
                metadata = self._new_metadata(0, "error", "root")
                metadata["error"] = "no_text_or_tables"
                final_docs = [
                    LangDocument(
                        page_content="ERROR: файл содержит только изображения или не содержит текста",
                        metadata=metadata,
                    )
                ]

            if diag_enabled:
                rag_artifacts.write_json("UP", "chunks", "chunks.json", serialize_documents(final_docs))
                rag_artifacts.write_text("UP", "chunks", "chunks.txt", combine_page_content(final_docs))
            return final_docs
        except Exception as exc:
            if diag_enabled:
                rag_artifacts.log_exception(
                    "UP",
                    "io_separate_file.separate_file",
                    exc,
                    {"file_path": self.file_path},
                )
            raise

################################
################################ Вспомогательные классы
################################

def fix_broken_tables(chunks):
    i = 0
    fixed = []
    while i < len(chunks):
        doc = chunks[i]

        # Если началась таблица, но не закончилась
        if "[TABLE_START]" in doc.page_content and "[TABLE_END]" not in doc.page_content:
            new_content = doc.page_content
            j = i + 1
            while j < len(chunks):
                new_content += chunks[j].page_content
                if "[TABLE_END]" in chunks[j].page_content:
                    break
                j += 1
            # Соединяем
            fixed.append(LangDocument(
                page_content=new_content,
                metadata=doc.metadata
            ))
            i = j + 1
        else:
            fixed.append(doc)
            i += 1
    return fixed

class SmartTextSplitter:
    def __init__(self, default_chunk_size=512, overlap=100):
        self.default_chunk_size = default_chunk_size
        self.overlap = overlap

    def split_documents(self, documents):
        result = []
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata

            # Если это таблица — добавляем как есть
            if metadata.get("type") == "table":
                result.append(doc)
            else:
                # Объединяем текст без разрыва по строкам
                clean_text = " ".join(content.split())
                chunks = self._split_text_by_chunk_size(clean_text)
                for i, chunk in enumerate(chunks):
                    new_metadata = metadata.copy()
                    new_metadata["chunk"] = i
                    result.append(LangDocument(page_content=chunk, metadata=new_metadata))
        return result

    def _split_text_by_chunk_size(self, text: str):
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.default_chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.default_chunk_size - self.overlap
        return chunks

# Фабрика загрузчиков, которая определяет тип файла и возвращает нужный загрузчик
class sfDocumentLoaderFactory:
    @staticmethod
    def create_loader(file_path: str):
        ext = sfFileTypeDetector.get_file_type(file_path)
        if ext == ".pdf":
            return sfPDFLoader(file_path)
        elif ext == ".docx":
            return sfDOCXLoader(file_path, loader_type="python-docx")
        elif ext == ".pptx":  
            return sfPPTXLoader(file_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {ext}")

#  Конвейер обработки документа: загрузка -> разделение на чанки
class sfDocumentProcessingPipeline:
    def __init__(self, file_path: str):
        self.loader = sfDocumentLoaderFactory.create_loader(file_path)
        # self.splitter = sfTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    def separate_file(self):
        documents = self.loader.load_documents()
        return self.splitter.split(documents)

class sfBaseDocumentLoader(ABC):
    @abstractmethod
    def load_documents(self) -> list:
        # Метод должен вернуть список объектов LangDocument
        pass

# Класс для определения расширения файла
class sfFileTypeDetector:
    @staticmethod
    def get_file_type(file_path: str) -> str:
        _, ext = os.path.splitext(file_path)
        return ext.lower()

class sfDOCXLoader(sfBaseDocumentLoader):
    def __init__(self, file_path: str, loader_type: str):
        self.file_path = file_path
        self.loader_type = loader_type

    def clean_text(self, text:str) -> str:
        import re
        invisible_chars = ['\u200b', '\ufeff', '\xa0', '\x0c']
        for char in invisible_chars:
            text = text.replace(char, ' ' if char == '\xa0' else '')
        text = re.sub(r'([\-=_*~#]{3,})', '', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n', text)
        text = "\n".join([line.strip() for line in text.splitlines()])
        return text.strip()

    def extract_blocks(self):
        from docx import Document
        from docx.table import Table
        doc = Document(self.file_path)
        blocks = []
        for child in doc.element.body:
            tag = child.tag.split('}')[-1]
            if tag == 'p':
                texts = [n.text for n in child.iter() if n.tag.endswith('}t') and n.text]
                text = self.clean_text(''.join(texts))
                if text:
                    blocks.append(('paragraph', text))
            elif tag == 'tbl':
                tbl = Table(child, doc)
                headers = []
                rows = []
                for i, row in enumerate(tbl.rows):
                    cells = [self.clean_text(cell.text) for cell in row.cells]
                    if i == 0:
                        headers = cells  # Первая строка — заголовки
                    else:
                        rows.append(cells)
                if not headers or not rows:
                    continue

                # Преобразуем в Markdown-таблицу
                markdown_table = tabulate(rows, headers=headers, tablefmt='github')

                # Добавляем разделители, чтобы таблица не разрывалась на чанки
                full_table = "[TABLE_START]\n" + markdown_table + "\n[TABLE_END]"

                blocks.append(('table', full_table))
        return blocks

    def extract_tables(self):
        try:
            from docx2python import docx2python
            result = docx2python(self.file_path)
            table_docs = []
            table_idx = 1
            for section in result.body:
                for element in section:
                    if isinstance(element, list) and element and all(isinstance(row, list) for row in element):
                        table_rows = []
                        for row in element:
                            clean_cells = [cell.strip().replace('\n', ' ') for cell in row]
                            if any(clean_cells):
                                table_rows.append(clean_cells)
                        if not table_rows:
                            continue
                        header = table_rows[0]
                        data_rows = table_rows[1:]
                        lines = ["\t".join(header)]
                        for row in data_rows:
                            lines.append("\t".join(row))
                        table_text = f"Таблица {table_idx}:\n" + "\n".join(lines)
                        table_docs.append(table_text)
                        table_idx += 1
            tables_text = "\n".join(table_docs)
            table_metadata = {"source": os.path.basename(self.file_path)}
            print("\n=== Итоговый результат для RAG ===\n")
            print(tables_text)
            return tables_text, table_metadata
        except Exception as e:
            print(f"Ошибка при обработке таблиц ({self.loader_type}): {e}")
            import traceback
            traceback.print_exc()
            return []

    def load_documents(self):
        try:
            docs = []
            blocks = self.extract_blocks()

            # Список для хранения обычного текста
            current_text = []

            for kind, text in blocks:
                if kind == "table":
                    # Если был накопленный текст — добавляем его как отдельный документ
                    if current_text:
                        full_text = "\n".join(current_text)
                        docs.append(LangDocument(
                            page_content=full_text,
                            metadata={"source": os.path.basename(self.file_path), "type": "paragraph"}
                        ))
                        current_text = []

                    # Добавляем таблицу как отдельный документ
                    docs.append(LangDocument(
                        page_content=text,
                        metadata={"source": os.path.basename(self.file_path), "type": "table"}
                    ))
                else:
                    # Накапливаем обычный текст
                    current_text.append(text)

            # Не забываем про оставшийся текст после последней таблицы
            if current_text:
                full_text = "\n".join(current_text)
                docs.append(LangDocument(
                    page_content=full_text,
                    metadata={"source": os.path.basename(self.file_path), "type": "paragraph"}
                ))

        except Exception as e:
            print(f"Ошибка при загрузке документа: {e}")
            return [LangDocument(
                page_content="ERROR: файл содержит только изображения или не содержит текста",
                metadata={"source": self.file_path, "type": "error", "error": "no_text_or_tables"})]

        return docs

class sfPDFLoader(sfBaseDocumentLoader):
    def __init__(
            self, 
            file_path: str, 
            flavor: str = "stream",
            hf_k: int=5, 
            hf_thr_ratio: float = 0.9 
            ):
        self.file_path = file_path
        self.flavor = flavor 
        self.k = hf_k
        self.thr = hf_thr_ratio

    def _clean(self, text: str) -> str:
        invisible = ["\u200b", "\ufeff", "\xa0", "\x0c"]
        for ch in invisible:
            text = text.replace(ch, " " if ch == "\xa0" else "")
        text = re.sub(r"[\t ]+", " ", text)
        text = re.sub(r"\n{3,}", "\n", text)
        return text.strip()

    def _bbox_to_rect(self, bbox: tuple[float, float, float, float], pad: float = 5) -> "fitz.Rect":
        x1, y1, x2, y2 = bbox 
        rect = fitz.Rect(x1, y1, x2, y2)
        if hasattr(rect, "inflate"):
            return rect.inflate(pad, pad)
        return fitz.Rect(x1 - pad, y1 - pad, x2 + pad, y2 + pad)

    @staticmethod
    def _norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"\d{1,4}[./-]\d{1,2}[./-]?\d{0,4}", "", s)
        s = re.sub(r"\d+", "", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def _collect_hf_candidates(self, pdf: fitz.Document) -> tuple[set[str], set[str]]:
        k = self.k
        thr_ratio = self.thr
        head_cnt: dict[str, int] = {} 
        foot_cnt: dict[str, int] = {}
        for page in pdf:
            blocks = sorted(page.get_text("blocks"), key=lambda b: b[1])
            top_lines = [blocks[i][4].strip() for i in range(min(k, len(blocks)))]
            bot_lines = [blocks[-(i + 1)][4].strip() for i in range(min(k, len(blocks)))]
            for ln in top_lines:
                key = self._norm(ln)
                if key:
                    head_cnt[key] = head_cnt.get(key, 0) + 1
            for ln in bot_lines:
                key = self._norm(ln)
                if key:
                    foot_cnt[key] = foot_cnt.get(key, 0) + 1
        thresh = max(1, int(len(pdf) * thr_ratio)) 
        headers_set = {s for s, c in head_cnt.items() if c >= thresh}
        footers_set = {s for s, c in foot_cnt.items() if c >= thresh}
        return headers_set, footers_set

    def load_documents(self) -> list[LangDocument]:
        import camelot
        docs: list[LangDocument] = []
        pdf = fitz.open(self.file_path)
        file_name = os.path.basename(self.file_path)
        hdr_set, ftr_set = self._collect_hf_candidates(pdf)
        table_idx = 1
        for page in pdf:
            page_id = page.number + 1
            try:
                tables = camelot.read_pdf(
                    self.file_path,
                    pages=str(page_id),
                    flavor=self.flavor,
                    strip_text="\n",
                    split_text=True,
                    edge_tol=200,
                    row_tol=5,
                )
            except Exception as e:
                print(f"[Camelot] ошибка при разборке {page_id}: {e}")
                tables = []
            tbl_rects: list[fitz.Rect] = []
            for t in tables:
                tbl_rects.append(self._bbox_to_rect(t._bbox))
            try:
                blocks = page.get_text("blocks")
                text_parts: list[str] = []
                for x0, y0, x1, y1, txt, *_ in blocks:
                    rect = fitz.Rect(x0, y0, x1, y1)
                    if not txt.strip():
                        continue
                    if page_id != 1 and any(
                        self._norm(line) in hdr_set or self._norm(line) in ftr_set
                        for line in txt.splitlines()
                        if line.strip()
                    ):
                        continue
                    text_parts.append(txt)
            except Exception as e:
                print(e)
            if text_parts:
                para_text = self._clean("\n".join(text_parts))
                is_guess = ("\t" in para_text and para_text.count("\t") >= 3) \
                            or ("..." in para_text and para_text.count("...") >= 5)
                docs.append(
                    LangDocument(
                        page_content=para_text,
                        metadata={
                            "source": file_name,
                            "type": "table_guess" if is_guess else "paragraph",
                            "page": page_id,
                        },
                    )
                )
            tables = tables or []
            table_order = sorted(zip(tbl_rects, tables), key=lambda p: p[0].y0)
            for rect, table in table_order:
                if table.df.shape[0] < 3 or table.df.shape[1] < 2:
                    continue
                first_cell_norm = self._norm(table.df.iloc[0, 0])
                # ✅ Изменение: таблица в формате Markdown внутри разделителя
                markdown_table = tabulate(table.df.values.tolist(), headers=table.df.columns.tolist(), tablefmt="github")
                content = f"[TABLE_START]\nТаблица {table_idx}:\n{markdown_table}\n[TABLE_END]"
                docs.append(
                    LangDocument(
                        page_content=content,
                        metadata={
                            "source": file_name,
                            "type": "table",
                            "page": page_id,
                            "rows": table.df.shape[0],
                            "cols": table.df.shape[1],
                        },
                    )
                )
                table_idx += 1
        pdf.close()
        return docs

class sfPPTXLoader(sfBaseDocumentLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_documents(self):
        from langchain_community.document_loaders import UnstructuredPowerPointLoader
        loader = UnstructuredPowerPointLoader(self.file_path)
        documents = loader.load()  # <-- ВАЖНО: вызываем .load()
        return documents

class get_keywords:
    import base64
    from app.config import settings_llm, settings
    from typing import List
    from langchain_ollama import OllamaLLM
    promt_category = {
        "Закон или нормативный акт": (
            {"Номер закона или нормативного акта " : "Какой номер документа?",
             "Наименование закона или нормативного акта " : "Какое наименование документа?",
             "Дата закона или нормативного акта " : "Какая дата документа?"}
        ),
        "Договор": (
            {"Номер договора ":"Какой номер договора?",
             "Дата заключения договора ":"Какая дата договора?",
             "Договор заключен между ":" Между кем заключен договор?",
             "Предмет договора ":"Какой предмет договора?"}
        ),
        "Письмо": (
            {"Письмо от ":"Кто написал письмо?",
            "Дата письма ":"Какая дата письма?",
            "Получатель письма " : "Кто получатель письма?",
            "Тема письма " : "Какая тема письма ?"}
        ),
        "Курсовая или дипломная работа": (
            {"Тема курсовой или дипломной работы ":"Какая тема работы?",
            "Автор курсовой или дипломной работы ":"Кто подготовил работу?"}
        ),
        "Информация об организации": (
            {"Наименование организации ":" Какое наименование организации?"}
        ),
        "Техническое задание или технический проект": (
            {"Наименование технического задания ":"Какое наименование документа?",
            "Номер технического задания ":"Какой номер документа?",
            "Автор технического задания ": "Кто подготовил документ?"}
        ),
        "Рассказ или повесть": (
            {"Автор художественного произведения ": "Кто автор документа?", 
            "Название произведения": "Какое название документа?"}
        ),
        "Неизвестная категория": (
            {}
        )
    }
    encoded_credentials = base64.b64encode(
        f"{settings_llm.REMOTE_AUTH_USER}:{settings_llm.REMOTE_AUTH_PASSWORD}".encode()
    ).decode()
    headers = {'Authorization': f'Basic {encoded_credentials}'}
    X_char = 1000
    llm_class = OllamaLLM(
        model="gemma3:12b",
        temperature=0.1,
        base_url=settings_llm.REMOTE_LLM_URL,
        client_kwargs={'headers': headers},
    )
    llm_keywords = OllamaLLM(
        model="gemma3:12b",
        temperature=0.0,
        base_url=settings_llm.REMOTE_LLM_URL,
        client_kwargs={'headers': headers},
    )
    def __init__(self, documents: List[LangDocument]):
        self.documents = documents
    def remove_text_between_tags(self, text: str):
        start_tag = '</think>'
        end_tag = '</think>'
        result = []
        last_position = 0
        while True:
            start_index = text.find(start_tag, last_position)
            if start_index == -1:
                break
            result.append(text[last_position:start_index])
            end_index = text.find(end_tag, start_index + len(start_tag))
            if end_index == -1:
                break
            last_position = end_index + len(end_tag)
        result.append(text[last_position:])
        return ''.join(result)
    def clean_doc(self):
        for doc in self.documents:
            doc.page_content = doc.page_content.replace('\n', '')
    def get_X_characters(self) -> str:
        result_start = ""
        result_end = ""
        for doc in self.documents:
            if len(result_start) >= self.X_char:
                break
            result_start += doc.page_content[:self.X_char - len(result_start)]
        for doc in reversed(self.documents):
            if len(result_end) >= self.X_char:
                break
            result_end = doc.page_content[-(self.X_char - len(result_end)):] + result_end
        return result_start + result_end
    def define_document_type(self, doc_for_context: str) -> str:
        category_str = ""
        for category, promt in self.promt_category.items():
            category_str = category_str + category + ", "
        promt_category = f"""Контекст: мы проводим работы по классификации текстов, необходимо определять к какой категории относится текст
        Роль: твоя роль по части текста определять к какой из предложенных категории относится этот текст 
        Задача: Тебе предоставлен текст {doc_for_context} ты должен определить к какой  
        категорий из списка он относится, список категорий: {category_str}. 
        Критерии Качества: необходимо предоставить точно одну из предоставленных списка категорий, нельзя менять использовать другие слова, должно быть только название категории"""
        llm_response = self.llm_class.invoke(promt_category)
        return self.remove_text_between_tags(llm_response)
    def find_category(self, text: str) -> str:
        for category in self.promt_category.keys():
            if category.lower() in text.lower():
                return category
        return "Неизвестная категория"
    def return_promt_find_keywords(self, doc_type: str) -> str:
        promt_ = ""
        dictionary = self.promt_category
        if doc_type in dictionary:
            promt_ =  dictionary[doc_type]
        return promt_
    def find_keywords(self, input_dic, doc_char: str) -> str:
        promt_keyword = "Вы полезный ассистент. Вы отвечаете на вопросы о документации, используя эти данные: {self.data}. Ответь на русском языке на этот запрос: {self.prompt} "
        promt_keyword = promt_keyword.replace("{self.data}", doc_char)
        modified_parts = []  
        for key in input_dic:
            promt_llm = promt_keyword.replace("{self.prompt}", input_dic[key])
            llm_response = self.llm_keywords.invoke(promt_llm)
            modified_parts.append(key + " " + llm_response)  
        return ";".join(modified_parts)  
    def add_keywords(self, additional_info: str):
        for doc in self.documents:
            if 'page_content' not in doc.__dict__:
                doc.page_content = ""
            doc.page_content += f"\n{additional_info}"
    def get_keywords_def(self):
        self.clean_doc()
        doc_char = self.get_X_characters()
        doc_type_llm = self.define_document_type(doc_char)
        doc_type = self.find_category(doc_type_llm)
        promt_key = self.return_promt_find_keywords(doc_type)
        return self.find_keywords(promt_key, doc_char)
    def enrich_chunk_with_additional_info(self, doc, additional_text):
        enriched_content = f"{additional_text}\n{doc.page_content}"  
        return LangDocument(page_content=enriched_content, metadata=doc.metadata)
