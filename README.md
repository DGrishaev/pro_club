# pro_club MVP

Команды бота:
- `/upload` — берёт все поддерживаемые файлы (`.pdf`, `.docx`, `.pptx`) из `LOCAL_RAG_UPLOAD_DIR` и индексирует их в RAG;
- `/status` — показывает список уже загруженных документов пользователя в Chroma;
- `/clear` — полностью очищает RAG-базу текущего пользователя.

Опциональные параметры ingestion через `.env`:
- `RAG_CHUNK_OVERLAP_CHARS` (по умолчанию `300`) — overlap для текстовых чанков;
- `RAG_TARGET_CHUNK_CHARS` (по умолчанию `1000`) — целевой размер агрегированного текстового чанка;
- `RAG_MAX_CHUNK_CHARS` (по умолчанию `1800`) — верхняя граница агрегированного чанка;
- `RAG_MIN_CHUNK_CHARS` (по умолчанию `300`) — нижняя желаемая граница, короткие абзацы склеиваются;
- `RAG_SKIP_TOC` (по умолчанию `true`) — не индексировать блок оглавления и строки toc;
- `RAG_SKIP_FRONT_MATTER` (по умолчанию `true`) — пропускать служебные строки титула в root;
- `RAG_EMBED_HEADINGS` (по умолчанию `false`) — индексировать заголовки как отдельные чанки;
- `RAG_SKIP_HEADINGS` (по умолчанию `true`) — пропускать заголовки как самостоятельный контент;
- `RAG_TABLE_WINDOW_THRESHOLD` (по умолчанию `50`) — порог числа строк таблицы для перехода в windowing;
- `RAG_TABLE_WINDOW_SIZE` (по умолчанию `20`) — размер окна строк в `table_rows_window`;
- `RAG_TABLE_SUMMARY_PREVIEW_ROWS` (по умолчанию `3`) — число preview-строк в `table_summary`;
- `RAG_TABLE_MAX_HEADER_ROWS` (по умолчанию `2`) — максимальная глубина шапки таблицы.

Smoke-проверка ingestion:
- `python3 -m python.webAPI.app.utils.ingest_smoke --file <path>`
