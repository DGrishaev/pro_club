.PHONY: dev api rag-debug

dev:
	@echo "Use: make api  OR  make rag-debug FILE=... QUERY='...'"

api:
	cd python/webAPI && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

rag-debug:
	python tools/rag_debug.py --file "$(FILE)" --query "$(QUERY)" --user "$(or $(USER_NAME),rag_debug_user)"
