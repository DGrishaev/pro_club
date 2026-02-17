# имена классов должны начинаться с s_
# и отображать основные характеристики (по усмотрению разработчика)

import os


def _k_from_env(default: int) -> int:
    try:
        return int(os.getenv("RAG_RETRIEVAL_K", str(default)))
    except ValueError:
        return default


class s_default:
    def __init__(self, prompt, user_name, vectordb):
        self.prompt = prompt
        self.user_name = user_name
        self.vectordb = vectordb

    def seach_from_db(self):
        k = _k_from_env(4)
        return self.vectordb.similarity_search(self.prompt, k=k)


class s_k_five:
    def __init__(self, prompt, user_name, vectordb):
        self.prompt = prompt
        self.user_name = user_name
        self.vectordb = vectordb

    def seach_from_db(self):
        k = _k_from_env(5)
        return self.vectordb.similarity_search(self.prompt, k=k)


class s_k_2:
    def __init__(self, prompt, user_name, vectordb):
        self.prompt = prompt
        self.user_name = user_name
        self.vectordb = vectordb

    def seach_from_db(self):
        k = _k_from_env(2)
        return self.vectordb.similarity_search(self.prompt, k=k)


class s_k_150:
    def __init__(self, prompt, user_name, vectordb):
        self.prompt = prompt
        self.user_name = user_name
        self.vectordb = vectordb

    def seach_from_db(self):
        threshold = float(os.getenv("RAG_SCORE_THRESHOLD", "0.4"))
        k = _k_from_env(150)
        results = self.vectordb.similarity_search_with_score(self.prompt, k=k)
        filtered_docs = []

        for doc, score in results:
            if score <= threshold:  # score = расстояние
                filtered_docs.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score,
                    }
                )
        return filtered_docs
