# имена классов должны начинаться с pvid_
# и отображать основные характеристики (по усмотрению разработчика)

import os

from langchain_chroma import Chroma

from app.config import settings


def _resolve_vectorstore_dir(user_name) -> str:
    user_folder_name = user_name if isinstance(user_name, str) else user_name.name
    safe_name = user_folder_name.replace("/", "_").replace("\\", "_")
    root = settings.MAIN_FOLDER_PATH
    return os.path.join(root, safe_name, "db")


class pvid_default:
    def __init__(self, separate_text, embedding, user_name):
        self.separate_text = separate_text
        self.embedding = embedding
        self.user_name = user_name

    def put_vector_in_db(self):
        if not self.separate_text:
            return False

        db_folder = _resolve_vectorstore_dir(self.user_name)
        os.makedirs(db_folder, exist_ok=True)

        Chroma.from_documents(
            collection_name="main",
            documents=self.separate_text,
            embedding=self.embedding,
            persist_directory=db_folder,
        )

        return True
