import os
import base64
import logging
from typing import Optional, Union, Type, List
from enum import Enum
import json
from dataclasses import dataclass, field
#import io_file_operation
#import io_json
from app.utils.llm_implementation import io_separate_file
import app.utils.llm_implementation.io_embeddings as io_embeddings
import app.utils.llm_implementation.io_put_vector_in_db as io_put_vector_in_db
import app.utils.llm_implementation.io_get_vectror_db as io_get_vectror_db
import app.utils.llm_implementation.io_search_from_db as io_search_from_db
import app.utils.llm_implementation.io_promt as io_promt
import app.utils.io_file_operation as io_file_operation
from app.utils.rag_debug import RagDebugger

from collections import defaultdict
from app.config import settings_llm, settings

from langchain_ollama import OllamaLLM
from langchain.schema import HumanMessage
from langchain_gigachat import GigaChat

MAX_TEXT_LENGTH = 5000  # Максимальная длина текста для резюме

class LLM_Models(Enum):
    Olama3 = settings_llm.MODEL

@dataclass
class Processed_Files:
    name: str

@dataclass
class Prompts:
    text: str

@dataclass
class ConfigLLM:
    processed_files: List[Processed_Files] = field(default_factory=list) 
    prompts: List[Prompts] = field(default_factory=list) 

    def to_dict(self):
        return {
           "processed_files": [file.__dict__ for file in self.processed_files],
           "prompts": [texts.__dict__ for texts in self.prompts]}
    
    @classmethod
    def from_dict(cls, data: dict):
        processed_files = [Processed_Files(**files) for files in data["processed_files"]]
        prompts = [Prompts(**texts) for texts in data["prompts"]]
        return cls(processed_files=processed_files, prompts=prompts)


#---------------------------------------------------
class DbHelper:

    def __init__(self, user_name, default_model: LLM_Models = LLM_Models.Olama3):
        #self.chat_id = chat_id
        self.user_name = user_name
        self.default_model= default_model
        self.logger = logging.getLogger("rag.db_helper")
        self.debugger = RagDebugger(self.logger)
        
    """def process_files(self, user_name):
        copy_user_files_from_input(user_name)
        db_helper = io_db.DbHelper(user_name)
        self.processing_user_files() """

    def processing_user_files(self, processing_all_files: Optional[bool]=False, saved_files = []):
        self.logger.info("Start processing_user_files")
        self.logger.info("files to process from save stage: %s", len(saved_files))
        configLLM_object = self.get_configLLM_file()

        all_pdf_file_list = self.get_all_user_files()
        if processing_all_files:
            pdf_files_list = all_pdf_file_list
            files_to_update = []
            configLLM_object.processed_files.clear
        else:
            # get differents between all user files and processed files
            set_difference = set(all_pdf_file_list) - set([file.name for file in configLLM_object.processed_files])
            pdf_files_list = list(set_difference)

            # Если файлы были добавлены ранее, то добавим в словарь повторной загрузки
            user_folder_path = io_file_operation.return_user_folder(self.user_name)
            user_folder_path = os.path.join(user_folder_path, 'pdf') 
            files_to_update = list(map(lambda file: os.path.join(user_folder_path, file), saved_files))
            self.logger.info("files to reindex: %s", len(files_to_update))

        if len(pdf_files_list) == 0 and len(files_to_update) == 0:
            #todo show user error
            return
        
        self.clear_files_in_db(files_to_update)
        pdf_files_list = list(set(pdf_files_list + files_to_update))
        
        for file_item in pdf_files_list:
            separate_text = self.separate_file(file_item)
            embedding = self.get_embeddings()
            self.put_vector_in_db(separate_text, embedding)
            vectordb = self.get_vectror_db(embedding)
            records_count = len(vectordb.get().get("ids", [])) if vectordb else None
            self.debugger.log_indexing(file_item, len(separate_text), records_count)
            #check file as processed
            configLLM_object.processed_files.append(Processed_Files(name=file_item))
    
        #save configLLM
        self.save_to_configLLM_file(configLLM_object)
        self.logger.info("finish processing_user_files")

    def get_configLLM_file(self):
        user_folder_path = io_file_operation.return_user_folder(self.user_name)
        configLLM_full_path = os.path.join(user_folder_path, 'configLLM.json')
        configLLM_object = ConfigLLM()
        if not os.path.exists(configLLM_full_path):
            # create confilLLM file
           with open(configLLM_full_path, 'w') as file:
                json.dump(configLLM_object.to_dict(), file, indent=4)
         
        with open(configLLM_full_path, 'r') as file:
            data = json.load(file)
            configLLM_object = configLLM_object.from_dict(data)

        return configLLM_object
        
    def save_to_configLLM_file(self, configLLM_object: ConfigLLM):
        user_folder_path = io_file_operation.return_user_folder(self.user_name)
        configLLM_full_path = os.path.join(user_folder_path, 'configLLM.json')
        if not os.path.exists(configLLM_full_path):
            return
        
        with open(configLLM_full_path, 'w') as file:
            json.dump(configLLM_object.to_dict(), file, indent=4)

    def delete_all_user_db(self):
        #db_user_files = io_file_operation.return_user_folder_db(self.user_name)
        #io_file_operation.delete_all_files_in_folder(self.chat_id, db_user_files, True)
        vectordb = self.get_vectror_db()
        vectordb._client.delete_collection("main")
        self.delete_proocessed_files_in_config()

    def delete_proocessed_files_in_config(self):
        user_folder_path = io_file_operation.return_user_folder(self.user_name)
        configLLM_full_path = os.path.join(user_folder_path, 'configLLM.json')
        configLLM_object = ConfigLLM()
        if os.path.exists(configLLM_full_path):
            # create confilLLM file
           with open(configLLM_full_path, 'w') as file:
                json.dump(configLLM_object.to_dict(), file, indent=4)


    def get_all_user_files(self):

        user_folder_path = io_file_operation.return_user_folder(self.user_name)
        if not os.path.exists(user_folder_path):
            return []
        
        pdf_folder = os.path.join(user_folder_path, 'pdf')

        if not os.path.exists(pdf_folder):
            return []
        
        pdf_files_all = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder)]

        return pdf_files_all

    def separate_file(self, file_path):

        class_name_separate_file = settings_llm.CLASS_NAME_SEPARATE_FILE
        separate_class = getattr(io_separate_file, class_name_separate_file)
        separate_object = separate_class(file_path)
        chunks = separate_object.separate_file()
        self.debugger.log_chunks(chunks, file_path)
        return chunks

    def get_embeddings(self):

        class_name_embedings = settings_llm.CLASS_NAME_EMBEDDINGS
        embedings_class = getattr(io_embeddings, class_name_embedings)
        embedings_object = embedings_class()
        return embedings_object.get_embeddings()

    def put_vector_in_db(self, separate_text, embedding):
        class_name_pvidb = settings_llm.CLASS_NAME_PUT_VECTOR_IN_DB
        pvidb_class = getattr(io_put_vector_in_db, class_name_pvidb)
        pvid_object = pvidb_class(separate_text, embedding, self.user_name)
        return pvid_object.put_vector_in_db()
    
    def clear_files_in_db(self, files_to_update):
        for files in files_to_update:
            print("Обновляю файл")
            print(files)
            vector_db = self.get_vectror_db()
            print("ищу данные в базе")
            docs_to_delete = vector_db.get(where={"source": files})
            for doc_id, metadata in zip(docs_to_delete["ids"], docs_to_delete["metadatas"]):
                print(f"ID: {doc_id}, Metadata: {metadata}")
                vector_db.delete(ids=doc_id)
            print("Документы успешно удалены!")

    def get_vectror_db(self, embedding=None):

        class_name_gvidb = settings_llm.CLASS_NAME_GET_VECTOR_DB
        gvidb_class = getattr(io_get_vectror_db, class_name_gvidb)
        gvid_object = gvidb_class(self.user_name, embedding)
        return gvid_object.get_vectror_db() 

    def get_answer(self, prompt, llm_model: LLM_Models = None):

        self.logger.info("get_answer")

        embedding = self.get_embeddings()
        vectordb = self.get_vectror_db(embedding)

        llm = self.get_llm(llm_model)

        self.logger.info("get_answer + llm")

        class_name_search = settings_llm.CLASS_NAME_SEARCH
        search_class = getattr(io_search_from_db, class_name_search)
        search_object = search_class(prompt, self.user_name, vectordb)
        data =  search_object.seach_from_db()
        if isinstance(data, list):
            normalized = []
            for item in data:
                if isinstance(item, dict):
                    normalized.append(item)
                else:
                    normalized.append({
                        "content": getattr(item, "page_content", ""),
                        "metadata": getattr(item, "metadata", {}),
                        "score": None,
                    })
            self.debugger.log_retrieval(
                prompt=prompt,
                k=int(os.getenv("RAG_RETRIEVAL_K", "5")),
                filters=None,
                results=normalized,
            )

        self.logger.info("get_answer + io_search_from_db")

        class_name_promt = settings_llm.CLASS_NAME_PROMT
        promt_class = getattr(io_promt, class_name_promt)
        promt_object = promt_class(data, prompt)
        question  =  promt_object.get_promt()

        self.logger.info("get_answer + get_promt")

        text = llm.invoke(question)

        if LLM_Models.Olama3.value == "gigachat":
            text = text.content

        return text
    
    def get_search_answer(self, prompt, llm_model: LLM_Models = None):

        self.logger.info("get_search_answer")

        embedding = self.get_embeddings()
        vectordb = self.get_vectror_db(embedding)

        llm = self.get_llm(llm_model)

        self.logger.info("get_search_answer + llm")

        class_name_search = settings_llm.CLASS_NAME_SEARCH_SEARCH
        search_class = getattr(io_search_from_db, class_name_search)
        search_object = search_class(prompt, self.user_name, vectordb)
        data =  search_object.seach_from_db()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            self.debugger.log_retrieval(
                prompt=prompt,
                k=int(os.getenv("RAG_RETRIEVAL_K", "150")),
                filters=None,
                results=data,
            )

        # ✅ Правильная проверка: если data — пустой список или None
        if not data or isinstance(data, dict):
            # Если data — словарь или пустой/ошибочный результат
            return "По вашему запросу не найдено ни одного релевантного документа."

        # Группируем
        grouped = self.group_chunks_by_document(data, "source")

        self.logger.info("get_search_answer + io_search_from_db")

        class_name_promt = settings_llm.CLASS_NAME_PROMT
        promt_class = getattr(io_promt, class_name_promt)
        promt_object = promt_class(data, prompt)
        question  =  promt_object.get_promt()

        self.logger.info("get_search_answer + get_promt")

        #text = llm.invoke(question)
        #text = f"filtered_docs: {len(grouped)}"
        text = self.generate_search_response(grouped, llm)

        if LLM_Models.Olama3.value == "gigachat":
            text = text.content

        return text

    def group_chunks_by_document(self, filtered_docs, metadata_key="source"):
        """
        Группирует чанки по имени файла (source) или другому полю в metadata.
        
        :param filtered_docs: список словарей вида
            {
                "content": "текст чанка",
                "metadata": {"source": "...", "chunk": 1, "type": "paragraph"},
                "score": 0.45
            }
        :param metadata_key: ключ в metadata, по которому группируем (например, "source")
        :return: список документов с объединёнными чанками
        """
        if not filtered_docs:
            return []

        groups = defaultdict(list)

        # Группируем чанки по значению metadata[metadata_key]
        for item in filtered_docs:
            key = item["metadata"].get(metadata_key)
            if not key:
                key = "unknown_source"
            groups[key].append({
                "content": item["content"],
                "score": item["score"],
                "chunk_metadata": {k: v for k, v in item["metadata"].items() if k != "source"}  # остальные метаданные
            })

        # Формируем результат
        result = []
        for source, chunks in groups.items():
            # Берём метаданные из первого чанка (source там уже есть)
            sample_item = filtered_docs[next(i for i, item in enumerate(filtered_docs)
                                            if item["metadata"].get(metadata_key) == source)]

            result.append({
                "source": source,  # имя файла
                "metadata": sample_item["metadata"],  # можно взять целиком
                "chunks": chunks,  # список всех подходящих чанков
                "content_snippets": [chunk["content"] for chunk in chunks],
                "full_text": "\n\n---\n\n".join([chunk["content"] for chunk in chunks]),
                "chunk_count": len(chunks),
                "min_score": min(chunk["score"] for chunk in chunks),  # лучшее совпадение
                "max_score": max(chunk["score"] for chunk in chunks),  # худшее
            })

        # Сортируем по релевантности: сначала документы с наилучшим (минимальным) score
        result.sort(key=lambda x: x["min_score"])

        return result
    
    def generate_summary_for_document(self, full_text: str, min_score: float, llm) -> str:
        """
        Отправляет укороченный текст в LLM и возвращает краткое резюме.
        Добавляет оценку релевантности (min_score) в промт (опционально).
        """
        # Обрезаем текст до MAX_TEXT_LENGTH символов
        truncated_text = full_text[:MAX_TEXT_LENGTH]

        prompt = f"""
        Проанализируй следующие фрагменты текста из одного документа и составь краткое резюме (3–5 предложений), о чём идёт речь в этом документе. Укажи основные темы, понятия и контекст. Не включай упоминания о структуре (например, "в параграфе 3 сказано..."), а передай суть содержания.

        Текст:
        {truncated_text}

        Ответ должен быть на русском языке, кратким и информативным.
        """

        try:
            response = llm.invoke(prompt)
            summary = response.strip()
        except Exception as e:
            summary = "Не удалось сгенерировать резюме (ошибка при обращении к LLM)."

        return summary

    def generate_search_response(self, grouped, llm, max_text_length=5000):
        if not grouped:
            return "По вашему запросу не найдено ни одного релевантного документа."

        # Устанавливаем лимит длины текста
        MAX_TEXT_LENGTH = max_text_length

        # Формируем заголовок
        count = len(grouped)
        if count == 1:
            header = f"**Найден 1 документ по вашему запросу.**"
        elif 2 <= count <= 4:
            header = f"**Найдено {count} документа по вашему запросу.**"
        else:
            header = f"**Найдено {count} документов по вашему запросу.**"

        result_lines = [
            header,
            ""  # пустая строка после заголовка
        ]

        for i, item in enumerate(grouped, 1):
            source = item["source"]
            full_text = item["full_text"]
            min_score = item["min_score"]
            chunk_count = item["chunk_count"]

            # Оценка релевантности
            if min_score < 0.3:
                relevance = "высокая"
            elif min_score < 0.5:
                relevance = "средняя"
            elif min_score < 0.7:
                relevance = "низкая"
            else:
                relevance = "очень низкая"

            # Генерируем краткое резюме
            summary = self.generate_summary_for_document(full_text, min_score, llm)

            # Добавляем информацию о документе
            result_lines.append(f"**{i}. {source}**")
            result_lines.append(f"Релевантность: {relevance} (score: {min_score:.2f})")
            result_lines.append(f"Найдено фрагментов: {chunk_count}")
            result_lines.append(f"Краткое содержание: {summary}")
            result_lines.append("")  # разделитель между документами

        return "\n".join(result_lines).strip()
    
    def get_free_answer (self, prompt, llm_model: LLM_Models = None):
        
        llm = self.get_llm(llm_model)     
        question = f"Ответь на русском языке: {prompt}  "
        text = llm.invoke([HumanMessage(content=question)])

        if LLM_Models.Olama3.value == "gigachat":
            text = text.content
        
        return text
    
    def get_llm(self, llm_model: LLM_Models = None):

        selected_llm_model = llm_model.value if isinstance(llm_model, LLM_Models) else llm_model
        if selected_llm_model is None:
            selected_llm_model = self.default_model.value if isinstance(self.default_model, LLM_Models) else self.default_model

        encoded_credentials = base64.b64encode(f"{settings_llm.USER_LLM}:{settings_llm.PASSWORD_LLM}".encode()).decode()
        headers = {'Authorization': f'Basic {encoded_credentials}'}
        
        if LLM_Models.Olama3.value == "gigachat":
            # Указываем ключ API
            gigachat_key = settings.GIGACHAT_TOKEN
            llm = GigaChat(credentials=gigachat_key, verify_ssl_certs=False,)
        else:
            llm = OllamaLLM( 
                model=selected_llm_model, temperature = 0.1, base_url=settings_llm.URL_LLM, client_kwargs={'headers': headers})
        
        return llm

        
        
