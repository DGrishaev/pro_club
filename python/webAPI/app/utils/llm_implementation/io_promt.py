# имена классов должны начинаться с promt_
# и отображать основные характеристики (по усмотрению разработчика)
from app.utils.rag_artifacts import rag_artifacts, build_context_blob


def _save_prompt_artifacts(class_name: str, data, question: str, full_prompt: str) -> None:
    """Сохраняет итоговый промт и чистый контекст."""
    if not rag_artifacts.has_session("QU"):
        return
    rag_artifacts.write_text("QU", "prompt", "final_prompt.txt", full_prompt)
    rag_artifacts.write_text("QU", "prompt", "context_only.txt", build_context_blob(data))
    rag_artifacts.write_json(
        "QU",
        "prompt",
        "prompt_meta.json",
        {
            "prompt_class": class_name,
            "question": question,
        },
    )


def _log_prompt_error(step: str, exc: Exception, meta: dict | None = None) -> None:
    """Сохраняет стек ошибки при формировании промта."""
    if rag_artifacts.has_session("QU"):
        rag_artifacts.log_exception("QU", step, exc, meta or {})

class promt_default:
    def __init__(self, data, prompt):
        self.data = data
        self.prompt = prompt
    def get_promt(self):
        try:
            result = (
                f"Вы полезный ассистент. Вы отвечаете на вопросы о документации, используя эти данные: {self.data}. "
                f"Ответь на русском языке на этот запрос: {self.prompt} и укажи source "
            )
            # сохраняем финальный промт для диагностики
            _save_prompt_artifacts(self.__class__.__name__, self.data, self.prompt, result)
            return result
        except Exception as exc:
            _log_prompt_error("io_promt.promt_default", exc, {"question": self.prompt})
            raise

class promt_instr:
    def __init__(self, data, prompt):
        self.data = data
        self.prompt = prompt
    def get_promt(self):
        try:
            result = f"""
Контекст (DOCUMENT):
{self.data}

Вопрос (QUESTION):
{self.prompt}

Инструкция:
Ответь на вопрос, используя исключительно информацию из документа выше.
Не додумывай и не делай предположений.
Если в документе нет достаточной информации для точного ответа, верни: НЕТ ОТВЕТА.

Обязательно укажи источник информации, откуда ты взял ответ. 
Источник содержится в метаданных в поле source. 

Формат ответа:
1. ответ на русском языке.
2. Источник(и): перечисли значение поля "source" из документа.

Пример:
Ответ: ...  
Источник: source_1.pdf
"""
            # фиксируем текст промта в артефактах
            _save_prompt_artifacts(self.__class__.__name__, self.data, self.prompt, result)
            return result
        except Exception as exc:
            _log_prompt_error("io_promt.promt_instr", exc, {"question": self.prompt})
            raise

class promt_test:
    def __init__(self, data, prompt):
        self.data = data
        self.prompt = prompt
    def get_promt(self):
        try:
            result = (
                f"DOCUMENT: {self.data} QUESTION: {self.prompt} INSTRUCTIONS: Answer the users QUESTION using the DOCUMENT text above. "
                "Keep your answer ground in the facts of the DOCUMENT. If the DOCUMENT doesnt contain the facts to answer the QUESTION "
                "return НЕТОТВЕТА. Ответь на русском языке "
            )
            _save_prompt_artifacts(self.__class__.__name__, self.data, self.prompt, result)
            return result
        except Exception as exc:
            _log_prompt_error("io_promt.promt_test", exc, {"question": self.prompt})
            raise
    
class promt_test_update:
    def __init__(self, data, prompt):
        self.data = data
        self.prompt = prompt

    def get_promt(self):
        try:
            result = f"""
<system prompt>  
ВЫ — ЭКСПЕРТНЫЙ АССИСТЕНТ ПО АНАЛИЗУ ДОКУМЕНТАЦИИ. ВАША ГЛАВНАЯ ЗАДАЧА — ОТВЕЧАТЬ НА ВОПРОСЫ ИСКЛЮЧИТЕЛЬНО НА ОСНОВЕ ПРЕДОСТАВЛЕННЫХ ДОКУМЕНТОВ.  

<instructions>  
- ВСЕ ОТВЕТЫ ДОЛЖНЫ ОСНОВЫВАТЬСЯ ТОЛЬКО НА ЗАГРУЖЕННЫХ ДОКУМЕНТАХ ({self.data}).  
- ЕСЛИ ИНФОРМАЦИЯ ОТСУТСТВУЕТ В ДОКУМЕНТАХ, ЯВНО УКАЖИТЕ, ЧТО ДАННЫЕ НЕ НАЙДЕНЫ.  
- ВЫ ОБЯЗАНЫ ССЫЛАТЬСЯ НА ИСТОЧНИКИ (source) В КАЖДОМ ОТВЕТЕ.  
- ФОРМАТ ОТВЕТА: СНАЧАЛА КРАТКИЙ ВЫВОД, ЗАТЕМ ДОСЛОВНАЯ ЦИТАТА ИЗ ДОКУМЕНТА С УКАЗАНИЕМ ИСТОЧНИКА.  
- ОТВЕЧАЙТЕ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.  

<what not to do>  
- НИКОГДА НЕ ВЫДУМЫВАЙТЕ ОТВЕТЫ И НЕ ДОБАВЛЯЙТЕ ИНФОРМАЦИЮ, ОТСУТСТВУЮЩУЮ В ДОКУМЕНТАХ.  
- НЕ ОТВЕЧАЙТЕ НА ВОПРОСЫ, ЕСЛИ ОНИ НЕ МОГУТ БЫТЬ ПОДТВЕРЖДЕНЫ ДОКУМЕНТАМИ.  
- НЕ ГЕНЕРИРУЙТЕ ОБЩИЕ ОТВЕТЫ БЕЗ ССЫЛКИ НА ИСТОЧНИК. 
- не указывай id документов 
</what not to do>  

<example>  
<USER MESSAGE>  
Какой регламент по обработке персональных данных?  
</USER MESSAGE>  

<ASSISTANT RESPONSE>  
Согласно загруженной документации, регламент обработки персональных данных следующий:  
**"Обработка персональных данных осуществляется в соответствии с Законом № XYZ от 01.01.2023, раздел 3.2."** (source: Документ_1.pdf, стр. 12)  
</ASSISTANT RESPONSE>  
</example>  

ВОПРОС: {self.prompt}  
</system prompt>
"""        
            _save_prompt_artifacts(self.__class__.__name__, self.data, self.prompt, result)
            return result
        except Exception as exc:
            _log_prompt_error("io_promt.promt_test_update", exc, {"question": self.prompt})
            raise
