import pandas, os
import datetime
import base64

from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.schema import HumanMessage

# llm = OllamaLLM(
#     model="llama3", temperature = "0.1"
# )

dotenv_path = os.path.join(os.path.dirname(__file__), '..', 'webAPI', 'app', '.env.llm')
dotenv_path = os.path.abspath(dotenv_path)
load_dotenv(dotenv_path)

user_llm = os.getenv('REMOTE_AUTH_USER')
password = os.getenv('REMOTE_AUTH_PASSWORD')
model_llm = os.getenv('REMOTE_LLM_MODEL')
url_llm = os.getenv('REMOTE_LLM_URL')

encoded_credentials = base64.b64encode(f"{user_llm}:{password}".encode()).decode()
headers = {'Authorization': f'Basic {encoded_credentials}'}

llm = OllamaLLM( 
                model=model_llm, temperature = 0.1, base_url=url_llm, client_kwargs={'headers': headers})

class rag_metrick:
    def __init__(self, logs_folder_path, logs_path_file_name, prime_path_file_name):
        self.logs_folder_path = logs_folder_path
        self.logs_path_file_name = logs_path_file_name
        self.prime_path_file_name = prime_path_file_name
    def gmetrics(self):
        logs_folder_path = self.logs_folder_path 
        logs_file_name = self.logs_path_file_name 
        prime_file_name = self.prime_path_file_name
        current_time = datetime.datetime.now()
        m_file_name = f'metrics_{current_time.strftime("%Y-%m-%d_%H-%M-%S")}.csv'
        metrics_file_name = os.path.join(logs_folder_path, m_file_name)

        metrics_file = pandas.DataFrame(columns=['date', 'request_text', 'used_files', 'response_text_log', 'response_text_prime', 'metrics' ])
        metrics_file.to_csv(metrics_file_name, index=False, encoding='utf-8')

        logs_file = pandas.read_csv(logs_file_name, encoding='utf-8')
        prime_file = pandas.read_excel(prime_file_name, header=0)

        for index, row in logs_file.iterrows():
            request_text = row["request_text"]
            response_text = row["response_text"]
            used_files = row["used_files"]
            # Удаляем символы новой строки и другие пробельные символы
            safe_request_text = request_text.replace('\n', '').replace('\r', '')
            filtered_rows = prime_file.query(f"request_text == '{safe_request_text}'")
            if len(filtered_rows) == 0:
                print ("По запросу " + request_text + " не найдено данных")
            else:
                a1 = response_text
                a2 = filtered_rows['response_text'].values

                question = f"Ты учитель и проверяшь информацию от ученика. Тебе предоставлен ответ ученика {a1} и ты должен сравнить его с правильным ответом {a2}. При проверке учитывай точность информации. Если ответы идентинчны, то напиши (Ответ 1). Если ответы различаются, то напиши (Ответ 0) и укажи различия. Ответ дай на русском языке "
                metrix_data = llm.invoke([HumanMessage(content=question)])
                
                date = datetime.datetime.now()

                new_array = pandas.DataFrame([{
                    'date'  : date,
                    'request_text'  : request_text,
                    'used_files'    : used_files,
                    'response_text_log' : a1,
                    'response_text_prime'    : a2,
                    'metrics'        : metrix_data
                }])

                metrics_file = metrics_file.dropna(axis=1, how='all')
                new_array = new_array.dropna(axis=1, how='all')
                metrics_file = pandas.concat([metrics_file.dropna(axis=1, how='all'), new_array], ignore_index=True)
                metrics_file.to_csv(metrics_file_name, index=False, encoding='utf-8')
    
        return metrics_file_name
