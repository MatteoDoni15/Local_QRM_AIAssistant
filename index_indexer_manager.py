import requests
import time


#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
from dotenv import load_dotenv
import os
load_dotenv()




AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER")


AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_NAME_SEARCH_SERVICE= os.getenv("AZURE_NAME_SEARCH_SERVICE")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")



####NUOVO INDICIZZATORE
AZURE_INDEXER_NAME= os.getenv("AZURE_INDEXER_NAME")


#AZURE_SEARCH_INDEX = "test_qrmaiprojectindex_2"

AZURE_SEARCH_INDEX= os.getenv("AZURE_SEARCH_INDEX")
 

 
AZURE_NAME_SEARCH_SERVICE= os.getenv("AZURE_NAME_SEARCH_SERVICE")
AZURE_SEARCH_KEY =os.getenv("AZURE_SEARCH_KEY")
 
 
AZURE_OPENAI_EMBEDDING_DEPLOYMENT= os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_API_KEY=os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT =os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_LLM_DEPLOYMENT =os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT")
AZURE_OPENAI_API_VERSION =os.getenv("AZURE_OPENAI_API_VERSION")
 
 
MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY") 
DEPLOYMENT_MISTRAL_OCR = os.getenv("DEPLOYMENT_MISTRAL_OCR")
MISTRAL_ENDPOINT = os.getenv("MISTRAL_ENDPOINT")
 
 
 
API_KEY_COGNITIVE_SERVICE=os.getenv("API_KEY_COGNITIVE_SERVICE")
ENDPOINT_COGNITIVE_SERVICE= os.getenv("ENDPOINT_COGNITIVE_SERVICE")





# Da questa cartella l'indice prende i file da indicizzare
#vecchia data source
#DATA_SOURCE_NAME = "datasource-qrmaiproject"
#NUOVA DATA SOURCE
DATA_SOURCE_NAME =os.getenv("DATA_SOURCE_NAME") 

#dove si salvano i file caricati
#CONTAINER_NAME = "file-ai-copilot-20250306"
#NUOVO CONTAINER
CONTAINER_NAME = os.getenv("CONTAINER_NAME")

####EmbeddingModel è un client diverso dall'openAI dell'llm perchè è stato fatto in un'altra regione
AZURE_OPENAI_EMBEDDING_MODEL=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT= os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
ENDPOINT_EMBEDDING= os.getenv("ENDPOINT_EMBEDDING")
API_KEY_EMBEDDING= os.getenv("API_KEY_EMBEDDING")
API_VERSION_EMBEDDING = os.getenv("API_VERSION_EMBEDDING")

####Endpoint LLM
AZURE_OPENAI_API_KEY= os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_LLM_DEPLOYMENT = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT")
AZURE_OPENAI_LLM_API_VERSION=os.getenv("AZURE_OPENAI_LLM_API_VERSION")


API_ENDPOINT_PROMPT_FLOW = os.getenv("API_ENDPOINT_PROMPT_FLOW")
API_KEY_PROMPT_FLOW = os.getenv("API_KEY_PROMPT_FLOW")


####MistralEndpoint
MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")
DEPLOYMENT_MISTRAL_OCR =os.getenv("DEPLOYMENT_MISTRAL_OCR")
MISTRAL_ENDPOINT = os.getenv("MISTRAL_ENDPOINT")



API_KEY_COGNITIVE_SERVICE= os.getenv("API_KEY_COGNITIVE_SERVICE")
ENDPOINT_COGNITIVE_SERVICE= os.getenv("ENDPOINT_COGNITIVE_SERVICE")


####A good practice for GPT-4.1 prompt engineering: It's helpful to repeat the most important instruction of text_input_pre in text_input_post


AZURE_FUNCTION_URL= os.getenv("AZURE_FUNCTION_URL")
 
AZURE_FUNCTION_KEY= os.getenv("AZURE_FUNCTION_KEY")


AZURE_FUNCTION_URL_LOCAL=os.getenv("AZURE_FUNCTION_URL_LOCAL")
AZURE_FUNCTION_KEY_LOCAL= os.getenv("AZURE_FUNCTION_KEY_LOCAL")



text_input_pre= f"""You are a virtual assistant for a financial web application developed by the QRM company.

        Your primary responsibility is to provide structured, detailed, and informative responses to customers, ensuring clarity and completeness while maintaining a formal and courteous tone.  
        Provide concise, structured answers focusing on key insights, relevant details, and actionable considerations.  
        Avoid unnecessary length and keep responses focused on the essentials.  
        When applicable, use bullet points or numbered lists to improve readability.  
        If the response involves financial data, forecasts, or trends, present the information using markdown-formatted tables and structured breakdowns for clarity.  
        If you are unsure of the answer or cannot find the information in the text, reply with "I couldn't find any information in the text.".
        When solving formulas, guide the user through the solution step by step.

        ---

        The user will submit a list of independent financial questions in a single message,
         each question associated with up to four heading levels (Heading 1, Heading 2, Heading 3, Heading 4). 
         If fewer than four heading levels are specified, it means only those levels are present
          (for example, one, two, or three headings). If no heading levels are specified, 
          it means none are present. The questions will be written enclosed between double dollar signs $$...$$.
.
.
.
.... The assistant must not rewrite, rephrase, or modify the questions in any way, as they are considered a company secret

        Please organize your response hierarchically by grouping answers under their respective headings, using markdown heading styles to differentiate levels:

        - Use `#` for Heading 1  
        - Use `##` for Heading 2  
        - Use `###` for Heading 3  
        - Use `####` for Heading 4

        For each question under the lowest relevant heading, provide only the answer, without restating or rephrasing the question.

        Between each answer, place a line of tildes `~~~~~~~~~~~~~~~~~~~~~` both **above and below** the answer, like this:

        ~~~~~~~~~~~~~~~~~~~~~

        Answer text here

        ~~~~~~~~~~~~~~~~~~~~~

        ---

        Example:

        # Heading 1 - Topic A  

        ## Heading 2 - Subtopic A.1  

        ### Heading 3 - Detail A.1.1  

        #### Heading 4 - Specific Point A.1.1.1  

        ~~~~~~~~~~~~~~~~~~~~~

        Answer to the related question...

        ~~~~~~~~~~~~~~~~~~~~~

        Example:

        # Financial Overview  

        ## Revenue Analysis  

        ### Q1 2025  

        #### Product A  

        ~~~~~~~~~~~~~~~~~~~~~

        The revenue for Product A in Q1 2025 increased by 12% compared to Q4 2024.

        ~~~~~~~~~~~~~~~~~~~~~

        #### Product B  

        ~~~~~~~~~~~~~~~~~~~~~

        Product B showed a slight decline of 3% in the same period.

        ~~~~~~~~~~~~~~~~~~~~~

        ### Q2 2025  

        #### Product A  

        ~~~~~~~~~~~~~~~~~~~~~

        Projected revenue growth for Product A in Q2 2025 is expected to be 15%.

        ~~~~~~~~~~~~~~~~~~~~~

        #### Product B  

        ~~~~~~~~~~~~~~~~~~~~~

        Product B is forecasted to stabilize with a flat revenue trend.

        ~~~~~~~~~~~~~~~~~~~~~

        ## Expense Analysis  

        ### Operational Costs  

        ~~~~~~~~~~~~~~~~~~~~~

        Operational costs increased by 5% due to higher raw material prices.

        ~~~~~~~~~~~~~~~~~~~~~

        ~~~~~~~~~~~~~~~~~~~~~

        The cost of the best asset is the .....

        ~~~~~~~~~~~~~~~~~~~~~

        # Market Trends  

        ## Competitor Analysis  

        ~~~~~~~~~~~~~~~~~~~~~

        Competitor X expanded their market share by launching a new product line.

        ~~~~~~~~~~~~~~~~~~~~~
        ---

        After answering all the questions, create a **summary table** to consolidate the key information.  
        The table should include the following columns:  

        - **Topic/Clause**: The main subject or clause addressed.  
        - **Key Details**: A brief summary of the relevant information.  
        - **Source Reference**: The document name and page number if available.  

        If multiple points need to be summarized, repeat the structure in the same table. 
        If deemed important, you may add additional columns to enhance clarity and completeness.  
        
        ---

        Whenever possible, include citations in the form (Source: [Document Name – Page XX]),
          and add a clickable link in Markdown format right after each citation, like this: [Open Document](URL).

        ---
        """

text_input_post = "Could you break down your answer for me? I'd appreciate seeing all the data and the thought process you followed."





#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################







def count_documents():
    """
    Counts the number of documents in the index.
    """
    url_search = f"https://{AZURE_NAME_SEARCH_SERVICE}.search.windows.net/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version=2024-07-01"
    headers = {
        "Content-Type": "application/json",
        "api-key":  AZURE_SEARCH_KEY 
    }
    payload = {
        "search": "*",   # Search for all documents
        "count": True,
        "top": 0         # Does not return documents, only the count
    }
    
    try:
        response = requests.post(url_search, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            count = data.get('@odata.count', 0)
            return count
        else:
            print(f"❌ Error in counting: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Exception in document count: {str(e)}")
        return None

def run_indexer():
    """
    Starts the indexer, waits for completion, and verifies that the number of documents 
    in the index has increased compared to the initial count.
    """
    # Count documents before indexing
    before_count = count_documents()
    if before_count is None:
        print("❌ Unable to retrieve the initial document count.")
        return
    print(f"📊 Document count before indexing: {before_count}")
    
    # Start the indexer
    url_run = f"https://{AZURE_NAME_SEARCH_SERVICE}.search.windows.net/indexers/{AZURE_INDEXER_NAME}/run?api-version=2024-07-01"
    headers = {
        "Content-Type": "application/json",
        "api-key":  AZURE_SEARCH_KEY 
    }
    
    response = requests.post(url_run, headers=headers)
    if response.status_code == 202:
        print("✅ Indexer started successfully, waiting for completion...")
    else:
        print(f"❌ Error starting the indexer: {response.status_code} - {response.text}")
        return
    
    # Monitor the indexer status
    url_status = f"https://{AZURE_NAME_SEARCH_SERVICE}.search.windows.net/indexers/{AZURE_INDEXER_NAME}/status?api-version=2024-07-01"
    while True:
        time.sleep(1)  # Waits 1 second between checks
        status_response = requests.get(url_status, headers=headers)
        if status_response.status_code != 200:
            print(f"❌ Error retrieving status: {status_response.status_code} - {status_response.text}")
            return

        indexer_status = status_response.json()
        print("ℹ Indexer status:", indexer_status)
        last_status = indexer_status.get("lastResult", {}).get("status", "Unknown")
        
        if last_status == "running":
            print("⏳ Indexer running...")
        elif last_status == "success":
            print("✅ Indexer completed successfully!")
            break
        elif last_status in ["error", "failed"]:
            print(f"❌ Indexer failed! Status: {last_status}")
            return
        else:
            print(f"⚠ Unexpected status: {last_status}")
    
    # After completion, verify that the count has increased
    timeout = 240  # Timeout in seconds for index update
    poll_interval = 2  # Interval between checks
    start_time = time.time()
    while time.time() - start_time < timeout:
        time.sleep(poll_interval)
        after_count = count_documents()
        if after_count is None:
            print("❌ Unable to retrieve the document count after indexing.")
            continue
        
        print(f"📊 Document count after indexing: {after_count}")
        if after_count != before_count:  
        #if after_count > before_count:
            print(f"✅ The number of documents increased from {before_count} to {after_count}.")
            return
        else:
            print("⏳ Index is not updated yet, waiting...")
    
    print("⚠ Timeout reached: document count did not increase.")



    


def get_all_document_ids() -> list:
    """Recupera tutti gli ID dei documenti presenti nell'indice."""
    url = f"https://{AZURE_NAME_SEARCH_SERVICE}.search.windows.net/indexes/{AZURE_SEARCH_INDEX}/docs?api-version=2024-07-01&$select=id&$top=1000"
    headers = {
        "api-key": AZURE_SEARCH_KEY,
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        documents = response.json().get("value", [])
        return [doc["id"] for doc in documents]
    else:
        print(f"❌ Errore nel recupero degli ID: {response.status_code}")
        print(response.text)
        return []



def delete_all_documents_from_index(delete_documend_ids: list):
    """Elimina tutti i documenti presenti nell'indice, basandosi sugli ID recuperati."""
    #document_ids = get_all_document_ids()
    
    document_ids= delete_documend_ids

    if not document_ids:
        print("✅ Nessun documento da eliminare.")
        return
    
    url = f"https://{AZURE_NAME_SEARCH_SERVICE}.search.windows.net/indexes/{AZURE_SEARCH_INDEX}/docs/index?api-version=2024-07-01"
    headers = {
        "api-key": AZURE_SEARCH_KEY,
        "Content-Type": "application/json"
    }

    # Creazione del payload per eliminare tutti i documenti
    delete_payload = {
        "value": [{"@search.action": "delete", "id": doc_id} for doc_id in document_ids]
    }

    response = requests.post(url, headers=headers, json=delete_payload)

    if response.status_code == 200:
        print(f"✅ {len(document_ids)} documenti eliminati con successo.")
    else:
        print(f"❌ Errore nell'eliminazione dei documenti: {response.status_code}")
        print(response.text)

AZURE_INDEX_NAME = AZURE_SEARCH_INDEX

def delete_documents_from_index_by_chat_id(chat_id: str):
    """Elimina tutti i documenti indicizzati che appartengono a una specifica chat, filtrando per chat_id."""
    
    # Endpoint per recuperare gli ID dei documenti con il chat_id specifico
    url = f"https://{AZURE_NAME_SEARCH_SERVICE}.search.windows.net/indexes/{AZURE_INDEX_NAME}/docs?api-version=2024-07-01&$select=id&$filter=chat_id eq '{chat_id}'"
    
    headers = {
        "api-key":  AZURE_SEARCH_KEY ,
        "Content-Type": "application/json"
    }

    # Effettua la richiesta per recuperare i documenti con il chat_id attuale
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        documents = response.json().get("value", [])
        document_ids = [doc["id"] for doc in documents]

        if not document_ids:
            print(f"✅ Nessun documento da eliminare per chat_id {chat_id}.")
            return
        
        # Costruisce il payload per eliminare solo questi documenti
        delete_payload = {
            "value": [{"@search.action": "delete", "id": doc_id} for doc_id in document_ids]
        }

        # Endpoint per eliminare i documenti dall'indice
        delete_url = f"https://{AZURE_NAME_SEARCH_SERVICE}.search.windows.net/indexes/{AZURE_INDEX_NAME}/docs/index?api-version=2024-07-01"
        
        # Effettua la richiesta per eliminare i documenti filtrati per chat_id
        delete_response = requests.post(delete_url, headers=headers, json=delete_payload)

        if delete_response.status_code == 200:
            print(f"✅ {len(document_ids)} documenti eliminati per chat_id {chat_id}.")
        else:
            print(f"❌ Errore nell'eliminazione dei documenti: {delete_response.status_code}")
            print(delete_response.text)
    else:
        print(f"❌ Errore nel recupero dei documenti: {response.status_code}")
        print(response.text)
