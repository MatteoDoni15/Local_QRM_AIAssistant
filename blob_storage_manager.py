import requests
import time

#id della chat non so se nel futuro sia comodo usare anche l'id del cliente 

from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions, ContentSettings
from datetime import datetime, timedelta, timezone
from datetime import datetime, timedelta
import os
import uuid
import os
from typing import Dict


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




def generate_chat_id() -> str:
    """Genera un identificatore univoco per il chat (UUID)."""
    return str(uuid.uuid4())

from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta, timezone


#def generate_sas_token(blob_name: str) -> str:
#    """Genera un SAS Token per l'accesso temporaneo al file."""
#    expiry_time = datetime.now(timezone.utc) + timedelta(hours=3)  # Usa timezone.utc per evitare il deprecato utcnow()
#
#    # Inizializza il BlobServiceClient dalla connection string
#    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
#
#    
#
#    # Ottiene le credenziali automaticamente dal client
#    account_name = blob_service_client.account_name
#    credential = blob_service_client.credential
#
#    # Genera il SAS Token
#    sas_token = generate_blob_sas(
#        account_name=account_name,
#        container_name=CONTAINER_NAME,
#        blob_name=blob_name,
#        account_key=credential.account_key,  # Ottiene la chiave automaticamente dalla connessione
#        permission=BlobSasPermissions(read=True),  # Solo lettura
#        expiry=expiry_time
#    )
#
#    return sas_token




  
def generate_sas_token(blob_name: str) -> str:  
    """Genera un SAS Token per l'accesso temporaneo al file."""  
    expiry_time = datetime.now(timezone.utc) + timedelta(hours=3)  
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)  
    account_name = blob_service_client.account_name  
    credential = blob_service_client.credential  
    # Ottiene la chiave di account per generare il SAS  
    account_key = credential.account_key if hasattr(credential, 'account_key') else credential  
  
    sas_token = generate_blob_sas(  
        account_name=account_name,  
        container_name=CONTAINER_NAME,  
        blob_name=blob_name,  
        account_key=account_key,  
        permission=BlobSasPermissions(read=True),  
        expiry=expiry_time  
    )  
    return sas_token  


def upload_file_to_blob(file_path: str, user_id: str, chat_id: str) -> Dict:
    """
    Uploads a file to Azure Blob Storage, generates a Shared Access Signature (SAS) token,
    and updates the blob's metadata.

    The function constructs a blob name based on the user ID and original filename.
    It supports specific file extensions (PDF, DOCX, DOC) and sets the appropriate
    content type. After uploading, it generates a SAS token for read access and
    embeds it into the blob's metadata.

    Args:
        file_path (str): The local path to the file to be uploaded.
        user_id (str): The ID of the user uploading the file, used for blob path organization.
        chat_id (str): The chat ID associated with the file, stored in blob metadata.

    Returns:
        Dict: A dictionary containing:
            - "blob_url" (str): The complete URL to access the uploaded blob, including the SAS token.
            - "sas_token" (str): The generated SAS token string.
            - "blob_name" (str): The complete URL to access the uploaded blob, including the SAS token
                                 (same as "blob_url" for convenience).
            - "pdf_name" (str): The base name of the uploaded file.
            - "metadata" (Dict): The metadata set on the blob, including 'chat_id', 'sas_token',
                                 and 'IsDeleted'.

    Raises:
        ValueError: If the file has an unsupported extension.
        AzureMissingResourceHttpError: If the blob container does not exist.
        AzureError: For other Azure-specific errors during blob operations.
        IOError: If there is an issue opening or reading the local file.
    """
    blob_name_path = f"documents/{user_id}/{os.path.basename(file_path)}" # Renamed to avoid confusion
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name_path)

    extension_to_content_type = {  
    ".pdf": "application/pdf",  
    #".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # X FUTURE 
    #".doc": "application/msword",  
    ".xls": "application/vnd.ms-excel",  
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",  
    ".xlsb": "application/vnd.ms-excel.sheet.binary.macroEnabled.12",  
    ".xltx": "application/vnd.openxmlformats-officedocument.spreadsheetml.template",  
    ".xltm": "application/vnd.ms-excel.template.macroEnabled.12",  
}  
    extension = os.path.splitext(file_path)[1].lower()
    content_type = extension_to_content_type.get(extension)
    if not content_type:
        raise ValueError(f"Unsupported file extension '{extension}'. Only PDF, DOCX, and DOC files are allowed.")

    with open(file_path, "rb") as data:
        blob_client.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type)
        )

    sas_token = generate_sas_token(blob_name_path) # Use the path for SAS generation
    
    # Construct the full URL with SAS token
    full_blob_url_with_sas = f"{blob_client.url}?{sas_token}"

    metadata = {
        "chat_id": chat_id,
        "sas_token": sas_token,
        "IsDeleted": "false"
    }
    blob_client.set_blob_metadata(metadata)
    
    return {
        "blob_url": full_blob_url_with_sas,
        "sas_token": sas_token,
        "blob_name": full_blob_url_with_sas, # Now blob_name also contains the full URL with SAS
        "pdf_name": os.path.basename(file_path),
        "metadata": metadata
    }
  




#def upload_file_to_blob(file_path: str, user_id: str, chat_id: str):  
#    """  
#    Upload a file to Azure Blob Storage with the correct Content-Type and metadata.  
#    Only allows PDF, DOCX, and DOC files.  
#    Raises an exception for any other file type.  
#    """  
#    blob_name = f"documents/{user_id}/{os.path.basename(file_path)}"  
#    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)  
#    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)  
#  
#    # Map extensions to Content-Types  
#    extension_to_content_type = {  
#        ".pdf": "application/pdf",  
#        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  
#        ".doc": "application/msword",  
#    }  
#  
#    extension = os.path.splitext(file_path)[1].lower()  
#    content_type = extension_to_content_type.get(extension)  
#  
#    # Raise an Exception if not a supported file type  
#    if not content_type:  
#        raise ValueError(f"Unsupported file extension '{extension}'. Only PDF, DOCX, and DOC files are allowed.")  
#  
#    with open(file_path, "rb") as data:  
#        blob_client.upload_blob(  
#            data,  
#            overwrite=True,  
#            content_settings=ContentSettings(content_type=content_type)  
#        )  
#  
#    sas_token = generate_sas_token(blob_name)  
#  
#    metadata = {  
#        "chat_id": chat_id,  
#        "sas_token": sas_token,  
#        "IsDeleted": "false"  
#    }  
#    blob_client.set_blob_metadata(metadata)  


#def upload_file_to_blob(file_path: str, user_id: str, chat_id: str):
#    """Carica un file su Azure Blob Storage e associa un cliente nei metadati, aggiungendo un SAS Token."""
#    blob_name = f"documents/{user_id}/{os.path.basename(file_path)}"
#    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
#    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
#
## Setting content_type="application/pdf" ensures that the link opens the PDF file in the browser instead of downloading it.
#    with open(file_path, "rb") as data:
#        blob_client.upload_blob(data,
#                                 overwrite=True, 
#                                content_settings=ContentSettings(content_type="application/pdf")
#                                )
#
#
#    sas_token = generate_sas_token(blob_name)
#    # Genera il SAS Token per il file appena caricato
#    
#    
#    # Aggiorna i metadati del file con chat_id e SAS Token
#    metadata = {
#        "chat_id": chat_id,
#        "sas_token": sas_token,
#        "IsDeleted": "false"
#    }
#    blob_client.set_blob_metadata(metadata)
#
#    return {
#        "success": True,
#        "message": "File uploaded successfully.",
#        "url": blob_client.url,
#        "sas_url": f"{blob_client.url}?{sas_token}",
#        "blob_name": blob_name,
#        "metadata": metadata
#    }



def update_blob_metadata(connection_string, container_name, blob_name, metadata_key, metadata_value):
    # Esempio di utilizzo
    #connection_string = "<your_storage_connection_string>"
    #container_name = "<your_container_name>"
    #blob_name = "<your_blob_name>"
    #metadata_key = "IsDeleted"
    #metadata_value = "true"

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    metadata = blob_client.get_blob_properties().metadata
    metadata[metadata_key] = metadata_value
    blob_client.set_blob_metadata(metadata)
    print(f"Metadata '{metadata_key}' aggiornato a '{metadata_value}' per il blob '{blob_name}' nel container '{container_name}'.")




 # Verifica caricamento file:
def is_blob_exists(blob_service_client, container_name, blob_name):
    """Verifica se il file esiste nel container BLOB."""
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(container_name)
    blob_list = container_client.list_blobs(name_starts_with=blob_name)
    for blob in blob_list:
        if blob.name == blob_name:
            return True
    return False   





#from azure.storage.blob import BlobServiceClient

def move_old_files_to_archive(user_id: str):
    """Sposta i file esistenti del cliente nella cartella di archivio prima di caricare un nuovo file."""
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    # Lista di tutti i file nella cartella del cliente
    blob_list = container_client.list_blobs(name_starts_with=f"documents/{user_id}/")

    for blob in blob_list:
        old_blob_name = blob.name
        new_blob_name = old_blob_name.replace(f"documents", f"archive")

        # Copia il file nella cartella di archivio
        source_blob = container_client.get_blob_client(old_blob_name)
        destination_blob = container_client.get_blob_client(new_blob_name)
        destination_blob.start_copy_from_url(source_blob.url)

        # Cancella il file dalla cartella attiva dopo la copia
        source_blob.delete_blob()

        return f"📂 File spostato in archivio: {old_blob_name} → {new_blob_name}"









def delete_client_folder_documents(user_id: str):
    """Elimina tutti i file nella cartella f'documents/{user_id}/' per rimuoverla dal Blob Storage."""
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    # Trova tutti i file nella cartella del cliente
    blob_list = container_client.list_blobs(name_starts_with=f"documents/{user_id}/")

    for blob in blob_list:
        container_client.delete_blob(blob.name)
        print(f"🗑️ File eliminato: {blob.name}")

    return f"✅ Cartella 'documents/{user_id}/' eliminata completamente!"




def delete_client_folder_archive(user_id: str):
    """Elimina tutti i file nella cartella f'documents/{user_id}/' per rimuoverla dal Blob Storage."""
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    # Trova tutti i file nella cartella del cliente
    blob_list = container_client.list_blobs(name_starts_with=f"archive/{user_id}/")

    for blob in blob_list:
        container_client.delete_blob(blob.name)
        print(f"🗑️ File eliminato: {blob.name}")

    return f"✅ Cartella 'archived/{user_id}/' eliminata completamente!"



def delete_folder_documents():
    """Elimina tutti i file dentro 'archive/' nel Blob Storage."""
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    # Lista di tutti i blob nella cartella archive/
    blob_list = container_client.list_blobs(name_starts_with="documents/")

    files_deleted = 0  # Contatore per debug

    for blob in blob_list:
        container_client.delete_blob(blob.name)
        print(f"🗑️ File eliminato: {blob.name}")
        files_deleted += 1

    if files_deleted == 0:
        return "⚠️ Nessun file trovato nella cartella 'documents/'."
    else:
        return f"✅ Cartella 'documents/' eliminata completamente! ({files_deleted} file rimossi)"



def delete_folder_archive()->str:
    """Elimina tutti i file dentro 'archive/' nel Blob Storage."""
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    # Lista di tutti i blob nella cartella archive/
    blob_list = container_client.list_blobs(name_starts_with="archive/")

    files_deleted = 0  # Contatore per debug

    for blob in blob_list:
        container_client.delete_blob(blob.name)
        print(f"🗑️ File eliminato: {blob.name}")
        files_deleted += 1

    if files_deleted == 0:
        return "⚠️ Nessun file trovato nella cartella 'archive/'."
    else:
        return f"✅ Cartella 'archive/' eliminata completamente! ({files_deleted} file rimossi)"


  
  
def set_isdeleted_true_on_all_blobs(connection_string, container_name ):  
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)  
    container_client = blob_service_client.get_container_client(container_name)  
    blobs_list = container_client.list_blobs()  
  
    count = 0  
    for blob in blobs_list:  
        blob_client = container_client.get_blob_client(blob.name)  
        # Ottiene i metadati attuali (può essere vuoto)  
        props = blob_client.get_blob_properties()  
        metadata = props.metadata if props.metadata else {}  
        metadata["IsDeleted"] = "true"  
        blob_client.set_blob_metadata(metadata)  
        print(f"Impostato IsDeleted=true sul blob '{blob.name}'")  
        count += 1  
  
    print(f"\nTotale blob aggiornati: {count}")  