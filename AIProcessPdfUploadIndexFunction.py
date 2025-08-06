# Standard library  
import os  
import io  
from io import BytesIO  
import time  
import hashlib  
import re  
import base64  
import json  
import logging  
import uuid  
from datetime import datetime, timezone  
from typing import List, Any, Dict, Optional, Awaitable, Callable   
from dotenv import load_dotenv


# Third-party libraries  
import requests  
import fitz  
import numpy as np  
import tiktoken  
import aiohttp  
import asyncio  
import grapheme  
from PIL import Image  
import pandas as pd 
  
# Azure SDK  
import azure.functions as func  
from azure.core.credentials import AzureKeyCredential  
#from azure.keyvault.secrets import SecretClient  
#from azure.identity import DefaultAzureCredential  
#from azure.ai.textanalytics import TextAnalyticsClient 
from azure.ai.textanalytics.aio import TextAnalyticsClient #importandolo da .aio è asincrono il metodo
#from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.aio import ImageAnalysisClient 
from azure.ai.vision.imageanalysis.models import VisualFeatures  
from azure.core.exceptions import HttpResponseError, ServiceRequestError, ServiceResponseError    

# OpenAI / Mistral SDK  
import openai
#from openai import AzureOpenAI  
from openai import AsyncAzureOpenAI  
from mistralai import Mistral  
from mistralai.utils.retries import RetryConfig, BackoffStrategy
from mistralai import  OCRPageObject

  
# Altri tool  
import pymupdf4llm



import yake
import langid  

##########LIBRERIE DA AGGIUNGERE
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import markdown2 
import gc



load_dotenv()

EMBEDDING_DIM= 1536



#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################







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
AZURE_SEARCH_KEY =   os.getenv("AZURE_SEARCH_KEY")
 
 
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





DATA_SOURCE_NAME =os.getenv("DATA_SOURCE_NAME") 


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




AZURE_FUNCTION_URL= os.getenv("AZURE_FUNCTION_URL")
 
AZURE_FUNCTION_KEY= os.getenv("AZURE_FUNCTION_KEY")


AZURE_FUNCTION_URL_LOCAL=os.getenv("AZURE_FUNCTION_URL_LOCAL")
AZURE_FUNCTION_KEY_LOCAL= os.getenv("AZURE_FUNCTION_KEY_LOCAL")






#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################




EMBEDDING_DIM= 1536


GLOBAL_TIMEOUT = 900  

GLOBAL_YAKE_EXTRACTORS_CACHE = {}

# Define common YAKE! parameters
YAKE_COMMON_PARAMS = {
    "n": 3,
    "top": 10,
    "dedupLim": 0.95,
    "dedupFunc": 'seqm'
}
logging.info("YAKE! Keyword Extractor parameters defined. Instances will be cached dynamically per language.")








#################################################################################################################################################################
#################################################################################################################################################################



def check_parent_ids(record_ids: List[str], chat_id:str):  
    """  
    Checks the existence of given parent IDs in the Azure Cognitive Search index.  
  
    Args:  
        record_ids (list of str): A list of parent ID strings to check in the Azure Search index.  
        chat_id (str): The chat_id to filter by.  
    Returns:  
        tuple:  
            found (list of str): Parent IDs found in the Azure Search index.  
            not_found (list of str): Parent IDs not found in the Azure Search index.  
    """  
  
    found = []  
    not_found = []  
    headers = {  
        "Content-Type": "application/json",  
        "api-key": AZURE_SEARCH_KEY  
    }  
    url = f"https://{AZURE_NAME_SEARCH_SERVICE}.search.windows.net/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version=2024-07-01"  
    # Query che cerca solo questi parent_id e lo chat_id specificato  
    ids_odata = " or ".join([f"parent_id eq '{pid}'" for pid in record_ids])     
    # Combinazione dei filtri con l'operatore AND  
    filter_query = f"chat_id eq '{chat_id}' and ({ids_odata})"
    logging.info(f"check_parent_ids: {filter_query} ")   
    body = {  
        "search": "*",  
        "filter": filter_query,  
        "select": "parent_id"  
    }  
    response = requests.post(url, headers=headers, json=body)  
    
    results = response.json()  
    logging.info("Risposta:\n%s", json.dumps(results, indent=2, ensure_ascii=False))  
    returned_ids = set(doc["parent_id"] for doc in results.get("value", []))  
    for pid in record_ids:  
        if pid in returned_ids:  
            found.append(pid) 
            logging.info(f"RecordId {pid} trovato") 
        else:  
            not_found.append(pid)
            logging.info(f"RecordId {pid} non trovato")  
    return found, not_found 



async def wait_for_record_ids_async(results,  record_ids: list,log_suffix:str,global_timeout_time: float, timeout_seconds=215, poll_interval=5   ):  
    started = time.time()  
    not_found = list(set(record_ids)) 
    processed_records = set()  # Track which records have been processed

    while not_found and (time.time() - started < timeout_seconds):  
        if time.time() >= global_timeout_time:
            logging.warning(f"{log_suffix} Global timeout reached, stopping processing")
            break
        
        try:
            chat_id = results[0].get("chat_id", "") if results else ""  
            if chat_id == "":
                logging.error(f"{log_suffix} No chat_id found in results, cannot proceed with record ID checks")
                break      
            found, not_found_now = check_parent_ids(record_ids=not_found, chat_id=chat_id) 
            logging.info(f"Found: {found} and not found {not_found_now}")
            
            # Track newly found records
            newly_found = set(found) - processed_records
            processed_records.update(found)
            
            # Update only newly found records to avoid race conditions
            for item in results:
                if item["RecordId"] in newly_found:
                    logging.info(f"Marking as processed RecordId: {item['RecordId']}")  
                    item["is_processed"] = True
                    
            if found:  
                logging.info(f"Found {len(found)} RecordId: {found}")  
            if not_found_now:  
                logging.info(f"Still to find {len(not_found_now)} RecordId: {not_found_now}")  
            if not_found_now:  
                await asyncio.sleep(poll_interval)  
            not_found = not_found_now
        except Exception as e:
            logging.error(f"{log_suffix} Error in wait_for_record_ids_async: {e}")
            break
            
    return not_found, results   





#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################



def encode_pdf(pdf_file_obj):
    """  
    Encodes the contents of a PDF file object into a base64 string.  
  
    Args:  
        pdf_file_obj (io.BytesIO): The PDF file object to encode.  
  
    Returns:  
        str or None: Base64-encoded string representation of the PDF file,  
        or None if encoding fails.  
    """
    try:
        return base64.b64encode(pdf_file_obj.getvalue()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding PDF: {e}")
        return None



async def get_ocr_markdown_chunks_pdf_mistral_async(pdf_file_obj: io.BytesIO,
                                                    global_timeout_time: float,
                                                     sem_ocr: asyncio.Semaphore,
                                                       client_mistral : Mistral,
                                                       timeout: int = 120)-> Optional[dict]:
    """  
    Asynchronously performs OCR on a PDF file using the Mistral OCR API, returning markdown-formatted results.  
  
    Args:  
        pdf_file_obj (io.BytesIO): The PDF file to process.  
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent OCR requests.  
        global_timeout_time (float): Global timeout for the OCR request.  
        client_mistral (Mistral): Mistral client instance.  
        timeout (int, optional): Timeout for the individual OCR request. Defaults to 120.  

    Returns:  
        dict or None: OCR result in markdown chunks, or None if processing or encoding fails.  
    """ 
    base64_pdf = encode_pdf(pdf_file_obj)
    if not base64_pdf:
        return None
 
    if global_timeout_time:  
        time_left = global_timeout_time - time.time()  
        if time_left <= 0:  
            logging.error("Global timeout expired in get_ocr_markdown_chunks_pdf_mistral_async.")  
            return None 
    else:  
        time_left = timeout  
    effective_timeout = min(timeout, time_left)

    try:
        async with sem_ocr: 
            res = await asyncio.wait_for( 
                client_mistral.ocr.process_async(
                model=DEPLOYMENT_MISTRAL_OCR,
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{base64_pdf}"
                },
                include_image_base64=True
            ),  
            timeout=effective_timeout
            )
            return res


    except Exception as e:
        error_str = str(e).lower()
        error_type = type(e).__name__
        
        # Check for rate limiting (429)
        if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
            logging.error(f"[get_ocr_markdown_chunks_pdf_mistral] Rate limit reached (429): {e}")
        
        # Check for authentication errors (401)
        elif "401" in error_str or "unauthorized" in error_str or "authentication" in error_str:
            logging.error(f"[get_ocr_markdown_chunks_pdf_mistral] Authentication error (401): {e}")
        
        # Check for permission errors (403)
        elif "403" in error_str or "forbidden" in error_str or "permission denied" in error_str:
            logging.error(f"[get_ocr_markdown_chunks_pdf_mistral] Permission denied (403): {e}")
        
        # Check for not found errors (404)
        elif "404" in error_str or "not found" in error_str:
            logging.error(f"[get_ocr_markdown_chunks_pdf_mistral] Resource not found (404): {e}")
        
        # Check for bad request errors (400)
        elif "400" in error_str or "bad request" in error_str:
            logging.error(f"[get_ocr_markdown_chunks_pdf_mistral] Bad request (400): {e}")
        
        # Check for unprocessable entity (422)
        elif "422" in error_str or "unprocessable" in error_str:
            logging.error(f"[get_ocr_markdown_chunks_pdf_mistral] Unprocessable entity (422): {e}")
        
        # Check for server errors (5xx)
        elif any(code in error_str for code in ["500", "502", "503", "504"]) or "server error" in error_str:
            logging.error(f"[get_ocr_markdown_chunks_pdf_mistral] Server error: {e}")
        
        # Check for timeout errors
        elif "timeout" in error_str or "timed out" in error_str:
            logging.error(f"[get_ocr_markdown_chunks_pdf_mistral] Request timeout: {e}")
        
        # Check for connection errors
        elif "connection" in error_str or "network" in error_str or "unreachable" in error_str:
            logging.error(f"[get_ocr_markdown_chunks_pdf_mistral] Connection error: {e}")
        
        # Check for data parsing errors
        elif error_type in ["KeyError", "ValueError", "TypeError", "IndexError"]:
            logging.error(f"[get_ocr_markdown_chunks_pdf_mistral] Data parsing error ({error_type}): {e}")
        
        # Generic error
        else:
            logging.error(f"[get_ocr_markdown_chunks_pdf_mistral] Unknown error ({error_type}): {e}")
        
        return None
    except asyncio.CancelledError:  
        logging.warning("describe_image_with_gpt41_async: Task was cancelled")  
        return None




async def describe_image_with_gpt41_async(base64_image: str,
                                           client: AsyncAzureOpenAI,
                                             sem: asyncio.Semaphore,
                                              global_timeout_time: float,
                                               timeout: int = 120) -> str:
    """  
    Asynchronously generates a brief image description using GPT-4.1 via Azure OpenAI.  
  
    Args:  
        base64_image (str): Base64-encoded PNG image string.  
        client (AsyncAzureOpenAI): Async Azure OpenAI client.  
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests.  
  
    Returns:  
        str: Short description of the image or an error message.  
    """  
    system = """You are a virtual assistant who extracts information from images of PDF pages using OCR.  
    Your goal is to convert the recognized content into a structured Markdown format.  

    Your answer must always include the following three sections, strictly in this order:

    ### Markdown Text  
    This section contains the plain text content extracted from the page, formatted in Markdown. If there are images, insert them in the appropriate place using the syntax:
    `![]({image_id})`  
    `<!-- {description} -->`  
    where `{image_id}` must follow the format `image_{page_idx + 1}_{image_counter}`. The image ID is generated programmatically and refers to the order of appearance on the current page.

    ### Page Description  
    Provide a short description of the layout and contents of the page, including any notable visual elements such as charts, diagrams, or decorative images.

    ### Markdown Tables  
    Convert any table found on the page into valid Markdown table format.  
    If there are no tables found, write: `No tables found.`

    Be accurate and preserve the visual and semantic structure of the original page as much as possible. """
    message = [{"role": "system", "content": system}]
    message.append({
        "role": "user", "content": [
            {"type": "text", "text": "Analyze the following pdf image:\n"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        ]
    })

    if global_timeout_time:  
        time_left = global_timeout_time - time.time()  
        if time_left <= 0:  
            logging.error("describe_image_with_gpt41_async: Global timeout expired.")  
            return ""  
    else:  
        time_left = timeout  # fallback  

    effective_timeout = min(timeout, time_left) 


    try:
        async with sem: 
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=AZURE_OPENAI_LLM_DEPLOYMENT,
                    messages=message
                ),
                timeout=effective_timeout
            )
            return response.choices[0].message.content
    except openai.RateLimitError as e:  
        logging.error(f"describe_image_with_gpt41_async:Rate limit reached (429): {e}")  
        return ""  
  
    except openai.APITimeoutError as e:  
        logging.error(f"describe_image_with_gpt41_async:Request timeout: {e}")  
        return ""  
  
    except openai.APIConnectionError as e:  
        logging.error(f"describe_image_with_gpt41_async:API connection error: {e}")  
        return ""  
  
    except openai.AuthenticationError as e:  
        logging.error(f"describe_image_with_gpt41_async:Authentication error: {e}")  
        return ""  
  
    except openai.PermissionDeniedError as e:  
        logging.error(f"describe_image_with_gpt41_async:Permission denied: {e}")  
        return ""  
  
    except openai.BadRequestError as e:  
        logging.error(f"describe_image_with_gpt41_async:Bad request (400): {e}")  
        return ""  
  
    except openai.NotFoundError as e:  
        logging.error(f"describe_image_with_gpt41_async:Resource not found (404): {e}")  
        return ""  
  
    except openai.UnprocessableEntityError as e:  
        logging.error(f"describe_image_with_gpt41_async:Unprocessable entity (422): {e}")  
        return ""
    except asyncio.CancelledError:  
        logging.warning("describe_image_with_gpt41_async: Task was cancelled")  
        return ""


async def get_info_image_with_azure_async(client_ImageAnalysis:ImageAnalysisClient ,
                                           image_stream,
                                             sem_ocr: asyncio.Semaphore,
                                             global_timeout_time: float,
                                             timeout: int = 120
                                             ) -> str:
    """
    Analyzes an image stream asynchronously using Azure Image Analysis to extract various visual features.

    This function utilizes Azure Cognitive Services to perform optical character recognition (OCR), 
    tagging, object detection, and people detection on the provided image. It respects a semaphore 
    to control the concurrency of requests to the Azure service.

    Args:
        client_ImageAnalysis (ImageAnalysisClient): An authenticated Azure Image Analysis client.
        image_stream: The image data as a stream (e.g., bytes, file-like object).
        sem_ocr (asyncio.Semaphore): An asyncio semaphore to limit concurrent OCR operations.

    Returns:
        str: A formatted string containing the extracted information. 
             This includes recognized text, tags, detected objects, and the count of people.
             Returns "Image content not clearly identifiable" if no significant information is found.

    Notes:
        - The function currently uses `VisualFeatures.READ`, `VisualFeatures.TAGS`, 
          `VisualFeatures.OBJECTS`, and `VisualFeatures.PEOPLE`.
        - Features like `CAPTION` and `DENSE_CAPTIONS` are commented out due to regional limitations 
          (e.g., 'Sweden Central') but are included as a placeholder for future use.
        - Only tags and objects with a confidence score greater than 0.6 are included in the output.
        - For people detection, it reports both the count of high-confidence detections and total detections.
    """

    if global_timeout_time:  
        time_left = global_timeout_time - time.time()  
        if time_left <= 0:  
            logging.error("get_info_image_with_azure_async: Global timeout expired.")  
            return ""  
    else:  
        time_left = timeout  # fallback  
  
    effective_timeout = min(timeout, time_left)  


    try:
        async with sem_ocr: 
            result = await asyncio.wait_for( 
                client_ImageAnalysis.analyze(
                image_data=image_stream,
                visual_features=[
                    #VisualFeatures.CAPTION, ### purtroppo non supportato nella regione sweden central ma apppena lo rendono utilizzabile da Usare
                    #VisualFeatures.DENSE_CAPTIONS,  ### purtroppo non supportato nella regione sweden central ma apppena lo rendono utilizzabile da Usare
                    VisualFeatures.READ,
                    VisualFeatures.TAGS,
                    VisualFeatures.OBJECTS,
                    #VisualFeatures.SMART_CROPS, ###forse per cose future
                    VisualFeatures.PEOPLE
                            ]
                        ), 
                    timeout=effective_timeout
                    )

        output = []

    #    if result.caption and result.caption.confidence > 0.5:
    #        output.append(f"Caption: '{result.caption.text}'\n") ###, Confidence: {result.caption.confidence:.4f}

    #    if result.dense_captions and result.dense_caption.confidence > 0.5:
    #        output.append("Dense Captions:")
    #        for caption in result.dense_captions.list:
    #            output.append(f" Caption text: '{caption.text}'") ###, {caption.bounding_box}, Confidence: {caption.confidence:.4f}
    #        output.append("")

        if result.read:
            text_parts =[]
            output.append("Read:")
            for block in result.read.blocks:
                for line in block.lines:
                    if line.text.strip():  # Evita testo vuoto
                        text_parts.append(line.text.strip())
            if text_parts:
                output.append(f"Text: {' '.join(text_parts)}")


        if result.tags:
            high_conf_tags = [tag.name for tag in result.tags.list if tag.confidence > 0.6]
            if high_conf_tags:
                output.append(f"Tags: {', '.join(high_conf_tags)}") 

        if result.objects:
            objects = []
            for obj in result.objects.list:
                if obj.tags and obj.tags[0].confidence > 0.6:
                    objects.append(obj.tags[0].name)
            if objects:
                output.append(f"Objects: {', '.join(set(objects))}") 

        if result.people:
            people_count = len(result.people.list)
            high_conf_people = [p for p in result.people.list if p.confidence > 0.6]
            if high_conf_people:
                output.append(f"People: {len(high_conf_people)} detected and Total People detected {people_count}")

    #    if result.smart_crops:
    #        output.append("Smart Cropping:")
    #        for crop in result.smart_crops.list:
    #            output.append(f"  Aspect ratio: {crop.aspect_ratio}, Smart crop: {crop.bounding_box}")

        return "\n".join(output) if output else ""
    except asyncio.TimeoutError:  
        logging.error("get_info_image_with_azure_async: Azure ImageAnalysis request timed out.")  
        return ""  
    except asyncio.CancelledError:  
        logging.warning("get_info_image_with_azure_async: Task was cancelled")  
        return ""
    except Exception as e:  
        logging.error(f"get_info_image_with_azure_async: Errore durante l'analisi immagine Azure: {type(e).__name__}: {e}")  
        return "" 














#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
 
async def get_embedding_async(  
    text: str,  
    global_timeout_time: float,  
    client,  # → AsyncAzureOpenAI  
    model_deployment: str,  
    sem_embedding: asyncio.Semaphore,  
    dimensions: int=1536,  
    timeout: int=120,  
) -> Optional[list]:  
 
    if global_timeout_time:  
        time_left = global_timeout_time - time.time()  
        if time_left <= 0:  
            logging.error("get_embedding_async: Global timeout expired.")  
            return []
    else:  
        time_left = timeout  # fallback  
  
    effective_timeout = min(timeout, time_left)  
    try:
        async with sem_embedding:   
            response = await asyncio.wait_for(  
                client.embeddings.create(  
                    input=[text],  
                    dimensions=dimensions,  
                    model=model_deployment,  
                    timeout=effective_timeout  # timeout interno della chiamata  
                ),  
                timeout=effective_timeout     # timeout totale di attesa per la await  
            )  
            return response.data[0].embedding  
    except asyncio.TimeoutError:  
        logging.error("get_embedding_async: Timeout triggered during embeddings.create.")  
        return []
    except Exception as e:  
        logging.error(f"get_embedding_async: Embedding failed: {e.__class__.__name__}: {e}")  
        return [] 
    except asyncio.CancelledError:  
        logging.warning("get_embedding_async: Task was cancelled")  
        return [] 

  

async def chunk_text_async(text:str, max_tokens:int=8000, overlap:int=100, encoding_name:int="cl100k_base"):  
    """  
    Asynchronously splits a text into chunks of up to max_tokens tokens, with overlap.  
  
    Args:  
        text (str): Input text to split.  
        max_tokens (int): Maximum tokens allowed per chunk.  
        overlap (int): Token overlap between chunks.  
        encoding_name (str): Tokenizer encoding name.  
  
    Returns:  
        list[str]: List of text chunks.  
    """  
    encoding = tiktoken.get_encoding(encoding_name)  
    tokens = encoding.encode(text)  
    chunks = []  
    start = 0  
    while start < len(tokens):  
        end = min(start + max_tokens, len(tokens))  
        chunk = encoding.decode(tokens[start:end])  
        chunks.append(chunk)  
        start += max_tokens - overlap  
    return chunks  
  
async def Matryoshka_embedding_async(  
    text:str,  
    client: AsyncAzureOpenAI,  
    model_deployment:str, 
    global_timeout_time:float,
    sem_embedding: asyncio.Semaphore, 
    dimensions:int=1536,  
    chunk_size:int=8000,  
    overlap:int=100, 
):  
    """  
    Asynchronously computes an embedding for long texts by recursively splitting the text into chunks,  
    processing each chunk (and sub-chunk if still too large), and averaging the embeddings into a final vector.  
  
    Args:  
        text (str): The input text to embed.  
        client (AsyncAzureOpenAI): Async Azure OpenAI client for embeddings.  
        model_deployment (str): The specific deployment to use for the embedding model.  
        dimensions (int): Output dimensionality of the embedding.  
        chunk_size (int): Maximum number of tokens in each chunk.  
        overlap (int): Number of overlapping tokens between chunks.  
  
    Returns:  
        list[float]: The averaged embedding vector for the input text.  
    """  
    time_left = global_timeout_time - time.time()  
    if time_left <= 5:  
        logging.warning("Global timeout less than 5 seconds.") 
        return [] 
    encoding = tiktoken.get_encoding("cl100k_base")  
    if len(encoding.encode(text)) <= chunk_size:  
        return await get_embedding_async(text= text,global_timeout_time= global_timeout_time, client= client,model_deployment= model_deployment,dimensions= dimensions,
                                         sem_embedding= sem_embedding)  
  
    chunks = await chunk_text_async(text, max_tokens=chunk_size, overlap=overlap)  
    
    # Lanciare tutte le embedding in parallelo (gather)  
    async def _embed(chunk):  
        if len(encoding.encode(chunk)) <= chunk_size:  
            return await get_embedding_async(text=chunk, global_timeout_time=global_timeout_time, client= client,model_deployment= model_deployment,dimensions= dimensions,
                                         sem_embedding= sem_embedding)  
        else:  
            return await Matryoshka_embedding_async(text=chunk, client= client,model_deployment= model_deployment, global_timeout_time=global_timeout_time, dimensions= dimensions,
                                         sem_embedding= sem_embedding)  
    time_left = global_timeout_time - time.time()  
    if time_left <= 5:  
        logging.warning("Global timeout less than 5 seconds.") 
        return [] 
  

    try:  
        embeddings = await asyncio.wait_for( 
            asyncio.gather(*[_embed(chunk) for chunk in chunks]),
            timeout= time_left
        ) 
        if not embeddings:  
            return [] 
        return np.mean(embeddings, axis=0).tolist() 

    except asyncio.TimeoutError:  
        logging.warning("Matryoshka_embedding_async: Global timeout reached while waiting for embeddings.")  
        # Potresti decidere di restituire ciò che hai raccolto finora, oppure []  
        return ["Blocking error"] 
    except Exception as e:  
        logging.error(f"Matryoshka_embedding_async: Error during batch embedding: {e}")    
        return ["Blocking error"] 
    except asyncio.CancelledError:  
        logging.warning("Matryoshka_embedding_async: Task was cancelled")
        return ["Blocking error"]


#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################



def chunk_text_to_docs_extract_key_phrases_inputs(text, language, max_elements=5120, overlap=1000, batch_size=10):  
    """  
    Splits text into character-based chunks for Azure Text Analytics key phrase extraction,  
    wrapping each chunk as a dictionary ready for API input, with batching.  
  
    Args:  
        text (str): The input text to split into chunks.  
        language (str): The language code of the text.  
        max_elements (int): Maximum number of characters per chunk.  
        overlap (int): Number of overlapping characters between consecutive chunks.  
        batch_size (int): Number of items per batch sent to the API.  
  
    Returns:  
        list[list[dict]]: Batches of chunk dicts, each with an "id", "text", and "language".  
    """    
    docs = []  
    text_elements = list(grapheme.graphemes(text))  
    start = 0  
    while start < len(text_elements):  
        end = min(start + max_elements, len(text_elements))  
        chunk = ''.join(text_elements[start:end])  
        doc = {  
            "id": str(uuid.uuid4()),  
            "text": chunk,  
            "language": language  
        }  
        docs.append(doc)  
        start += max_elements - overlap  
  
    # Split docs in batch sottoliste di max batch_size elementi  
    docs_batches = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]  
    return docs_batches



def detected_language_with_langid_async(text: str) -> str:  
    """ Detects language using langid as a fallback mechanism."""
    # Fallback sync con langid (non ha api async) 
    logging.info("Falback to langid for language detection")  
    try:  
        lang, score = langid.classify(text)  
        logging.info(f"detect_language_with_langid_async: Detected language via langid: {lang} (score: {score})")  
        return lang  
    except Exception as e:  
        logging.error(f"detect_language_with_langid_async: langid fallback also failed: {e}")  
        return "en"  




async def detect_language_with_fallback_async(text:str,
                                               text_analytics_client:TextAnalyticsClient,
                                               global_timeout_time: float,
                                               timeout: int = 120) -> str:  
    """Detects language using Azure Text Analytics (async), falls back to langid if Azure fails."""  

    if not text or not text.strip():
        logging.warning("Empty or whitespace-only text provided for language detection")
        return "en"

    if global_timeout_time:  
        time_left = global_timeout_time - time.time()  
        if time_left <= 0:  
            logging.error("detect_language_with_fallback_async: Global timeout expired.")  
            return "en"  # Default fallback language 
    else:  
        time_left = timeout  # fallback 

    effective_timeout = min(timeout, time_left) 

    if text_analytics_client:  
        try:  

            language_result = await asyncio.wait_for(text_analytics_client.detect_language(  
                documents=[{"id": "1", "text": text[:4900]}]  
                ), 
            timeout=effective_timeout 
            )              
            if (language_result 
                and len(language_result) > 0 
                and not language_result[0].is_error 
                and hasattr(language_result[0], "primary_language")
                and language_result[0].primary_language
                and hasattr(language_result[0].primary_language, "iso6391_name")):                              
              
                detected_language = language_result[0].primary_language.iso6391_name  
                confidence = getattr(language_result[0].primary_language, "confidence_score", 0)
                logging.info(f"Detected language via Azure: {detected_language} (confidence: {confidence:.2f})")   
                return detected_language  
            else:  
                logging.warning("detect_language_with_fallback_async: Azure failed to detect language or returned an error.") 
            return detected_language_with_langid_async(text)                 
        except Exception as e:  
            logging.error(f"detect_language_with_fallback_async: Azure language detection failed due to Exception: {e}")  
            return detected_language_with_langid_async(text)
        except asyncio.CancelledError:  
            logging.warning("detect_language_with_fallback_async: Task was cancelled")  
            return detected_language_with_langid_async(text)







# Utility function to extract keywords using YAKE! as a fallback
def extract_keywords_with_yake(text: str, language: str, num_keywords: int = 10) -> list[str]:
    """
    Extracts keywords from a text using YAKE! as a fallback mechanism.
    It uses a cache to store YAKE! KeywordExtractor instances per language,
    avoiding repeated initialization.
    """
    try:
        # Check if an extractor for this language already exists in the cache
        if language not in GLOBAL_YAKE_EXTRACTORS_CACHE:
            logging.info(f"Initializing YAKE! KeywordExtractor for language '{language}' and adding to cache.")
            # If not, create a new instance and store it
            GLOBAL_YAKE_EXTRACTORS_CACHE[language] = yake.KeywordExtractor(
                lan=language,
                n=YAKE_COMMON_PARAMS["n"],
                top=YAKE_COMMON_PARAMS["top"],
                dedupLim=YAKE_COMMON_PARAMS["dedupLim"],
                dedupFunc=YAKE_COMMON_PARAMS["dedupFunc"]
            )
        
        # Retrieve the extractor from the cache
        kw_extractor = GLOBAL_YAKE_EXTRACTORS_CACHE[language]
        
        keywords_with_scores = kw_extractor.extract_keywords(text)
        # YAKE! returns a list of (keyword, score) tuples. Extract only the keyword strings.
        # Note: In YAKE!, a lower score indicates higher relevance.
        return [kw for kw, score in keywords_with_scores[:num_keywords]]
    except Exception as e:
        # Log any errors during YAKE! execution
        logging.error(f"Error during keyword extraction with YAKE! (fallback) for language '{language}': {e}")
        return []  


async def extract_keywords_with_fallback_async(
    text: str,
    language: str,
    text_analytics_client: TextAnalyticsClient,
    global_timeout_time: float,
    timeout: int = 120,
    max_retries: int = 3,
    retry_wait_sec: int = 5
) -> list:
    """
    Extracts keywords using Azure Text Analytics with aggressive 429 handling.
    On rate limit (429), immediately falls back to YAKE without retries.
    
    Args:
        text: Text to analyze
        language: Language code (e.g., 'en', 'it')
        text_analytics_client: Azure Text Analytics client (can be None)
        max_retries: Maximum retry attempts for transient errors (not 429)
        retry_wait_sec: Wait time between retries
    
    Returns:
        list: List of unique keywords/key phrases
    """
    
    # Input validation
    if not text or not text.strip():
        logging.warning("extract_keywords_with_fallback_async: Empty or whitespace-only text provided for keyword extraction")
        return []




    # Try Azure Text Analytics first
    if text_analytics_client:
        list_documents_for_ta = chunk_text_to_docs_extract_key_phrases_inputs(
            text=text,
            language=language
        )

        if global_timeout_time:  
            time_left = global_timeout_time - time.time()  
        if time_left <= 0:  
            logging.error("extract_keywords_with_fallback_async: Global timeout expired.")  
            raise []
        else:  
            time_left = timeout  # fallback  
    
        effective_timeout = min(timeout, time_left)

        all_keywords = []
        rate_limit_hit = False
        
        # Process each document batch
        for documents_batch in list_documents_for_ta:
            if rate_limit_hit:
                break  # Stop processing if we hit rate limit
                
            retries = 0
            batch_success = False
            
            while retries < max_retries and not rate_limit_hit:
                if global_timeout_time < time.time():
                    logging.error("extract_keywords_with_fallback_async:Global timeout expired in extract_keywords_with_fallback_async")
                    return []
                try:
                    result = await asyncio.wait_for(
                        text_analytics_client.extract_key_phrases(
                        documents=documents_batch
                        ),  
                    timeout=effective_timeout
                    )    

                    # Process successful results
                    batch_keywords = []
                    for doc_result in result:
                        if not doc_result.is_error:
                            batch_keywords.extend(doc_result.key_phrases)
                        else:
                            logging.warning(f"extract_keywords_with_fallback_async: Document processing error: {doc_result.error}")
                    
                    all_keywords.extend(batch_keywords)
                    batch_success = True
                    break  # Success, exit retry loop for this batch
                    
                except (HttpResponseError, ServiceRequestError, ServiceResponseError) as e:
                    # AGGRESSIVE 429 handling - immediate fallback
                    if hasattr(e, "status_code") and e.status_code == 429:
                        logging.warning("extract_keywords_with_fallback_async: Azure rate limit reached (429). Immediately falling back to YAKE.")
                        rate_limit_hit = True
                        break  # Exit retry loop and trigger fallback




                    # Handle other retryable errors
                    msg = str(e).lower()
                    if ("timeout" in msg or "timed out" in msg or 
                        isinstance(e, ServiceRequestError)):
                        retries += 1
                        if retries < max_retries:
                            logging.warning(f"Timeout (attempt {retries}/{max_retries}): {e}")
                            await asyncio.sleep(retry_wait_sec)
                        else:
                            logging.error(f"Max retries reached for batch. Error: {e}")
                            break  # Exit retry loop, continue with next batch
                    else:
                        logging.error(f"Non-retryable Azure error: {e}")
                        break  # Exit retry loop, continue with next batch
                
                except Exception as e:
                    logging.error(f"Unexpected error in Azure extraction: {e}")
                    break  # Exit retry loop, continue with next batch
                except asyncio.CancelledError:  
                    logging.warning("extract_keywords_with_fallback_async: Task was cancelled")  
                    break    

        # If we got some keywords and didn't hit rate limit, return them
        if all_keywords and not rate_limit_hit:
            logging.info(f"Azure extraction successful. Found {len(all_keywords)} keywords.")
            return list(set(all_keywords))
        
        # Log why we're falling back
        if rate_limit_hit:
            logging.info("Falling back to YAKE due to rate limit")
        else:
            logging.info("Azure returned no keywords, falling back to YAKE")
    
    else:
        logging.info("Text Analytics client not available, using YAKE fallback")

    if global_timeout_time < time.time():
        logging.error("Global timeout expired in extract_keywords_with_fallback_async")
        return []

    # Fallback to YAKE
    try:
        keywords = extract_keywords_with_yake(text, language)
        if keywords:
            logging.info(f"YAKE fallback successful. Found {len(keywords)} keywords.")
            return list(set(keywords))
        else:
            logging.warning("YAKE fallback did not find keywords")
    except Exception as e:
        logging.error(f"YAKE fallback failed: {e}")
    
    return []



async def process_single_chunk_for_keywords_async(chunk: Dict[str, Any],
                                                   detected_language: str,
                                                     text_analytics_client:TextAnalyticsClient,
                                                     global_timeout_time :float) -> Dict[str, Any]:  
    """
    Processes a single text chunk to extract keywords using an asynchronous Azure Text Analytics pipeline.

    This function takes a dictionary representing a text chunk, extracts its raw text content,
    and then attempts to identify key phrases using the `extract_keywords_with_fallback_async`
    function. It also incorporates any headers found within the chunk as keywords.

    Args:
        chunk (Dict[str, Any]): A dictionary containing the chunk's data. Expected keys include:
            - "text_raw" (str): The raw text content of the chunk.
            - "metadata" (Dict[str, Any]): Metadata associated with the chunk,
              potentially including a "page" number.
            - "base64_imgs_list" (List[str]): A list of Base64 encoded images found in the chunk.
            - "headers" (Dict[str, str]): A dictionary of headers found in the chunk,
              where keys are header levels (e.g., "h1", "h2") and values are header texts.
            - "text_markdown" (str): The markdown formatted text content of the chunk.
        detected_language (str): The language of the text content within the chunk (e.g., "en", "it").
        text_analytics_client (TextAnalyticsClient): An authenticated Azure Text Analytics client
                                                     (assumed to be an aio.TextAnalyticsClient instance).

    Returns:
        Dict[str, Any]: A dictionary containing the original chunk data augmented with:
            - "keywords" (List[str]): A list of unique keywords extracted from the text content
              and headers.
            - "detected_language" (str): The language that was used for keyword extraction.

    Notes:
        - Keyword extraction is skipped if `text_content` is empty or only contains whitespace.
        - The `extract_keywords_with_fallback_async` function handles the actual
          Azure Text Analytics API call and any fallback logic (e.g., to YAKE!).
        - Headers are directly added to the list of keywords.
        - The returned "keywords" list contains only unique entries.
    """
    text_content = chunk.get("text_raw", "")  
    page_number = chunk.get("metadata", {}).get("page", "unknown")  
    base64_imgs_list = chunk.get("base64_imgs_list", [])  
    headers= chunk.get("headers", {})
    logging.debug(f"Processing chunk page {page_number}, text length: {len(text_content)}")  
  
    key_phrases = []  

    if text_content and text_content.strip():   
        try:  
            key_phrases = await extract_keywords_with_fallback_async(  
                text= text_content,  
                language=detected_language,  
                text_analytics_client= text_analytics_client, 
                global_timeout_time=global_timeout_time 
            )  
        except Exception as e:  
            logging.error(f"Error during async key phrase extraction for page {page_number}: {e}")  
    else:  
        logging.info(f"Skipping keyword extraction for page {page_number}: empty text.") 

    if headers: 
        for header_level, header_text in chunk.get("headers").items():
            key_phrases.append(header_text)
    


    return {  
        "text_raw": text_content,
        "text_markdown": chunk.get("text_markdown", ""),
        "metadata": chunk.get("metadata", {}),  
        "base64_imgs_list": base64_imgs_list,  
        "keywords": list(set(key_phrases)),
        "detected_language": detected_language
    }  




def markdown_to_raw(md_text: str) -> str:
    """
    Converts a Markdown string to its raw text equivalent by first converting it to HTML
    and then stripping all HTML tags.

    Args:
        md_text (str): The input string in Markdown format.

    Returns:
        str: The raw text content extracted from the Markdown, with leading/trailing whitespace removed.
    """
    html = markdown2.markdown(md_text)
    raw = re.sub('<[^<]+?>', '', html)
    return raw.strip()


def split_markdown_and_raw(markdown_text, chunk_size=600, chunk_overlap=200):
    """
    Splits a given Markdown text into smaller chunks, preserving header context
    and converting each chunk to raw text.

    The splitting process involves two main steps:
    1. Splitting the Markdown text by specified header levels (H1 and H2).
    2. Further splitting the resulting Markdown sections into fixed-size chunks
       with a defined overlap using a recursive character splitter.

    For each final chunk, both the Markdown content and its raw text equivalent
    are provided, along with any associated headers.

    Args:
        markdown_text (str): The input text in Markdown format.
        chunk_size (int, optional): The desired maximum size for each text chunk. Defaults to 600.
        chunk_overlap (int, optional): The number of characters to overlap between consecutive chunks. Defaults to 200.

    Returns:
        list: A list of dictionaries, where each dictionary represents a chunk and contains:
            - "markdown" (str): The Markdown content of the chunk.
            - "raw" (str): The raw text content of the chunk (HTML tags stripped).
            - "headers" (dict): A dictionary of headers associated with the chunk.

    Logs:
        - INFO: The total number of chunks created.
        - WARNING: If an empty or whitespace-only chunk is encountered and skipped.
    """
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
#        ("###", "Header 3"),
#        ("####", "Header 4"),
    ]

    # MD splits
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False,
    )
    md_header_splits = markdown_splitter.split_text(markdown_text)
    md_header_splits
    # Char-level splits
 


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Split
    splits = text_splitter.split_documents(md_header_splits)
    results = []
    for chunk in splits:
        markdown= chunk.page_content
        if not markdown or markdown.strip() == "":
            logging.warning(f"Skipping empty or whitespace chunk. Metadata: {chunk.metadata}")
            continue
        headers= chunk.metadata
        
        raw = markdown_to_raw(markdown)
        results.append({
            "markdown": markdown,
            "raw": raw,
            "headers": headers,
        })

    logging.info(f"Document split into {len(results)} chunks.")
    return results
   


def make_thumbnail_base64(base64_str, thumb_size=(200, 200), format="JPEG"):  

    print(base64_str[:20])
    # Rimuovi eventuale header base64  
    if base64_str.startswith('data:image'):  
        base64_str = base64_str.split(',', 1)[1]  
    print(base64_str[:20])

    try:  
        image_bytes = base64.b64decode(base64_str)  
    except Exception as e:  
        print("Errore nella decodifica base64:", e)  
        raise  
  
    try:  
        with Image.open(io.BytesIO(image_bytes)) as img:  
            orig_size = img.size  
            print(f"Original image size (pixels): {orig_size}")  
            print(f"Original image size (bytes): {len(image_bytes)}")  
  
            img = img.convert("RGB")  
            img.thumbnail(thumb_size)  
            thumb_size_actual = img.size  
            print(f"Thumbnail size (pixels): {thumb_size_actual}")  
  
            buf = io.BytesIO()  
            img.save(buf, format=format)  
            thumb_bytes = buf.getvalue()  
            print(f"Thumbnail size (bytes): {len(thumb_bytes)}")  
  
            # Calcola la riduzione percentuale  
            perc_pixel = 100 * (thumb_size_actual[0] * thumb_size_actual[1]) / (orig_size[0] * orig_size[1])  
            perc_bytes = 100 * len(thumb_bytes) / len(image_bytes)  
            print(f"Riduction pixel: {perc_pixel:.1f}% from the original size")  
            print(f"Riduction bytes: {perc_bytes:.1f}% from the original size")  
  
            thumb_base64 = base64.b64encode(thumb_bytes).decode('utf-8')  
            return thumb_base64  
  
    except Exception as e:  
        print("Errore nell'apertura o conversione dell'immagine:", e)  
        with open("immagine_non_valida.bin", "wb") as f:  
            f.write(image_bytes)  
        raise   


async def _process_single_mistral_page_async(
    page: OCRPageObject,
    page_idx: int,
    client_azure_openai: AsyncAzureOpenAI,
    client_ImageAnalysis: ImageAnalysisClient,
    sem: asyncio.Semaphore,
    sem_ocr:asyncio.Semaphore,
    global_timeout_time: float
) -> List[Dict[str, Any]]:
    """
    Asynchronously processes a single page object obtained from Mistral OCR.

    This function extracts text and image information from the page. For each image,
    it either generates a description using Azure OpenAI's GPT-4V (if the page text is empty)
    or extracts visual features using Azure Image Analysis (if there is text content).
    Image descriptions are then embedded back into the Markdown text as HTML comments.
    Finally, the updated Markdown text is split into smaller chunks (both Markdown and raw text)
    and returned.

    Args:
        page (OCRPageObject): An object representing a single page from Mistral OCR,
                              containing markdown text and image data.
        page_idx (int): The 0-based index of the current page being processed.
        client_azure_openai (AsyncAzureOpenAI): An asynchronous Azure OpenAI client for GPT-4V image descriptions.
        client_ImageAnalysis (ImageAnalysisClient): An Azure Image Analysis client for extracting visual features.
        sem (asyncio.Semaphore): A semaphore to control concurrency for GPT-4V calls.
        sem_ocr (asyncio.Semaphore): A semaphore to control concurrency for Azure Image Analysis (OCR-related) calls.

    Returns:
        Dict[str, Any]: A list of dictionaries, where each dictionary represents a processed
                        chunk from the page. Each chunk dictionary contains:
            - "text_raw" (str): The raw text content of the chunk.
            - "text_markdown" (str): The Markdown content of the chunk with image descriptions embedded.
            - "metadata" (Dict[str, Any]): Metadata for the chunk, including the page number.
            - "base64_imgs_list" (List[str]): A list of Base64 encoded images found in the original chunk.
            - "headers" (Dict[str, str]): Headers associated with the chunk.

    Logs:
        - DEBUG: Information about image paths not found or missing base64 data.
        - ERROR: Errors encountered during image preparation for Azure Image Analysis or image description tasks.
    """
    print("mistral")
    text = page.markdown
    base64_images_in_chunk = []
    image_map = {img.id: img for img in page.images}
    
    image_description_tasks = []
    image_descriptions = {}
    matches_to_process = []


    for match in re.finditer(r"!\[(?P<alt_text>.*?)\]\((?P<image_path>.*?)\)", text, re.IGNORECASE):
        image_path = match.group("image_path")
        img_obj = image_map.get(image_path)
        
        if img_obj and img_obj.image_base64:
            base64_data = img_obj.image_base64.replace('\n', '').replace('\r', '')
            #base64_images_in_chunk.append(base64_data)
            try:  
                base64_thumb = make_thumbnail_base64(base64_data, thumb_size=(200, 200))  
                base64_images_in_chunk.append(base64_thumb)  
            except Exception as e:  
                logging.error(f"Error in creating thumbnail: {e}")  
                if base64_data.startswith('data:image'):
                    base64_data = base64_data.split(',', 1)[1]   
                base64_thumb = base64_data  
                base64_images_in_chunk.append(base64_thumb) 

            task_id = f"image_task_{page_idx}_{len(image_description_tasks)}"
            
            if text.strip() == "":
                # Se il testo è vuoto, descrivi con GPT-41
                image_description_tasks.append(
                    (task_id, describe_image_with_gpt41_async(base64_image=base64_thumb,
                                                              client= client_azure_openai,
                                                               sem= sem, 
                                                               global_timeout_time=global_timeout_time))
                )

            elif client_ImageAnalysis:
                # Se c'è testo, ottieni info con Azure Image Analysis
                try:
                    image_bytes = base64.b64decode(base64_thumb)
                    try:  
                        img = Image.open(io.BytesIO(image_bytes))  
                        img.verify()  # Eccezione se NON è una vera immagine decodificabile  
                    except Exception as e:  
                        raise ValueError("The base64 data is not a decodable image: " + str(e)) 
                    image_stream = io.BytesIO(image_bytes)
                    image_description_tasks.append(
                        (task_id, get_info_image_with_azure_async(client_ImageAnalysis=client_ImageAnalysis,
                                                                   image_stream=image_stream.getvalue(),
                                                                    sem_ocr= sem_ocr,
                                                                     global_timeout_time=global_timeout_time))
                    )
                except Exception as e:
                    logging.error(f"Error preparing image stream for Azure Image Analysis: {e}")
                    image_description_tasks.append((task_id, asyncio.sleep(0, result=f"[Errore nella preparazione dell'immagine: {e}]")))
            
            matches_to_process.append((match.start(), match.end(), task_id, base64_thumb))
        else:
            logging.warning(f"Image with path {image_path} not found or missing base64 data.")
            matches_to_process.append((match.start(), match.end(), None, None)) # Indichiamo di non processare questa

    time_left = global_timeout_time - time.time() 
    if time_left <= 5:  
        logging.warning("Global timeout less than 5 seconds in _process_single_pymupdf_chunk_async, stopping processing") 
        return [] 

    if image_description_tasks:
        try: 
            results = await asyncio.wait_for(
                asyncio.gather(*[task for _, task in image_description_tasks]),
                timeout=time_left
            )
        except asyncio.TimeoutError:  
            logging.warning("_process_single_mistral_page_async:Global timeout reached while waiting for asyncio.gather(*[task for _, task in image_description_tasks]) in .")  
            # Potresti decidere di restituire ciò che hai raccolto finora, oppure []  
            return [] 
        except asyncio.CancelledError:  
            logging.warning("_process_single_mistral_page_async: Task was cancelled!")  
            return [] 
        except Exception as e:  
            logging.error(f"_process_single_mistral_page_async: Error during asyncio.gather(*[task for _, task in image_description_tasks]) {e}")    
            return []   
 


        for (task_id, _), result in zip(image_description_tasks, results):
            image_descriptions[task_id] = result

    # Ora sostituisci il testo in base ai risultati ottenuti
    updated_text_parts = []
    last_end = 0
    image_count_on_page = 0
    
    for start, end, task_id, original_base64_data in matches_to_process:
        updated_text_parts.append(text[last_end:start])
        
        if task_id and task_id in image_descriptions:
            image_count_on_page += 1
            description = image_descriptions[task_id]
            image_id = f"image_{page_idx + 1}_{image_count_on_page}"
            
            if not description or description.strip() == "":
                updated_text_parts.append(f"![]({image_id})\n<!-- This is a general description and you need to understand the image from the text above and below -->\n")
            else:
                updated_text_parts.append(f"![]({image_id})\n<!-- {description.strip()} -->\n")

        else:

            updated_text_parts.append(text[start:end])
        
        last_end = end
    
    updated_text_parts.append(text[last_end:]) # Aggiungi il resto del testo
    updated_text = "".join(updated_text_parts)



    results = []
    for small_chunk in split_markdown_and_raw(updated_text):
        #print(f"{page_idx} + 1")
        results.append({
            "text_raw": small_chunk.get("raw", ""),
            "text_markdown": small_chunk.get("markdown",""),
            "metadata": {"page": page_idx + 1},
            "base64_imgs_list": base64_images_in_chunk,
            "headers": small_chunk.get("headers", {}),
        })

    return results







async def _process_single_pymupdf_chunk_async(
    chunk: Dict[str, Any],
    client_azure_openai: AsyncAzureOpenAI,
    client_ImageAnalysis: ImageAnalysisClient,
    pattern: re.Pattern,
    image_counter: Dict[Any, int],
    global_timeout_time: float,
    sem: asyncio.Semaphore,
    sem_ocr: asyncio.Semaphore
) -> List[Dict[str, Any]]:
    """
    Asynchronously processes a single text chunk (typically from pymupdf4llm output)
    to identify and describe embedded images.

    This function searches for Base64 encoded images within the chunk's text using
    a provided regex pattern. For each found image, it either generates a description
    using Azure OpenAI's GPT-4V (if the chunk text is empty) or extracts visual features
    using Azure Image Analysis (if there is text content). Image descriptions are then
    embedded back into the Markdown text as HTML comments. Finally, the updated Markdown
    text is split into smaller chunks (both Markdown and raw text) and returned.

    Args:
        chunk (Dict[str, Any]): A dictionary representing a text chunk, expected to contain
                                "text" and "metadata" (with "page").
        client_azure_openai (AsyncAzureOpenAI): An asynchronous Azure OpenAI client for GPT-4V image descriptions.
        client_ImageAnalysis (ImageAnalysisClient): An Azure Image Analysis client for extracting visual features.
        pattern (re.Pattern): A compiled regular expression pattern to find Base64 image data in the text.
        image_counter (Dict[Any, int]): A dictionary to maintain a unique count of images per page.
                                        This is modified in place.
        sem (asyncio.Semaphore): A semaphore to control concurrency for GPT-4V calls.
        sem_ocr (asyncio.Semaphore): A semaphore to control concurrency for Azure Image Analysis (OCR-related) calls.

    Returns:
        Dict[str, Any]: A list of dictionaries, where each dictionary represents a processed
                        sub-chunk from the original chunk. Each sub-chunk dictionary contains:
            - "text_raw" (str): The raw text content of the sub-chunk.
            - "text_markdown" (str): The Markdown content of the sub-chunk with image descriptions embedded.
            - "metadata" (Dict[str, Any]): Original metadata from the input chunk.
            - "base64_imgs_list" (List[str]): A list of Base64 encoded images found in the original chunk.
            - "headers" (Dict[str, str]): Headers associated with the sub-chunk.

    Logs:
        - ERROR: Errors encountered during image preparation for Azure Image Analysis.
    """
    print("pymupdf")

    text = chunk.get("text", "")
    page_number = chunk.get("metadata", {}).get("page", "unknown")
    image_counter.setdefault(page_number, 0)
    base64_images_in_chunk = []
    
    image_description_tasks = []
    image_descriptions = {}
    matches_to_process = []

    for match in pattern.finditer(text):
        base64_data = match.group(1).replace('\n', '').replace('\r', '')
        #base64_images_in_chunk.append(base64_data)
        try:  
            base64_thumb = make_thumbnail_base64(base64_data, thumb_size=(200, 200))  
            base64_images_in_chunk.append(base64_thumb)  
        except Exception as e:  
            logging.error(f"Error in creating thumbnail: {e}") 
            if base64_data.startswith('data:image'):
                base64_data = base64_data.split(',', 1)[1]   
            base64_thumb = base64_data  # fallback  
            base64_images_in_chunk.append(base64_thumb) 


        task_id = f"image_task_{page_number}_{len(image_description_tasks)}"
        
        if text.strip() == "":
            image_description_tasks.append(
                (task_id, describe_image_with_gpt41_async(base64_image=base64_thumb,
                                                           client= client_azure_openai,
                                                           sem= sem,
                                                           global_timeout_time=global_timeout_time))
            )
        else:
            try:
                image_bytes = base64.b64decode(base64_thumb)
                try:  
                    img = Image.open(io.BytesIO(image_bytes))  
                    img.verify()  # Eccezione se NON è una vera immagine decodificabile  
                except Exception as e:  
                    raise ValueError("The base64 data is not a decodable image: " + str(e)) 
                image_stream = io.BytesIO(image_bytes)
                image_description_tasks.append(
                    (task_id, get_info_image_with_azure_async(client_ImageAnalysis=client_ImageAnalysis,
                                                              image_stream= image_stream.getvalue(),
                                                               sem_ocr= sem_ocr,
                                                               global_timeout_time=global_timeout_time))
                )
            except Exception as e:
                logging.error(f"Error preparing image stream for Azure Image Analysis (pymupdf): {e}")
                image_description_tasks.append((task_id, asyncio.sleep(0, result=f"[Errore nella preparazione dell'immagine: {e}]")))

        matches_to_process.append((match.start(), match.end(), task_id, base64_thumb))

    time_left = global_timeout_time - time.time()  
    if time_left <= 5:  
        logging.warning("Global timeout less than 5 seconds in _process_single_pymupdf_chunk_async, stopping processing") 
        return [] 

    if image_description_tasks:
        try: 
            results = await asyncio.wait_for(
                asyncio.gather(*[task for _, task in image_description_tasks]),
                timeout=time_left
            )
        except asyncio.TimeoutError:  
            logging.warning("_process_single_pymupdf_chunk_async:Global timeout reached while waiting for asyncio.gather(*[task for _, task in image_description_tasks]).")  
            # Potresti decidere di restituire ciò che hai raccolto finora, oppure []  
            return [] 
        except Exception as e:  
            logging.error(f"_process_single_pymupdf_chunk_async:Error during asyncio.gather(*[task for _, task in image_description_tasks]) {e}")    
            return []
        except asyncio.CancelledError:  
            logging.warning("_process_single_pymupdf_chunk_async: Task was cancelled")  
            return []
 

        for (task_id, _), result in zip(image_description_tasks, results):
            image_descriptions[task_id] = result

    updated_text_parts = []
    last_end = 0
    image_count_on_page = 0

    for start, end, task_id, original_base64_data in matches_to_process:
        updated_text_parts.append(text[last_end:start])
        
        if task_id and task_id in image_descriptions:
            image_count_on_page += 1
            description = image_descriptions[task_id]
            image_id = f"image_{page_number}_{image_count_on_page}"
            
            if not description or description.strip() == "":
                updated_text_parts.append(f"![]({image_id})\n<!-- This is a general description and you need to understand the image from the text above and below -->\n")
            else:
                updated_text_parts.append(f"![]({image_id})\n<!-- {description.strip()} -->\n")
        else:
            updated_text_parts.append(text[start:end])
        
        last_end = end
    
    updated_text_parts.append(text[last_end:])
    updated_text = "".join(updated_text_parts)



    results = []
    for small_chunk in split_markdown_and_raw(updated_text):
        results.append({
            "text_raw": small_chunk.get("raw", ""),
            "text_markdown": small_chunk.get("markdown",""),
            "metadata": chunk.get("metadata", {}),
            "base64_imgs_list": base64_images_in_chunk,
            "headers": small_chunk.get("headers", {}),
        })
    return results



async def process_chunks_with_descriptions_async(
    ocr_data_source: Any,
    client_azure_openai: AsyncAzureOpenAI,
    client_ImageAnalysis: ImageAnalysisClient,
    sem_ocr: asyncio.Semaphore,
    sem: asyncio.Semaphore,
    global_timeout_time: float,
    source_type: str = "mistral",
) -> List[Dict[str, Any]]:
    """
    Asynchronously processes chunks of OCR data, adding AI-generated image descriptions
    based on the specified source type (Mistral or pymupdf4llm).

    This function orchestrates the parallel processing of either Mistral page objects
    or pymupdf4llm chunks. It dispatches individual chunks/pages to the appropriate
    helper function (`_process_single_mistral_page_async` or `_process_single_pymupdf_chunk_async`)
    to handle image description generation and text splitting.

    Args:
        ocr_data_source (Any): The source of OCR data.
            - If `source_type` is "mistral", this should be an object with a `.pages` attribute,
              where `pages` is an iterable of `OCRPageObject`.
            - If `source_type` is "pymupdf4llm", this should be an iterable of dictionaries,
              each representing a chunk from pymupdf4llm's output.
        client_azure_openai (AsyncAzureOpenAI): An asynchronous Azure OpenAI client for GPT-4V image descriptions.
        client_ImageAnalysis (ImageAnalysisClient): An Azure Image Analysis client for extracting visual features.
        sem_ocr (asyncio.Semaphore): A semaphore to control concurrency for Azure Image Analysis (OCR-related) calls.
        sem (asyncio.Semaphore): A semaphore to control concurrency for GPT-4V calls.
        source_type (str, optional): The type of OCR data source. Can be "mistral" or "pymupdf4llm".
                                     Defaults to "mistral".

    Returns:
        List[Dict[str, Any]]: A flattened list of dictionaries, where each dictionary represents a processed
                              chunk with embedded image descriptions and split text.

    Logs:
        - INFO: When starting the chunk processing.
        - DEBUG: Details about the OCR response type and content (for Mistral).
        - INFO: When Mistral OCR processing is successful.
        - WARNING: If no valid OCR pages are found and a fallback to pymupdf4llm is initiated.
        - INFO: When pymupdf4llm processing is successful.
        - ERROR: If `processed_chunks` from Mistral or pymupdf4llm is empty or None.
    """

    processed_chunks = []
    logging.info(f"process_chunks_with_descriptions_async: Starting async chunk processing with {source_type} source type")
    if source_type == "mistral":
        time_left = global_timeout_time - time.time()  
        if time_left <= 5:
            logging.warning(f"process_chunks_with_descriptions_async: Global timeout reached, stopping processing")
            return []
        # Creiamo una lista di coroutine, una per ogni pagina Mistral
        page_processing_tasks = []
        for page_idx, page in enumerate(ocr_data_source.pages):
            page_processing_tasks.append(
                _process_single_mistral_page_async(
                    page, page_idx, client_azure_openai, client_ImageAnalysis,sem, sem_ocr, global_timeout_time
                )
            )
        # Eseguiamo tutte le coroutine in parallelo
        try:
            processed_chunks = await asyncio.wait_for( asyncio.gather(*page_processing_tasks), timeout= time_left)
        except asyncio.TimeoutError:  
            logging.warning("process_chunks_with_descriptions_async: Global timeout reached while waiting for wait_for( asyncio.gather(*page_processing_tasks)")    
            return [] 
        except Exception as e:  
            logging.error(f"process_chunks_with_descriptions_async: Error during wait_for( asyncio.gather(*page_processing_tasks): {e}")    
            return [] 
        except asyncio.CancelledError:  
            logging.warning("process_chunks_with_descriptions_async: Task was cancelled")  
            return []

        flat_chunks = []
        for chunk_list in processed_chunks:
            if isinstance(chunk_list, list):
                flat_chunks.extend(chunk_list)
            else:
                flat_chunks.append(chunk_list)

    

    elif source_type == "pymupdf4llm":
        time_left = global_timeout_time - time.time()  
        if time_left <= 5:
            logging.warning(f"process_chunks_with_descriptions_async: Global timeout reached, stopping processing")
            return []
        
        image_counter = {} # Questo contatore deve essere per l'intera esecuzione per mappare le immagini per pagina
        pattern = re.compile(r"!\[\]\(data:image/(?:png|jpeg|gif);base64,([A-Za-z0-9+/=\s]+?)\)", re.IGNORECASE)

        # Usiamo asyncio.gather per processare i chunk in parallelo
        chunk_processing_tasks = []
        for chunk in ocr_data_source:
            chunk_processing_tasks.append(
                _process_single_pymupdf_chunk_async(
                    chunk, client_azure_openai, client_ImageAnalysis,
                      pattern, 
                      image_counter,global_timeout_time, sem, sem_ocr
                )
            )
        
        try:    
            # Eseguiamo tutte le coroutine in parallelo
            processed_chunks = await asyncio.wait_for( asyncio.gather(*chunk_processing_tasks), timeout= time_left)
        except asyncio.TimeoutError:  
            logging.warning("process_chunks_with_descriptions_async: Global timeout reached while waiting for asyncio.gather(*chunk_processing_tasks).")  
            # Potresti decidere di restituire ciò che hai raccolto finora, oppure []  
            return [] 
        except Exception as e:  
            logging.error(f"process_chunks_with_descriptions_async: Error during wait_for( asyncio.gather(*chunk_processing_tasks): {e}")    
            return []
        except asyncio.CancelledError:  
            logging.warning("process_chunks_with_descriptions_async: Task was cancelled")  
            return []

            # Flatten the list of lists
        flat_chunks = []
        for chunk_list in processed_chunks:
            if isinstance(chunk_list, list):
                flat_chunks.extend(chunk_list)
            else:
                flat_chunks.append(chunk_list)

    return flat_chunks
    

async def add_keywords_pdf_async(pdf_file_obj: io.BytesIO,
                                client_azure_openai: AsyncAzureOpenAI,                                    
                                text_analytics_client:TextAnalyticsClient,
                                client_ImageAnalysis: ImageAnalysisClient,
                                client_mistral: Mistral,
                                global_timeout_time: float,
                                sem: asyncio.Semaphore,
                                sem_ocr: asyncio.Semaphore,
                                ) -> List[Dict[str, Any]]:
    """
    Asynchronously extracts text, images, and keywords from a PDF,
    using Mistral OCR (with fallback to pymupdf4llm) and adds AI-generated image descriptions.

    It also detects the language for the entire PDF and extracts key phrases for all processed chunks
    in parallel.

    Args:
        pdf_file_obj (io.BytesIO): In-memory PDF for input.
        client_azure_openai (AsyncAzureOpenAI): Async OpenAI client for GPT-4V image processing.
        text_analytics_client (TextAnalyticsClient): Azure Text Analytics client for language detection and keyword extraction.
        client_ImageAnalysis (ImageAnalysisClient): Azure Image Analysis client for extracting visual features from images.
        client_mistral (Mistral): Mistral client for OCR processing.
        sem (asyncio.Semaphore): Semaphore to control concurrency for GPT-4V image description calls.
        sem_ocr (asyncio.Semaphore): Semaphore to control concurrency for Azure Image Analysis OCR calls.

    Returns:
        List[dict]: A list of enriched chunk dictionaries. Each dictionary contains:
            - "text_raw" (str): The raw text content of the chunk.
            - "text_markdown" (str): The Markdown content of the chunk (with image descriptions embedded).
            - "metadata" (Dict[str, Any]): Metadata for the chunk (e.g., page number).
            - "base64_imgs_list" (List[str]): A list of Base64 encoded images found in the chunk.
            - "headers" (Dict[str, str]): Headers associated with the chunk.
            - "keywords" (List[str]): A list of unique keywords extracted from the chunk.
            - "detected_language" (str): The detected language for the entire document.

    Logs:
        - INFO: Steps of the OCR and processing pipeline, including language detection.
        - DEBUG: Details about the OCR response.
        - WARNING: If Mistral OCR fails or no valid pages are found, triggering the pymupdf4llm fallback.
        - WARNING: If no text is found for language detection.
        - ERROR: If both OCR methods fail, or errors occur during language detection or keyword extraction.
    """ 
    final_documents = []
    detected_language = "en" # Default language
    processed_chunks = None

    # Tentativo con Mistral OCR se disponibile
    if client_mistral:
        logging.info("Attempting OCR with Mistral async.")
        try:
            ocr_response = await get_ocr_markdown_chunks_pdf_mistral_async(
                pdf_file_obj=pdf_file_obj,
                global_timeout_time= global_timeout_time, 
                sem_ocr=sem_ocr, 
                client_mistral=client_mistral
            )
            
            if ocr_response and hasattr(ocr_response, "pages") and ocr_response.pages:
                logging.info("Mistral OCR successful. Processing chunks.")
                processed_chunks = await process_chunks_with_descriptions_async(
                    ocr_data_source=ocr_response,
                    client_azure_openai=client_azure_openai,
                    client_ImageAnalysis=client_ImageAnalysis,
                    sem=sem,
                    sem_ocr=sem_ocr,
                    global_timeout_time=global_timeout_time,
                    source_type="mistral"

                )
            else:
                logging.warning("Mistral OCR returned empty response.")
                
        except Exception as e:
            logging.error(f"Mistral OCR failed: {e}")
    else:
        logging.warning("Mistral client not available.")

    # Fallback con pymupdf4llm se Mistral non ha funzionato
    if not processed_chunks:
        logging.warning("Falling back to pymupdf4llm.")
        try:
            pdf_bytes = pdf_file_obj.getvalue()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            md_text_pymupdf = pymupdf4llm.to_markdown(
                doc=doc,
                page_chunks=True,
                write_images=False,
                embed_images=True,
                show_progress=False 
            )
            logging.info("pymupdf4llm processing successful.")
            processed_chunks = await process_chunks_with_descriptions_async(
                ocr_data_source=md_text_pymupdf,
                client_azure_openai=client_azure_openai,
                source_type="pymupdf4llm",
                sem=sem,
                sem_ocr=sem_ocr,
                client_ImageAnalysis=client_ImageAnalysis,
                global_timeout_time=global_timeout_time
            )
        except Exception as e:
            logging.error(f"Both Mistral OCR and pymupdf4llm failed: {e}")
            return []

    if not processed_chunks:
        logging.error("No chunks processed at all, returning empty list.")
        return []

    if time.time() >= global_timeout_time:
            logging.warning("Global timeout reached, stopping processing")
            return []

    all_text_for_language_detection = " ".join([chunk.get("text_raw", "") for chunk in processed_chunks if chunk.get("text_raw")])
    if all_text_for_language_detection.strip():
        try:
            max_length = 4900
            sample_text = all_text_for_language_detection[:max_length]
            detected_language = await detect_language_with_fallback_async(  
                text= sample_text, text_analytics_client= text_analytics_client, global_timeout_time=global_timeout_time  
            )  
            logging.info(f"Detected language: {detected_language}")  
        except Exception as e:  
            logging.error(f"Error during async language detection: {e}") 
    else:  
          logging.warning("No text found for language detection. Using default language 'en'.") 

    time_left = global_timeout_time - time.time()  
    if time_left <= 5:  
        logging.warning("Global timeout less than 5 seconds in add_keywords_pdf_async.") 
        return [] 

    try:  
        keyword_tasks = [  
            process_single_chunk_for_keywords_async(chunk= chunk,
                                                     detected_language= detected_language,
                                                       text_analytics_client= text_analytics_client,
                                                         global_timeout_time= global_timeout_time)  
            for chunk in processed_chunks  
        ]  



        final_documents = await asyncio.wait_for( 
             asyncio.gather(*keyword_tasks),
             timeout=time_left
        )  
    except asyncio.TimeoutError:  
        logging.warning("add_keywords_pdf_async: Global timeout reached while waiting for add_keywords_pdf_async.")  
        # Potresti decidere di restituire ciò che hai raccolto finora, oppure []  
        return [] 
    except Exception as e:  
        logging.error(f"add_keywords_pdf_async: Error during async keyword extraction: {e}")  
        return []  
    except asyncio.CancelledError:  
        logging.warning("add_keywords_pdf_async: Task was cancelled")  
        return []



    return final_documents  

async def get_pdf_filelike_from_url_async(blob_url: str, sem_download:asyncio.Semaphore) -> Optional[io.BytesIO]:
    """
    Asynchronously downloads a PDF from a given URL into a BytesIO object.

    Args:
        blob_url (str): The URL of the PDF blob to download.

    Returns:
        Optional[io.BytesIO]: A BytesIO object containing the PDF's binary data
            if the download is successful, otherwise None.
    """
    try:
        async with sem_download: 
            async with aiohttp.ClientSession() as session:
                async with session.get(blob_url) as response:
                    response.raise_for_status()  # Solleva un'eccezione per codici di stato HTTP 4xx/5xx
                    pdf_bytes = await response.read()
                    return io.BytesIO(pdf_bytes)
    except aiohttp.ClientError as e:
        logging.error(f"Error downloading PDF from {blob_url}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while downloading PDF from {blob_url}: {e}")
        return None


def batch_by_payload_size(dicts, max_bytes=14*1024*1024):  # 14MB margine  
    """
    A generator that batches a list of dictionaries into smaller lists,
    ensuring that the JSON-encoded size of each batch does not exceed a specified maximum.

    This is useful for preparing payloads for APIs that have size limitations on requests.

    Args:
        dicts (list): A list of dictionaries to be batched.
        max_bytes (int, optional): The maximum allowed size (in bytes) for the JSON-encoded
                                   payload of a single batch. Defaults to 14MB.

    Yields:
        list: A list of dictionaries forming a batch, whose total JSON-encoded size
              is less than or equal to `max_bytes`.
    """
    batch = []  
    size = 0  
    for d in dicts:  
        s = len(json.dumps(d, ensure_ascii=False).encode('utf-8'))  
        if batch and size+s > max_bytes:  
            yield batch  
            batch = [d]  
            size = s  
        else:  
            batch.append(d)  
            size += s  
    if batch:  
        yield batch  

async def upload_batches(session, url, api_key, dicts_for_indexing):  
    """
    Asynchronously uploads batches of documents to a specified URL using HTTP POST requests.

    This function takes a list of dictionaries, batches them by payload size using `batch_by_payload_size`,
    and then sends each batch as a JSON payload to the target URL. It includes an API key in the headers
    for authentication. It logs the progress of the upload and raises an exception if any batch upload fails.

    Args:
        session (aiohttp.ClientSession): An aiohttp client session to make HTTP requests.
        url (str): The URL endpoint to which the batches will be uploaded.
        api_key (str): The API key to be included in the request headers for authentication.
        dicts_for_indexing (list): A list of dictionaries, where each dictionary represents a document
                                   to be indexed.

    Returns:
        list: A list of JSON responses received from the server for each successful batch upload.

    Raises:
        Exception: If a batch upload fails (i.e., the server responds with a status code other than 200).

    Logs:
        - INFO: For each batch uploaded, indicating the number of documents in the batch and the total uploaded count.
        - INFO: Once the entire upload process is completed.
    """
    headers = {'Content-Type': 'application/json', 'api-key': api_key}  
    uploaded = 0  
    responses = []  
    for batch in batch_by_payload_size(dicts_for_indexing):  
        payload = {"value": batch}  
        async with session.post(url, json=payload, headers=headers) as response:  
            if response.status != 200:  
                errtext = await response.text()  
                raise Exception(f"Batch upload failed: {response.status} {errtext}")  
            res = await response.json() 
            responses.append(res)  
            uploaded += len(batch)  
            logging.info(f"Uploaded a batch of {len(batch)} documents. Total uploaded: {uploaded}")  
    logging.info("Upload completed.") 
    return responses  




async def get_pdf_with_retry(blob_url: str, sem_download: asyncio.Semaphore, global_timeout_time: float, max_retries: int = 3, delay: float = 3, timeout: int=120) -> Optional[io.BytesIO]:  
    """
    Attempts to download a PDF file from a given blob URL with a retry mechanism.

    This function calls `get_pdf_filelike_from_url_async` (assumed to be defined elsewhere)
    to download the PDF. If the download fails, it retries up to `max_retries` times,
    pausing for `delay` seconds between attempts. It uses a semaphore to control
    the concurrency of download operations.

    Args:
        blob_url (str): The URL of the PDF blob to download.
        sem_download (asyncio.Semaphore): An asyncio semaphore to limit concurrent PDF downloads.
        max_retries (int, optional): The maximum number of times to retry the download. Defaults to 3.
        delay (float, optional): The time in seconds to wait between retries. Defaults to 3.

    Returns:
        Optional[io.BytesIO]: A file-like object (BytesIO) containing the PDF content if the download is successful,
                              otherwise None after all retries are exhausted.

    Logs:
        - WARNING: For each failed attempt, indicating the attempt number, retry delay, and URL.
        - ERROR: If all download attempts fail.
    """
# fallback 

    for attempt in range(1, max_retries+1):  
        if global_timeout_time:  
            time_left = global_timeout_time - time.time()  
        if time_left <= 0:  
            logging.error("Global timeout expired.")  
            return None
        else:  
            time_left = timeout  

        try:
            pdf_io = await asyncio.wait_for( get_pdf_filelike_from_url_async(blob_url=blob_url, sem_download=sem_download),  
            timeout=time_left  
            )  
            if pdf_io:  
                return pdf_io  
        except asyncio.TimeoutError:  
            logging.error(f"get_pdf_with_retry: Global timeout reached in get_pdf_with_retry")
            pdf_io = None  
        except Exception as e:
            logging.error(f"get_pdf_with_retry: Error downloading PDF (attempt {attempt}): {e}")
        except asyncio.CancelledError:  
            logging.warning("get_pdf_with_retry: Task was cancelled")  
            return None


        logging.warning(f"get_pdf_with_retry: Attempt {attempt} to download PDF failed. Retrying in {delay}s... (url: {blob_url})")  
        await asyncio.sleep(delay)  
    logging.error(f"get_pdf_with_retry: All {max_retries} attempts to download PDF from {blob_url} failed.")  
    return None  




async def upload_index_pdf_from_url_async(
    blob_url: str,
    chat_id: str,
    parent_id: str,
    url: str, 
    embedding_function: Callable[[str, AsyncAzureOpenAI, str, asyncio.Semaphore, int, int, int], Awaitable[list[float]]]   , # Potrebbe essere Matryoshka_embedding_async
    api_key: str, 
    client_embedding: AsyncAzureOpenAI,
    model_deployment: str,
    client: AsyncAzureOpenAI, 
    pdf_name: str,
    record_id: str,
    sem:asyncio.Semaphore,
    sem_download: asyncio.Semaphore,
    sem_ocr:asyncio.Semaphore,
    sem_embedding:asyncio.Semaphore,
    client_mistral: Mistral,
    text_analytics_client:TextAnalyticsClient,
    client_ImageAnalysis: ImageAnalysisClient,
    global_timeout_time: float,
) -> bool:
    """
    Asynchronously downloads a PDF from a URL, processes its content (OCR, image description, keyword extraction),
    generates embeddings for text chunks, and uploads the enriched documents to an Azure Search index.

    This function orchestrates a multi-step process:
    1. Downloads the PDF with a retry mechanism.
    2. Processes the PDF using Mistral OCR (with a fallback to pymupdf4llm) to extract text, images, and
       generate image descriptions.
    3. Detects the dominant language of the entire PDF.
    4. Extracts keywords for each processed text chunk using Azure Text Analytics with a fallback.
    5. Generates embeddings for each text chunk using a specified embedding function.
    6. Prepares the documents for indexing with relevant metadata and content.
    7. Uploads the prepared documents in batches to the Azure Search index.

    Args:
        blob_url (str): The URL of the PDF blob in Azure Storage.
        chat_id (str): A unique identifier for the chat session.
        parent_id (str): An identifier for the parent document or source.
        url (str): The URL of the Azure Search index endpoint.
        embedding_function (Callable): An asynchronous callable function responsible for generating text embeddings.
                                      Expected signature: `async def func(text: str, client: AsyncAzureOpenAI,
                                      model_deployment: str, sem_embedding: asyncio.Semaphore, ...) -> List[float]`.
                                      Example: `Matryoshka_embedding_async`.
        api_key (str): The API key for authenticating with the Azure Search index.
        client_embedding (AsyncAzureOpenAI): An asynchronous Azure OpenAI client specifically for embedding generation.
        model_deployment (str): The name of the Azure OpenAI model deployment to use for embeddings.
        client (AsyncAzureOpenAI): A general-purpose asynchronous Azure OpenAI client (e.g., for GPT-4V).
        pdf_name (str): The name of the PDF file.
        record_id (str): A unique identifier for the record (document).
        sem (asyncio.Semaphore): A semaphore to control concurrency for general OpenAI calls (e.g., GPT-4V).
        sem_download (asyncio.Semaphore): A semaphore to control concurrency for PDF downloads.
        sem_ocr (asyncio.Semaphore): A semaphore to control concurrency for OCR-related operations (e.g., Azure Image Analysis).
        sem_embedding (asyncio.Semaphore): A semaphore to control concurrency for embedding generation calls.
        client_mistral (Mistral): The client for Mistral OCR processing.
        text_analytics_client (TextAnalyticsClient): The Azure Text Analytics client for language detection and keyword extraction.
        client_ImageAnalysis (ImageAnalysisClient): The Azure Image Analysis client for extracting visual features from images.

    Returns:
        None: The function performs indexing operations and does not return a value.

    Raises:
        Exception: Catches and logs various exceptions that can occur during the process,
                   such as download failures, processing errors, or indexing errors.

    Logs:
        - INFO: Progress of various stages (download, OCR attempt, embedding, upload).
        - WARNING: If PDF download fails, no processed documents are found, or text for embedding is empty.
        - ERROR: For critical failures at any stage (download, OCR, embedding calculation, indexing).
    """


  
    pdf_io = await get_pdf_with_retry(blob_url=blob_url, sem_download=sem_download,global_timeout_time=global_timeout_time, max_retries=3, delay=3)  
  
    if not pdf_io:  
        logging.error(f"upload_index_pdf_from_url_async: Failed to download PDF from {blob_url}. Aborting indexing.")  
        return False

    if global_timeout_time:  
        time_left = global_timeout_time - time.time()  
        if time_left <= 5:  
            logging.error("Global timeout expired.")  
            raise TimeoutError("Global timeout expired.")  
    else:  
        return False  # Se non c'è un timeout globale, non procediamo



    try:
        processed_docs_with_keywords = await asyncio.wait_for(
             add_keywords_pdf_async(pdf_file_obj=pdf_io,client_azure_openai= client, 
                                                                text_analytics_client=text_analytics_client,
                                                                client_ImageAnalysis=client_ImageAnalysis,
                                                                client_mistral=client_mistral,
                                                                global_timeout_time=global_timeout_time,
                                                                sem=sem,
                                                                sem_ocr=sem_ocr
                                                                ),
                                                         timeout=time_left
                                                        )
    
    except asyncio.TimeoutError:  
        logging.warning("upload_index_pdf_from_url_async: Timeout nella fase di add_keywords_pdf_async. Skipping indexing.")  
        return False  
    except asyncio.CancelledError:  
        logging.warning("upload_index_pdf_from_url_async: Task cancellato nella fase di add_keywords_pdf_async.")  
        return False  
    except Exception as e:  
        logging.error(f"upload_index_pdf_from_url_async: Errore durante add_keywords_pdf_async: {e}")  
        return False  

    
    if not processed_docs_with_keywords:
        logging.warning(f"upload_index_pdf_from_url_async: No processed documents found for PDF {pdf_name}. Skipping indexing.")
        return False

    dicts_for_indexing = []
    embedding_tasks = []
    docs_for_embedding= []


    if time.time() >= global_timeout_time:
        logging.warning("upload_index_pdf_from_url_async: Global timeout reached, stopping processing")
        return False

    # Prepara le task per gli embedding in parallelo
    for doc in processed_docs_with_keywords:
        page_number = str(doc.get("metadata", {}).get("page", "-1"))
        context_for_embedding = doc.get("text_raw", "")




        if not context_for_embedding or context_for_embedding.strip() == "":
            logging.warning(
                f"upload_index_pdf_from_url_async: Skipping embedding for empty or whitespace 'text_raw' for document on page {page_number}. "
                "This document will NOT be indexed with a content_vector."
            )

            doc["_skip_embedding_and_indexing"] = True 
            docs_for_embedding.append(doc) # Aggiungi comunque il doc per mantenere l'allineamento degli indici
            embedding_tasks.append(asyncio.sleep(0)) # Aggiungi una task "dummy" che non fa nulla per mantenere l'ordine
            continue
 

        # Aggiungi la task di embedding
        docs_for_embedding.append(doc)
        embedding_tasks.append(
            embedding_function(
                text=context_for_embedding,
                global_timeout_time= global_timeout_time,
                client=client_embedding,
                model_deployment=model_deployment,
                sem_embedding=sem_embedding
            )
        )

    time_left = global_timeout_time - time.time()
    if time_left <= 5:
        logging.warning("upload_index_pdf_from_url_async:Global timeout less than 5 seconds.")
        return False

    try:  
        embeddings = await asyncio.wait_for( 
            asyncio.gather(*embedding_tasks),
            timeout= time_left
        ) 
        if  not embeddings or ["Blocking error"] in embeddings:
            logging.error(f"upload_index_pdf_from_url_async: Blocking error detected for document on page {page_number}. Skipping all pdf from indexing.")
            return False
        


    except asyncio.TimeoutError:  
        logging.warning("upload_index_pdf_from_url_async:Global timeout reached while waiting for embeddings.")  
        # Potresti decidere di restituire ciò che hai raccolto finora, oppure []  
        return False
    except Exception as e:  
        logging.error(f"upload_index_pdf_from_url_async:Error during batch embedding: {e}")    
        return False
    except asyncio.CancelledError:  
        logging.warning("upload_index_pdf_from_url_async: Task was cancelled")  
        return False


    if time.time() >= global_timeout_time:
        logging.warning("upload_index_pdf_from_url_async: Global timeout reached, stopping processing")
        return False


    for i, doc in enumerate(docs_for_embedding):
        if doc.get("_skip_embedding_and_indexing", False):
            logging.info(f"upload_index_pdf_from_url_async:Skipping indexing for document on page {doc.get('metadata', {}).get('page', '-1')} due to empty text_raw.")
            continue 
        page_number = str(doc.get("metadata", {}).get("page", "-1"))
        embedding = embeddings[i]

        if isinstance(embedding, Exception):
            logging.error(f"⚠️ upload_index_pdf_from_url_async: Error calculating embedding for page {page_number}: {embedding}. Skipping this chunk from indexing.")
            continue


        idx_id = generate_safe_id() 
        doc_payload = {
            "@search.action": "upload",
            "id": idx_id,
            "content": doc.get("text_raw", ""), 
            "filepath": blob_url, 
            "title": pdf_name,
            "chat_id": chat_id,
            "parent_id": parent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_vector": embedding,
            "page_number": page_number,
            "keywords": doc.get("keywords", []),
            "base64_data": "", 
            "content_markdown": doc.get("text_markdown", ""), 
            "base64_imgs_list": doc.get("base64_imgs_list", []),
            "pdf_language": doc.get("detected_language", "en"), 
            "pdf_name": pdf_name
        }
        dicts_for_indexing.append(doc_payload)


    # 3. Caricamento su Azure Search
    if not dicts_for_indexing:
        logging.warning(f"upload_index_pdf_from_url_async:No documents prepared for indexing for PDF {pdf_name}. Aborting indexing.")
        return False 




    try:
        async with aiohttp.ClientSession() as session:

            all_responses = await upload_batches(session, url, api_key, dicts_for_indexing)  
            logging.info(f"upload_index_pdf_from_url_async:Document processed and indexed in {len(all_responses)} batches!" ) 
            return True
    except aiohttp.ClientError as e:
        logging.error(f"upload_index_pdf_from_url_async:Errore durante l'indicizzazione: {e}")
        return False
    except Exception as e:
        logging.error(f"upload_index_pdf_from_url_async:Errore inatteso durante l'indicizzazione: {e}")
        return False








#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################






#def generate_parent_id():  
#    return str(uuid.uuid4())  

  
def generate_safe_id():  
    return str(uuid.uuid4()) 




def get_base64data(pymupdf_page):
    """
    Obtains the image of a PyMuPDF page in base64 format.

    This function renders the PyMuPDF page into a pixmap, converts it to a Pillow Image object,
    saves it as a PNG in memory, and then encodes it to a base64 string.

    Args:
        pymupdf_page: A PyMuPDF page object from which to extract the image.

    Returns:
        str: A base64 encoded string representation of the page's image in PNG format.
    """
    """Ottiene l'immagine di una pagina in formato base64."""
    pix = pymupdf_page.get_pixmap()
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    byte_io = io.BytesIO()
    image.save(byte_io, format='PNG')
    return base64.b64encode(byte_io.getvalue()).decode('utf-8')


#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################


async def download_blob(blob_url, RecordId,  log_suffix, sem_download: asyncio.Semaphore ):  
    """
    Asynchronously downloads the content of a blob from a given URL.

    This function uses an asyncio semaphore to control concurrency of downloads.
    It performs an asynchronous HTTP GET request and returns the content as bytes.
    Errors during download are logged, and None is returned.

    Args:
        blob_url (str): The URL of the blob to download.
        RecordId (str): The ID of the record associated with this download, used for logging.
        log_suffix (str): A suffix to append to log messages for identification.
        semaphore (asyncio.Semaphore, optional): An asyncio semaphore to limit concurrent
                                                 download operations. Defaults to `sem_download`.

    Returns:
        Optional[bytes]: The content of the blob as bytes if successful, otherwise None.
    """
    try:
        async with sem_download:   
            async with aiohttp.ClientSession() as session:  
                async with session.get(blob_url) as response:  
                    response.raise_for_status()  
                    content = await response.read()  
                    return content  # Oppure BytesIO(content), se serve come file-like  
    except Exception as e:  
        logging.warning(f"{log_suffix} Skipping due to blob_url fetch error for RecordId={RecordId}: {e}")  
        return None  # L'errore viene gestito dal chiamante! 
 
async def process_single_value(value:Dict[str, Any],
                        log_suffix: str,
                        client_openai:AsyncAzureOpenAI,
                        sem:asyncio.Semaphore,
                        sem_download: asyncio.Semaphore,
                        sem_ocr:asyncio.Semaphore,
                        sem_embedding:asyncio.Semaphore,
                        client_mistral: Mistral,
                        text_analytics_client:TextAnalyticsClient,
                        client_ImageAnalysis: ImageAnalysisClient,
                        global_timeout_time: float, 
                        values: List[Dict[str, Any]]):


    # Controlla se c'è ancora tempo prima del timeout globale
    if time.time() >= global_timeout_time:
        logging.error(f"{log_suffix} Global timeout reached before processing")
        return {}
    
    RecordId = value.get('RecordId')  
    blob_url = value.get('blob_url')  
    pdf_name = value.get('pdf_name')  
    is_processed = value.get('is_processed')  
    chat_id = value.get('chat_id')  
  
    if not is_processed:
        blob_content = await download_blob(blob_url= blob_url,
                                        RecordId= RecordId,
                                        log_suffix= log_suffix,
                                        sem_download=sem_download)  
        if blob_content is None:  
            return {  
                "RecordId": RecordId,  
                "status": "failed",   
                "blob_url": blob_url, 
                "pdf_name": pdf_name,
                "is_processed": is_processed,
                "chat_id": chat_id
            }  
    

        #Il recordId è generato in maniera deterministica dal backend e quindi identifica in maniera univoca il documento, nel futuro si pensa di controllare 
        #se il documento è già stato processato in precedenza, per evitare di ripetere l'operazione
        parent_id = RecordId  
        try:  
            url = (  
                f"https://{AZURE_NAME_SEARCH_SERVICE}.search.windows.net/indexes/{AZURE_SEARCH_INDEX}/docs/search.index?api-version=2024-07-01"  
            )  
            partial_results= await upload_index_file_from_url_async(  
                blob_url=blob_url,  
                chat_id=chat_id,  
                parent_id=parent_id,  
                url=url,  
                embedding_function=Matryoshka_embedding_async,  
                client_embedding=client_openai,  
                model_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,  
                client=client_openai,  
                api_key=AZURE_SEARCH_KEY,  
                pdf_name=pdf_name,
                record_id=RecordId,
                sem=sem,
                sem_download=sem_download,
                sem_ocr=sem_ocr,
                sem_embedding=sem_embedding,
                client_mistral= client_mistral,
                text_analytics_client= text_analytics_client,
                client_ImageAnalysis= client_ImageAnalysis,   
                global_timeout_time= global_timeout_time,
            )  
            logging.info(f'{log_suffix} Successfully processed PDF.') 


            if not partial_results:  
                logging.warning(f"{log_suffix} No results returned from upload_index_file_from_url_async for RecordId={RecordId}.")  
                return {  
                    "RecordId": RecordId,  
                    "status": "failed",  
                    "error": "No results returned from upload_index_file_from_url_async",  
                    "blob_url": blob_url, 
                    "pdf_name": pdf_name,
                    "is_processed": is_processed,
                    "chat_id": chat_id  
                }



            await wait_for_record_ids_async(results=values,
                                            record_ids=[RecordId],
                                            log_suffix= log_suffix,
                                            timeout_seconds=215,
                                            poll_interval=5,
                                            global_timeout_time= global_timeout_time 
                                            )

            return {  
                "RecordId": RecordId,  
                "status": "ok",  
                "blob_url": blob_url, 
                "pdf_name": pdf_name,
                "is_processed": is_processed,
                "chat_id": chat_id  
            }
        except Exception as e:  
            logging.info(f"{log_suffix} Error during indexing: {str(e)}")  
            return {  
                "RecordId": RecordId,  
                "status": "failed",  
                "error": str(e),  
                "blob_url": blob_url, 
                "pdf_name": pdf_name,
                "is_processed": is_processed,
                "chat_id": chat_id  
            }   
    else:
        logging.info(f"{log_suffix} PDF already processed, skipping: {pdf_name}")  
        return {  
            "RecordId": RecordId,  
            "status": "ok",  
            "blob_url": blob_url, 
            "pdf_name": pdf_name,
            "is_processed": is_processed,
            "chat_id": chat_id  
        } 


async def process_values(values: List[Dict[str, Any]],
                        log_suffix: str,
                        client_openai:AsyncAzureOpenAI,
                        sem:asyncio.Semaphore,
                        sem_download: asyncio.Semaphore,
                        sem_ocr:asyncio.Semaphore,
                        sem_embedding:asyncio.Semaphore,
                        client_mistral: Mistral,
                        text_analytics_client:TextAnalyticsClient,
                        client_ImageAnalysis: ImageAnalysisClient,
                        global_timeout_time: float):  
                # Controlla se c'è ancora tempo prima del timeout globale

    time_left = global_timeout_time - time.time()  
    if time_left <= 5:  
        logging.warning("process_values:Global timeout less than 5 seconds.") 
        return {"results": []} 

#    tasks = [  
#        process_single_value(value, log_suffix, client_openai,
#                             sem,
#                        sem_download,
#                        sem_ocr,
#                        sem_embedding,
#                        client_mistral,
#                        text_analytics_client,
#                        client_ImageAnalysis,
#                        global_timeout_time, values                                                   
#                             )  
#
#        for value in values  
#    ]
    tasks = [  
        asyncio.create_task(
            process_single_value(value, log_suffix, client_openai,
                                 sem,
                                sem_download,
                                sem_ocr,
                                sem_embedding,
                                client_mistral,
                                text_analytics_client,
                                client_ImageAnalysis,
                                global_timeout_time, values)  
        )
        for value in values  
    ]    
    try:  
        results = await asyncio.wait_for(  asyncio.gather(*tasks), 
                                         timeout= time_left + 10 ##### The 10-second delta was added to allow child coroutines to close.
                                        )   
    except asyncio.TimeoutError:  
        logging.warning("process_values: Global timeout reached while waiting for asyncio.wait_for(  asyncio.gather(*tasks).")  
        for task in tasks:
            if not task.done():
                task.cancel()  # Cancella task ancora in corso
            if task.done() and task.exception():  # ✅ Controlla se è completata E ha eccezioni
                try:
                    task.exception()  # "Consuma" l'eccezione
                except:
                    pass
        # Potresti decidere di restituire ciò che hai raccolto finora, oppure []  
        return {"results": []} 
    except Exception as e:  
        logging.error(f"process_values:Error during asyncio.wait_for(  asyncio.gather(*tasks) {e}")    
        return {"results": []} 
    except asyncio.CancelledError:  
        logging.warning("process_values: Task was cancelled") 
        for task in tasks:
            if not task.done():
                task.cancel()  # Cancella task ancora in corso
            if task.done() and task.exception():  # ✅ Controlla se è completata E ha eccezioni
                try:
                    task.exception()  # "Consuma" l'eccezione
                except:
                    pass
        return {"results": []}



    return {  
        "results": results
    }  







###################################################################################################################################################################
###################################################################################################################################################################
######## FINE ELABORAZIONE EXCEL #######################################################################################################################################
###################################################################################################################################################################

def trim_empty_edges(df):  
    # Elimina le righe e colonne tutte vuote (solo stringhe vuote)  
    df = df.loc[~(df == '').all(axis=1)]   # Righe  
    df = df.loc[:, ~(df == '').all(axis=0)] # Colonne  
    return df  





  
async def process_excel(file_obj: io.BytesIO,
                         record_id: str,text_analytics_client:TextAnalyticsClient, max_keywords: int=20):  
    """
    Extracts markdown from each sheet of an Excel file, detects the overall language,
    extracts keywords from the combined text, and returns a list of dictionaries
    structured similarly to processed PDF documents for indexing.

    This function performs the following steps:
    1. Iterates through each sheet in the provided Excel file object.
    2. Cleans and converts each sheet's data into a GitHub-flavored Markdown table.
    3. Aggregates all sheet Markdown content to detect the overall language of the Excel file.
    4. Extracts keywords from the combined Markdown text using Azure Text Analytics (with fallback).
    5. Appends the detected language and extracted keywords to each sheet's dictionary representation.

    Args:
        file_obj (io.BytesIO): A BytesIO object containing the Excel file's binary content.
        record_id (str): A unique identifier for the Excel record being processed.
        text_analytics_client (TextAnalyticsClient): An Azure Text Analytics client for
                                                    language detection and keyword extraction.
        max_keywords (int, optional): The maximum number of keywords to extract. Defaults to 20.

    Returns:
        list: A list of dictionaries, where each dictionary represents a processed Excel sheet
              with its Markdown content, metadata (including sheet name), detected language,
              and extracted keywords.
              Each dictionary has the following keys:
              - "context": The raw Markdown content of the sheet.
              - "context_markdown": Same as "context" (for symmetry with PDF processing).
              - "sheet_name": The name of the Excel sheet.
              - "metadata": A dictionary containing sheet-specific metadata, e.g., {"sheet": sheet_name}.
              - "record_id": The ID of the Excel record.
              - "keywords": A list of extracted keywords for the entire Excel file.
              - "excel_language": The detected language of the Excel file.
    """  
   
    excel_data = pd.read_excel(file_obj, sheet_name=None)  
    all_text = ""  
    sheet_markdown_chunks = []  
 
    for sheet_name, df in excel_data.items():  
        df = trim_empty_edges(df)
        df = df.fillna('').astype(str).applymap(str.strip)  

        markdown_table = df.to_markdown(index=False, tablefmt="github")  

        md_chunk = f"## {sheet_name}\n\n{markdown_table}"  
  
        all_text += "\n" + md_chunk  

        sheet_markdown_chunks.append(  
            {  
                "context": md_chunk,         # testo intero markdown del sheet  
                "context_markdown": md_chunk,# stessa cosa (per simmetria PDF)  
                "sheet_name": sheet_name,  
                "metadata": {"sheet": sheet_name},    # come metti page nei PDF  
                "record_id": record_id  
                # NB: Qui NON hai base64_imgs_list, puoi lasciare campo vuoto o non metterlo  
            }  
        )  
  
    #### LA PARTE EXCEL DEVO SISTEMARLA PER QUESTO IL GLOBAL TIMOEOUT_TIME È PIÙ ALTO E NON E' NELL'INPUT DELLA FUNZIONE ####
    detected_language = await detect_language_with_fallback_async(text= all_text,
                                                                   text_analytics_client=text_analytics_client,
                                                                     global_timeout_time=3000 ) if all_text.strip() else 'en'  
  

    #### GLOBAL TIMEOUT FISSATO AD UN LIVELLO MOLTO ALTO PERCHE' NON ANCORA GESTITO CORRETTAMENTE NELLE FUNZIONI DI EXCEL
    all_keywords = await extract_keywords_with_fallback_async(  
        text= all_text,language=  detected_language, text_analytics_client= text_analytics_client,global_timeout_time=3000, max_retries=3, retry_wait_sec=5  
    )
    keywords= all_keywords[:max_keywords]  
  
 
    for chunk in sheet_markdown_chunks:  
        chunk["keywords"] = keywords  
        chunk["excel_language"] = detected_language  
  
    return sheet_markdown_chunks  



async def get_excel_filelike_from_url_async(blob_url: str, sem_download: asyncio.Semaphore) -> io.BytesIO:  
    """
    Asynchronously downloads an Excel file from a given blob URL and returns it as a BytesIO object.

    It uses an `aiohttp.ClientSession` for the download and respects a download semaphore
    to control concurrency. Raises an HTTPError for non-200 responses.

    Args:
        blob_url (str): The URL of the Excel blob to download.
        sem_download (asyncio.Semaphore): An asyncio semaphore to limit concurrent Excel downloads.

    Returns:
        io.BytesIO: A file-like object (BytesIO) containing the Excel file's content if successful,
                    otherwise None if an error occurs during download.

    Logs:
        - ERROR: If an error occurs during the download process.
    """
    try:  
        async with sem_download:  
            async with aiohttp.ClientSession() as session:  
                async with session.get(blob_url) as resp:  
                    resp.raise_for_status()  
                    excel_bytes = await resp.read()  
                    return io.BytesIO(excel_bytes)  
    except Exception as e:  
        logging.error(f"Error downloading Excel from {blob_url}: {e}")  
        return None




async def upload_index_excel_from_url_async(  
    blob_url: str,  
    chat_id: str,  
    parent_id: str,  
    url: str,  
    embedding_function,  
    api_key: str,  
    client_embedding: AsyncAzureOpenAI,  
    model_deployment: str,  
    client:AsyncAzureOpenAI,  
    excel_name: str,  
    record_id: str,
    sem:asyncio.Semaphore,
    sem_download: asyncio.Semaphore,
    sem_ocr:asyncio.Semaphore,
    sem_embedding:asyncio.Semaphore,
    client_mistral: Mistral,
    text_analytics_client:TextAnalyticsClient,
    client_ImageAnalysis: ImageAnalysisClient,
    global_timeout_time: float,  
)-> bool:  
    """
    Asynchronously downloads, processes, and uploads an Excel file to an Azure Search index.

    This function orchestrates the following steps:
    1. Downloads the Excel file from the provided blob URL.
    2. Processes the Excel file to extract content from each sheet, detect language, and extract keywords.
    3. Generates embeddings for each processed sheet's content.
    4. Prepares the documents with relevant metadata and content for indexing.
    5. Uploads all prepared documents to the Azure Search index in a single batch.

    Args:
        blob_url (str): The URL of the Excel blob in Azure Storage.
        chat_id (str): A unique identifier for the chat session.
        parent_id (str): An identifier for the parent document or source.
        url (str): The URL of the Azure Search index endpoint.
        embedding_function (Callable): An asynchronous callable function responsible for generating text embeddings.
        api_key (str): The API key for authenticating with the Azure Search index.
        client_embedding (AsyncAzureOpenAI): An asynchronous Azure OpenAI client specifically for embedding generation.
        model_deployment (str): The name of the Azure OpenAI model deployment to use for embeddings.
        client (AsyncAzureOpenAI): A general-purpose asynchronous Azure OpenAI client.
        excel_name (str): The name of the Excel file.
        record_id (str): A unique identifier for the record (document).
        sem (asyncio.Semaphore): A semaphore for general concurrency control.
        sem_download (asyncio.Semaphore): A semaphore to control concurrent Excel downloads.
        sem_ocr (asyncio.Semaphore): A semaphore for OCR-related operations (not directly used here for Excel, but passed).
        sem_embedding (asyncio.Semaphore): A semaphore to control concurrency for embedding generation calls.
        client_mistral (Mistral): The client for Mistral OCR processing (not directly used here for Excel, but passed).
        text_analytics_client (TextAnalyticsClient): The Azure Text Analytics client for language detection and keyword extraction.
        client_ImageAnalysis (ImageAnalysisClient): The Azure Image Analysis client (not directly used here for Excel, but passed).

    Returns:
        None: The function performs indexing operations and does not return a value.

    Raises:
        aiohttp.ClientResponseError: If an HTTP error occurs during the indexing upload.
        aiohttp.ClientError: If a network-related error occurs during the indexing upload.
        Exception: For any unexpected errors during the process.

    Logs:
        - ERROR: If the Excel file download fails, no documents are prepared for indexing,
                 or an error occurs during embedding calculation or indexing upload.
        - WARNING: If no processed documents are found for the Excel file.
        - INFO: Upon successful indexing of the Excel document.
    """

    excel_io = await get_excel_filelike_from_url_async(blob_url= blob_url,sem_download=sem_download)  
    if not excel_io:  
        logging.error(f"upload_index_excel_from_url_async: Failed to download Excel from {blob_url}")  
        return  False
  
    documents = await process_excel(  
        file_obj=excel_io,  
        record_id=record_id,
        text_analytics_client=text_analytics_client  
    )  

    if not documents:
        logging.warning(f"upload_index_excel_from_url_async: No processed documents found for Excel {excel_name}. Skipping indexing.")
        return False
  
    dicts_with_metadata = []
    dicts_for_indexing = []  
    embedding_tasks = []  
    docs_for_embedding = []

    for doc in documents:  
        context_for_embedding = doc.get("context", "")
        number_sheet = doc["sheet_name"],
        if not context_for_embedding or context_for_embedding.strip() == "":
            logging.warning(
                f"upload_index_excel_from_url_async:Skipping embedding for empty or whitespace 'context' in (sheet: {number_sheet}) Excel document (record_id: {record_id}). "
            )
            doc["_skip_embedding_and_indexing"] = True
            docs_for_embedding.append(doc) 
            embedding_tasks.append(asyncio.sleep(0)) 
            continue 
        
        docs_for_embedding.append(doc) 
        embedding_tasks.append(  
            embedding_function(  
                text=context_for_embedding,  
                client=client_embedding,  
                model_deployment=model_deployment, 
                sem_embedding=sem_embedding 
            )  
        )  
  
    ##################DEVO AGGIUNGERE IL TIMEOUT ANCHE A QUESTO GATHER, MA PER FUTURI LAVORI, devo mettere il waii_for
    # mettere il try e gestire l'eccezione     except asyncio.CancelledError: 
    embeddings = await asyncio.gather(*embedding_tasks, return_exceptions=True)  


    for idx, doc in enumerate(docs_for_embedding): 
        if doc.get("_skip_embedding_and_indexing", False):
            logging.info(
                f"upload_index_excel_from_url_async: Skipping indexing for (sheet: {number_sheet}) Excel document (record_id: {record_id}) "
                "due to empty context or prior embedding skip."
            )
            continue 

        embedding = embeddings[idx]  
        if isinstance(embedding, Exception):  
            logging.error(f"upload_index_excel_from_url_async:Error calculating embedding for Excel sheet: {embedding}. Skipped.")  
            continue  
        
        if embedding is None:
             logging.error(f"⚠️ upload_index_excel_from_url_async: embedding_function returned None for Excel document (record_id: {record_id}). Skipping this chunk from indexing.")
             continue
        idx_id = generate_safe_id()  
        doc_payload = {  
            "@search.action": "upload",  
            "id": idx_id,  
            "content": doc["context"],  
            "content_markdown": doc["context_markdown"],  
            "filepath": blob_url,  
            "title": excel_name,  
            "content_vector": embedding,  
            "chat_id": chat_id,  
            "parent_id": parent_id,  
            "timestamp": datetime.now(timezone.utc).isoformat(),  
            "keywords": doc.get("keywords", []),  
            "page_number": doc["sheet_name"],  
            "base64_imgs_list": [],  
            "pdf_language": doc.get("excel_language", "unknown"),  
            "pdf_name": excel_name  
        }  
        dicts_for_indexing.append(doc_payload)  
  
    if not dicts_for_indexing:  
        logging.error(f"No documents prepared for indexing for Excel {excel_name}. Aborting indexing.")  
        return  False
####NOTA RISPETTO ALL'ALTRA FUNZIONE QUA NON DIVIDO IN BATCH PERCHE' PER ORA GLI EXCEL SONO PICCOLI E NON HA SENSO PERDERCI TEMPO, ALTRE PRIORITA'
    payload = {"value": dicts_for_indexing}  
    headers = {'Content-Type': 'application/json', 'api-key': api_key}  
      
    try:  
        async with aiohttp.ClientSession() as session:  
            async with session.post(url, json=payload, headers=headers) as response:  
                response.raise_for_status()  
                response_json = await response.json()  
                logging.info(f"Excel document processed and indexed successfully! Response: {response_json}")  
                return True
    except aiohttp.ClientResponseError as e:  
        logging.error(f"HTTP error during Excel indexing [{e.status}]: {e.message}")  
        return False
    except aiohttp.ClientError as e:  
        logging.error(f"Network error during Excel indexing: {e}")  
        return False
    except Exception as e:  
        logging.error(f"Unexpected error during Excel indexing: {e}") 
        return False
###################################################################################################################################################################
###################################################################################################################################################################
######## FINE ELABORAZIONE EXCEL ##################################################################################################################################
###################################################################################################################################################################
######## DECISIONE SU CHE ELABORAZIONE FARE #######################################################################################################################
###################################################################################################################################################################     

def detect_file_type(nomefile: str) -> str:  
    """
    Detects the type of file based on its extension.

    Args:
        nomefile (str): The name of the file, including its extension.

    Returns:
        str: A string indicating the detected file type:
             - 'pdf' if the file ends with '.pdf'
             - 'excel' if the file ends with '.xlsx', '.xls', '.xlsm', '.xlsb', or '.ods'
             - 'unknown' for any other file type.
    """
    normalized = nomefile.lower().strip()  
    if normalized.endswith(('.pdf',)):  
        return 'pdf'  
    elif normalized.endswith(('.xlsx', '.xls', '.xlsm', '.xlsb', '.ods')):  
        return 'excel'   
    else:  
        return 'unknown' 





async def upload_index_file_from_url_async(  
    blob_url: str,  
    pdf_name: str,  # può essere anche Excel!  
    chat_id: str,  
    parent_id: str,  
    url: str,  
    embedding_function: Callable[[str,AsyncAzureOpenAI, str, int, int], List[float]],  
    api_key: str,  
    client_embedding: AsyncAzureOpenAI,  
    model_deployment: str,  
    client: AsyncAzureOpenAI,   
    record_id: str,
    sem:asyncio.Semaphore,
    sem_download: asyncio.Semaphore,
    sem_ocr:asyncio.Semaphore,
    sem_embedding:asyncio.Semaphore,
    client_mistral: Mistral,
    text_analytics_client:TextAnalyticsClient,
    client_ImageAnalysis: ImageAnalysisClient,
    global_timeout_time: float,
): 
    """
    Asynchronously uploads and indexes a file from a given URL based on its detected type.
    It acts as a dispatcher, forwarding the indexing request to specialized functions
    for PDF and Excel files.

    Args:
        blob_url (str): The URL of the file in Azure Blob Storage.
        pdf_name (str): The name of the file (can be PDF or Excel).
        chat_id (str): A unique identifier for the chat session.
        parent_id (str): An identifier for the parent document or source.
        url (str): The URL of the Azure Search index endpoint.
        embedding_function (Callable): An asynchronous callable function for generating text embeddings.
        api_key (str): The API key for authenticating with the Azure Search index.
        client_embedding (AsyncAzureOpenAI): An asynchronous Azure OpenAI client for embedding generation.
        model_deployment (str): The name of the Azure OpenAI model deployment for embeddings.
        client (AsyncAzureOpenAI): A general-purpose asynchronous Azure OpenAI client (e.g., for GPT-4V).
        record_id (str): A unique identifier for the record (document).
        sem (asyncio.Semaphore): A semaphore to control concurrency for general OpenAI calls.
        sem_download (asyncio.Semaphore): A semaphore to control concurrency for file downloads.
        sem_ocr (asyncio.Semaphore): A semaphore to control concurrency for OCR-related operations.
        sem_embedding (asyncio.Semaphore): A semaphore to control concurrency for embedding generation calls.
        client_mistral (Mistral): The client for Mistral OCR processing.
        text_analytics_client (TextAnalyticsClient): The Azure Text Analytics client.
        client_ImageAnalysis (ImageAnalysisClient): The Azure Image Analysis client.

    Returns:
        The result of the specialized upload function (e.g., `upload_index_pdf_from_url_async`
        or `upload_index_excel_from_url_async`).

    Raises:
        ValueError: If the detected file type is not supported (neither PDF nor Excel).
    """


    file_type = detect_file_type(pdf_name)  
    logging.info(" Processing Pdf file: %s", pdf_name )
    if file_type == 'pdf':  
        return await upload_index_pdf_from_url_async(  
            blob_url=blob_url,  
            chat_id=chat_id,  
            parent_id=parent_id,    
            url=url,  
            embedding_function=embedding_function,  
            api_key=api_key,  
            client_embedding=client_embedding,  
            model_deployment=model_deployment,  
            client=client,  
            pdf_name=pdf_name,  
            record_id=record_id,
            sem=sem,
            sem_download=sem_download,
            sem_ocr=sem_ocr,
            sem_embedding=sem_embedding,
            client_mistral= client_mistral,
            text_analytics_client= text_analytics_client,
            client_ImageAnalysis= client_ImageAnalysis ,
            global_timeout_time=global_timeout_time
        )  
    elif file_type == 'excel':  
        logging.info(" Processing Excel file: %s", pdf_name )
        return await upload_index_excel_from_url_async(  
            blob_url=blob_url,  
            chat_id=chat_id,  
            parent_id=parent_id,  
            url=url,  
            embedding_function=embedding_function,  
            api_key=api_key,  
            client_embedding=client_embedding,  
            model_deployment=model_deployment, 
            client = client,
            excel_name=pdf_name,  
            record_id=record_id,
            sem=sem,
            sem_download=sem_download,
            sem_ocr=sem_ocr,
            sem_embedding=sem_embedding,
            client_mistral= client_mistral,
            text_analytics_client= text_analytics_client,
            client_ImageAnalysis= client_ImageAnalysis,
            global_timeout_time=global_timeout_time  
        )

    else:  
        logging.error(f"Unsupported file type for '{pdf_name}'. Only PDF and Excel supported.")  
        raise ValueError("Unsupported file type")  










###################################################################################################################################################################
######## DECISIONE SU CHE ELABORAZIONE FARE #######################################################################################################################
###################################################################################################################################################################     
class ResourceManager:
    """
    Manages all resources (clients and semaphores) used by the application,
    ensuring their proper initialization and cleanup.

    This class is designed to be used as an asynchronous context manager (`async with`)
    to ensure that all resources are properly closed upon exiting the block.

    Attributes:
        clients (dict): A dictionary to store initialized clients (e.g., Mistral, Azure OpenAI).
        semaphores (dict): A dictionary to store asyncio semaphores for concurrency control.
        _cleanup_tasks (list): A list to keep track of pending cleanup tasks.
    """
    
    def __init__(self):
        self.clients = {}
        self.semaphores = {}
        self._cleanup_tasks = []
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Asynchronous context manager exit method.
        Ensures that the cleanup function is called upon exiting the `async with` block.
        """
        await self.cleanup()
    
    async def initialize_clients(self) -> bool:
        """
        Initializes all necessary service clients and semaphores for the application.
        Clients for Mistral, Azure OpenAI, Azure Text Analytics, and Azure Image Analysis are initialized.
        The function handles initialization errors for each client and logs warnings
        or critical errors depending on the resource.

        Returns:
            dict: A dictionary indicating the initialization status of each client (True for success, False for failure).
                  If Azure OpenAI client initialization fails, the process cannot proceed,
                  and the status will reflect this critical error.
        """
        
        all_status = {} 
        # Inizializza semafori
        self.semaphores = {
            'main': asyncio.Semaphore(4),
            'download': asyncio.Semaphore(2),
            'ocr': asyncio.Semaphore(4),
            'embedding': asyncio.Semaphore(3)
        }
        
        # Mistral Client
        try:
            self.clients['mistral'] = Mistral(
                api_key=MISTRAL_API_KEY,
                server_url=MISTRAL_ENDPOINT,
                retry_config=RetryConfig(
                    strategy="backoff",
                    backoff=BackoffStrategy(
                        initial_interval=500,
                        max_interval=5000,
                        exponent=2.0,
                        max_elapsed_time=20000
                    ),
                    retry_connection_errors=True
                ),
                timeout_ms=300000
            )
            logging.info("Mistral client initialized successfully")
        except Exception as e:
            self.clients['mistral'] = None
            logging.warning(f"Mistral client failed: {e} (fallback available, continuing)")  
            all_status['mistral'] = False  
        
        # Azure OpenAI Client
        try:
            self.clients['azure_openai'] = AsyncAzureOpenAI(
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                max_retries=5,
                timeout=30.0  # Timeout esplicito
            )
            all_status['azure_openai'] = True
            logging.info("Azure OpenAI client initialized successfully")
            
        except Exception as e:
            self.clients['azure_openai'] = None  
            logging.critical(f"FAILED TO INITIALIZE OpenAI client, cannot proceed: {e}")  
            all_status['azure_openai'] = False
        
        # Text Analytics Client
        try:
            credential_text_analytics = AzureKeyCredential(API_KEY_COGNITIVE_SERVICE)
            self.clients['text_analytics'] = TextAnalyticsClient(
                endpoint=ENDPOINT_COGNITIVE_SERVICE,
                credential=credential_text_analytics
            )
            all_status['text_analytics'] = True
            logging.info("Text Analytics client initialized successfully")
        except Exception as e:
            self.clients['text_analytics'] = None  
            logging.warning(f"Text Analytics client failed: {e} (fallback with YAKE/langid)")  
            all_status['text_analytics'] = False
        
        # Image Analysis Client
        try:
            credential_image_analysis = AzureKeyCredential(API_KEY_COGNITIVE_SERVICE)
            self.clients['image_analysis'] = ImageAnalysisClient(
                endpoint=ENDPOINT_COGNITIVE_SERVICE,
                credential=credential_image_analysis
            )
            all_status['image_analysis'] = True
            logging.info("Image Analysis client initialized successfully")
        except Exception as e:
            self.clients['image_analysis'] = None  
            logging.warning(f"Image Analysis client failed: {e} (fallback generic description only)")  
            
        
        return all_status
    
    async def cleanup(self):
        """
        Performs a forced cleanup of all managed resources.
        This includes canceling pending cleanup tasks, closing open clients
        (especially Azure OpenAI, which requires explicit asynchronous closure),
        and invoking the garbage collector to free memory.
        """
        logging.info("Starting resource cleanup...")
        
        # Cancella task di cleanup pendenti
        for task in self._cleanup_tasks:
            if not task.done():
                task.cancel()
        
        # Pulisci client Azure OpenAI
        if 'azure_openai' in self.clients and self.clients['azure_openai']:
            try:
                await self.clients['azure_openai'].close()
                logging.info("Azure OpenAI client closed")
            except Exception as e:
                logging.error(f"Error closing Azure OpenAI client: {e}")
        
        # Pulisci altri client se hanno metodi di chiusura
        for client_name, client in self.clients.items():
            if client and hasattr(client, 'close'):
                try:
                    if asyncio.iscoroutinefunction(client.close):
                        await client.close()
                    else:
                        client.close()
                    logging.info(f"{client_name} client closed")
                except Exception as e:
                    logging.error(f"Error closing {client_name} client: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Small pause to allow connections to close
        await asyncio.sleep(0.1)
        
        logging.info("Resource cleanup completed")




def validate_request_payload(req_body: Dict[str, Any]) -> bool:
    """
    Validates that the request payload has the expected structure for idempotent processing.
    
    Args:
        req_body: The request body to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(req_body, dict):
        return False
    
    if 'values' not in req_body:
        return False
    
    values = req_body['values']
    if not isinstance(values, list):
        return False
    
    # Validate each value has required fields
    required_fields = ['RecordId', 'blob_url', 'pdf_name', 'is_processed', 'chat_id']
    for value in values:
        if not isinstance(value, dict):
            return False
        for field in required_fields:
            if field not in value:
                return False
    
    return True





#async def process_pdf_indexing_request(req_body: dict) -> dict:  
#
#
#    sem = asyncio.Semaphore(2)
#    sem_download = asyncio.Semaphore(2)     
#    sem_ocr = asyncio.Semaphore(2)          
#    sem_embedding = asyncio.Semaphore(2)   
#
#    try:  
#        client_mistral = Mistral(  
#            api_key=MISTRAL_API_KEY,  
#            server_url=MISTRAL_ENDPOINT,
#            retry_config=RetryConfig(
#                strategy= "backoff",
#                backoff=BackoffStrategy(
#                    initial_interval=500,       # 0.5 secondi
#                    max_interval=5000,          # massimo 5 secondi tra due retry
#                    exponent=2.0,               # backoff esponenziale (500, 1000, 2000, ecc.)
#                    max_elapsed_time=20000      # massimo 20 secondi totali di retry
#                ),
#                retry_connection_errors= True
#            ),
#            timeout_ms=30000        
#            )  
#    except Exception as e:  
#        client_mistral = None  
#        logging.critical(f"Failed to initialize Mistral client: {e}")  
#    
#
#
#    
#    try:  
#        client_azure_openai_async = AsyncAzureOpenAI(  
#            api_version=AZURE_OPENAI_API_VERSION,  
#            azure_endpoint=AZURE_OPENAI_ENDPOINT,  
#            api_key=AZURE_OPENAI_API_KEY,
#            max_retries=5  
#        )  
#    except Exception as e:  
#        client_azure_openai_async = None  
#        logging.critical(f"Failed to initialize AsyncAzureOpenAI client: {e}")  
#
#
#
#
#    try:  
#        credential_text_analytics = AzureKeyCredential(API_KEY_COGNITIVE_SERVICE)  
#        text_analytics_client = TextAnalyticsClient(  
#            endpoint=ENDPOINT_COGNITIVE_SERVICE,  
#            credential=credential_text_analytics  
#        )  
#    except Exception as e:  
#        text_analytics_client = None  
#        logging.critical(f"Failed to initialize TextAnalyticsClient: {e}")  
#
#
#
#    
#    try:  
#        credential_image_analysis = AzureKeyCredential(API_KEY_COGNITIVE_SERVICE)  
#        client_ImageAnalysis = ImageAnalysisClient(  
#            endpoint=ENDPOINT_COGNITIVE_SERVICE,  
#            credential=credential_image_analysis  
#        )  
#    except Exception as e:  
#        client_ImageAnalysis = None  
#        logging.critical(f"Failed to initialize ImageAnalysisClient: {e}")  
#
#
# 
#
#
#
#    log_suffix = "AIProcessPdfUploadIndexScript: "  
#    values = req_body.get("values", [])  
#   
#  
#    processed = await process_values(values= values,
#        log_suffix= log_suffix,
#        client_openai= client_azure_openai_async,
#        sem=sem,
#        sem_download=sem_download,
#        sem_ocr=sem_ocr,
#        sem_embedding=sem_embedding,
#        client_mistral=client_mistral,
#        text_analytics_client=text_analytics_client,
#        client_ImageAnalysis=client_ImageAnalysis
#          )  
#    results = processed["results"]  
#  
#    processable = [r for r in results if r["status"] == "ok"]  
#    final_parent_ids_to_check = list(set(r["parent_id"] for r in processable))  
#  
#    not_found_parent_ids = await wait_for_parent_ids_async(final_parent_ids_to_check, timeout_seconds=180, poll_interval=5)  
#    found_parent_ids = set(final_parent_ids_to_check) - set(not_found_parent_ids)  
#  
#    error_upload_documents = [r for r in results if r["status"] == "failed"]  
#    upload_documents = [  
#        {"RecordId": r["RecordId"]}  
#        for r in processable if r["parent_id"] in found_parent_ids  
#    ]  
#    timed_out_docs = [  
#        {"RecordId": r["RecordId"], "error": "Timeout waiting for indexing"}  
#        for r in processable if r["parent_id"] not in found_parent_ids  
#    ]  
#    error_upload_documents.extend(timed_out_docs)  
#  
#    response_body = {  
#        "uploaded_documents": upload_documents,  
#        "failed_document": error_upload_documents,  
#        "results": results,  
#    }  
#    return response_body  
#
#
#
#  
#def process_pdf_indexing_request(req_body: dict) -> dict:  
#    """Processes a batch PDF indexing request.
#
#    This function mirrors the core logic of the Azure Function's main processing
#    (as described in `AIProcessPdfUploadIndexFunction.py`) but operates as a
#    pure Python function. It receives a Python dictionary as input and returns
#    a dictionary as output, making it suitable for direct integration into
#    scripts or applications without HTTP concerns.
#
#    The processing involves:
#    - Initializing an Azure OpenAI client.
#    - Asynchronously processing a list of documents (presumably downloading,
#      extracting content, enriching with AI, splitting into chunks, and
#      generating embeddings).
#    - Uploading processed chunks to Azure Cognitive Search.
#    - Polling Azure Cognitive Search to confirm successful indexing of documents
#      based on their parent IDs.
#    - Categorizing documents into successfully uploaded and failed (due to
#      processing errors or indexing timeouts).
#
#    Args:
#        req_body (dict): A dictionary containing the input payload, expected to
#                         have a 'values' key whose value is a list of documents
#                         to be processed. Each document in the list should be
#                         a dictionary with relevant metadata (e.g., RecordId,
#                         blob_url, pdf_name, etc.).
#
#    Returns:
#        dict: A dictionary summarizing the processing results, similar to the
#              Azure Function's HTTP response body. It includes:
#              - 'uploaded_documents': A list of dictionaries, each with a
#                                      'RecordId' for successfully indexed documents.
#              - 'failed_document': A list of dictionaries, each with a 'RecordId'
#                                   and an 'error' description for documents that
#                                   failed processing or timed out during indexing.
#              - 'results': A list of raw processing results for each document,
#                           containing 'status' (e.g., 'ok', 'failed') and other
#                           details including 'parent_id'.
#    """
#    log_suffix = "AIProcessPdfUploadIndexScript: "  
#  
#    # Prendi la lista di documenti da processare  
#    values = req_body.get("values", [])  
#  
#    # Inizializza il client OpenAI come facevi nel main  
##    client_openai = AzureOpenAI(  
##        api_version=AZURE_OPENAI_API_VERSION,  
##        azure_endpoint=AZURE_OPENAI_ENDPOINT,  
##        api_key=AZURE_OPENAI_API_KEY,  
##    )  
#  
#    loop = asyncio.get_event_loop()  
#    processed = loop.run_until_complete(process_values(values, log_suffix))  
#    results = processed["results"]  
#  
#    processable = [r for r in results if r["status"] == "ok"]  
#    final_parent_ids_to_check = list(set(r["parent_id"] for r in processable))  
#  
#    not_found_parent_ids = await wait_for_parent_ids_async(final_parent_ids_to_check, timeout_seconds=180, poll_interval=5)  
#    found_parent_ids = set(final_parent_ids_to_check) - set(not_found_parent_ids)  
#  
#    error_upload_documents = [r for r in results if r["status"] == "failed"]  
#    upload_documents = [  
#        {"RecordId": r["RecordId"]}  
#        for r in processable if r["parent_id"] in found_parent_ids  
#    ]  
#    timed_out_docs = [  
#        {"RecordId": r["RecordId"], "error": "Timeout waiting for indexing"}  
#        for r in processable if r["parent_id"] not in found_parent_ids  
#    ]  
#    error_upload_documents.extend(timed_out_docs)  
#  
#    response_body = {  
#        "uploaded_documents": upload_documents,  
#        "failed_document": error_upload_documents,  
#        "results": results,  
#    }  
#    return response_body  

  
async def process_pdf_indexing_request(req_body: dict) -> dict:  
    """  
    Funzione batch analoga a main(), ma senza HttpResponse: ritorna solo il JSON dict finale.  
    Args:  
        req_body (dict): payload già decodificato ({ "values": [...] })  
    Returns:  
        dict: { "values": values_processed } come nel body della HttpResponse della main  
    """  
    log_suffix = 'DurableIdemAIProcessPdfUploadIndexFunction.py: '  
    logging.warning("Activity chiamata (no HTTP trigger, pure Python)!")  
  
    global_start_time = time.time()  
    global_timeout_time = global_start_time + 215.0  
  
    try:  
        logging.info(f'{log_suffix} Triggered process_pdf_indexing_request().')  
        values = req_body.get('values', [])  
        # pattern: la funzione process_values modifica 'values' in-place sulle is_processed etc.  
                # Validate request payload structure for idempotency
        if not validate_request_payload(req_body):
            logging.error(f'{log_suffix}:Invalid request payload structure. No Retry')
            return req_body
  
        async with ResourceManager() as resource_manager:  
            try:  
                status_clients = await resource_manager.initialize_clients()  
  
                if not status_clients['azure_openai']:  
                    logging.critical("Critical: Azure OpenAI (embedding/chat) not initialized. Aborting process!")  
                    # Come main(), ritorna l'input come body, status semi-error  
                    return req_body  
  
                # Optional: warning per client non essenziali  
                optional_failed = [key for key, val in status_clients.items() if not val and key != 'azure_openai']  
                if optional_failed:  
                    logging.warning(f"Some optional clients failed to initialize (fallbacks will be used): {optional_failed}")  
  
                # Timeout check  
                if time.time() >= global_timeout_time:  
                    logging.error(f"{log_suffix} Global timeout reached before processing")  
                    return req_body  
  
                try:  
                    results =await process_values(  
                        values=values,  
                        log_suffix=log_suffix,  
                        client_openai=resource_manager.clients.get('azure_openai'),  
                        sem=resource_manager.semaphores['main'],  
                        sem_download=resource_manager.semaphores['download'],  
                        sem_ocr=resource_manager.semaphores['ocr'],  
                        sem_embedding=resource_manager.semaphores['embedding'],  
                        client_mistral=resource_manager.clients.get('mistral'),  
                        text_analytics_client=resource_manager.clients.get('text_analytics'),  
                        client_ImageAnalysis=resource_manager.clients.get('image_analysis'),  
                        global_timeout_time=global_timeout_time  
                    )  
                except Exception as e:  
                    logging.error(f"{log_suffix} Error in process_values: {e}")  
                    # Stessa logica: ritorna il payload originale in caso di errori gravi  
                    return req_body  

                if results["results"]== []: 
                    logging.warning(f"{log_suffix}: No file was elaborated in this request because an exception was raised in the process_values function.")
  
                final_return = {"values": values}  

                elapsed_time = time.time() - global_start_time  
                logging.info(f'{log_suffix} Successfully processed PDF in {elapsed_time:.2f} seconds.')  
                return final_return  
  
            except Exception as inner_e:  
                logging.exception(f"{log_suffix} Error within ResourceManager context: {inner_e}")  
                return req_body  
  
    except Exception as outer_e:  
        logging.exception(f"{log_suffix} Unhandled exception in process_pdf_indexing_request: {outer_e}")  
        return req_body  


