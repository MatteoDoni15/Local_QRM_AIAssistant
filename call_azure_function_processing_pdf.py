import requests  
import json  
from blob_storage_manager import upload_file_to_blob, move_old_files_to_archive, generate_chat_id, update_blob_metadata, set_isdeleted_true_on_all_blobs

def build_json_value(record_id: int, chat_id: str, blob_url: str, is_processed: bool, pdf_name: str = "") -> dict:  
    """  
    Build a JSON value dictionary for a single PDF document upload.  
  
    Args:  
        record_id (int): Sequential or unique identifier for the document (as in 'RecordId').  
        chat_id (str): The chat/session/user identifier for this document.  
        blob_url (str): Full URL to the uploaded blob including the SAS token for access.  
        sas_token (str): The SAS token string (for logging or integration purposes).  
        pdf_name (str, optional): The display name of the PDF file (default: "").  
  
    Returns:  
        dict: Dictionary in the format required by the Azure indexing function.  
    """  
    return {  
        "RecordId": str(record_id),  
        "chat_id": chat_id,  
        "blob_url": blob_url,  
        "is_processed": is_processed,  
        "pdf_name": pdf_name  
    }



def call_AIProcessPdfUploadIndexFunction_azure_function(  
    function_url: str,  
    input_data: dict,  
    api_key: str = None,  
    timeout: int = 120  
) -> dict:  
    """  
    Calls the Azure Function endpoint for batch PDF processing and indexing.  
  
    This function sends a POST request to the specified Azure Function HTTP trigger endpoint  
    with the provided JSON payload. Optionally, an API key for the Azure Function can be specified.  
  
    Args:  
        function_url (str): The HTTP endpoint URL of the Azure Function (including any query string,  
                            e.g., a function code if authentication is needed).  
        input_data (dict): The JSON payload as a Python dictionary, must include a 'values' key with the list of items to index.  
        api_key (str, optional): An API key or function code if the function is protected.  
        timeout (int, optional): Timeout for the HTTP request in seconds (default 120).  
  
    Returns:  
        dict: The parsed JSON response from the Azure Function, or an error dictionary if the call fails.  
  
    Example:  
        result = call_AIProcessPdfUploadIndexFunction_azure_function(  
            function_url='https://<function-app>.azurewebsites.net/api/AIProcessPdfUploadIndexFunction?code=...',  
            input_data={  
                "values": [  
                    {"RecordId": "1", ...},  
                    ...  
                ]  
            }  
        )  
    """  
  
    headers = {  
        'Content-Type': 'application/json',  
    }  
    if api_key:  
        headers['x-functions-key'] = api_key  # Or pass via query string: ?code=...  
  
    try:  
        response = requests.post(  
            function_url,  
            data=json.dumps(input_data),  
            headers=headers,  
            timeout=timeout  
        )  
        response.raise_for_status()  
        return response.json()  
    except requests.RequestException as e:  
        # Return a structured error dict  
        return {  
            "error": "RequestException",  
            "message": str(e),  
            "details": getattr(e.response, 'text', None)  
        }  
    except Exception as e:  
        return {  
            "error": "Exception",  
            "message": str(e)  
        }  