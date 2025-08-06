import ssl
import urllib.request
import urllib.error
import json
from typing import Dict, Any
import datetime
import os
from datetime import datetime
import json

import openpyxl
from collections import defaultdict




# Constants for API paths and configurations
DATA_SAVE_HTML_REPOSITORY_PATH_PROSP="..\..\..\data\ARA_VC\Tests"
DATA_SAVE_RAWJSON_REPOSITORY_PATH_PROSP= "..\..\data\ARA_VC\Data"
DATA_SAVE_MARKDOWN_REPOSITORY_PATH_PROSP="..\..\..\data\ARA_VC\Markdown"

DATA_SAVE_HTML_REPOSITORY_PATH_BPM = "..\..\..\data\LP - BPM\Tests"
DATA_SAVE_RAWJSON_REPOSITORY_PATH_BPM="..\..\..\data\LP - BPM\Data"
DATA_SAVE_MARKDOWN_REPOSITORY_PATH_BPM="..\..\..\data\LP - BPM\Markdown"

DATA_SAVE_HTML_REPOSITORY_PATH_ARA_PE="..\..\..\data\ARA_PE\Tests"
DATA_SAVE_RAWJSON_REPOSITORY_PATH_ARA_PE="..\..\..\data\ARA_PE\Data"
DATA_SAVE_MARKDOWN_REPOSITORY_PATH_ARA_PE="..\..\..\data\ARA_PE\Markdown"

DATA_SAVE_HTML_REPOSITORY_PATH_DATAEXTR="..\..\..\data\Data-Extraction\Tests"
DATA_SAVE_RAWJSON_REPOSITORY_PATH_DATAEXTR="..\..\..\data\Data-Extraction\Data"
DATA_SAVE_MARKDOWN_REPOSITORY_PATH_DATAEXTR="..\..\..\data\Data-Extraction\Markdown"

DATA_SAVE_HTML_REPOSITORY_PATH_ARA_PD= "..\..\..\data\ARA_PD\Tests"
DATA_SAVE_RAWJSON_REPOSITORY_PATH_ARA_PD= "..\..\..\data\ARA_PD\Data"
DATA_SAVE_MARKDOWN_REPOSITORY_PATH_ARA_PD= "..\..\..\data\ARA_PD\Markdown"

DATA_SAVE_HTML_REPOSITORY_PATH= "..\..\..\data\Generic_Tests"
DATA_SAVE_RAWJSON_REPOSITORY_PATH= "..\..\..\data\Generic_Data"
DATA_SAVE_MARKDOWN_REPOSITORY_PATH= "..\..\..\data\Generic_Markdown"

COST_MILLION_TOKEN_INPUT= 1.70
COST_MILLION_TOKEN_OUTPUT=7.04





def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

def ask_question_with_token_context( data: Dict[str, Any], url: str, api_key: str) -> Dict[str, Any]:
    """Invia un JSON all'API di Azure."""
    allowSelfSignedHttps(True)

    if not api_key:
        raise ValueError("❌ Errore: API key non fornita.")

    # Stampa il messaggio di attesa prima di inviare la richiesta
    print("🤔 Bella domanda, fammi pensare un attimo...")


    # Codifica il corpo della richiesta in formato JSON
    body = str.encode(json.dumps(data))

    # URL dell'endpoint
     

    # Intestazioni della richiesta
    headers = {'Content-Type': 'application/json', 
                   'Accept': 'application/json', 
                   'Authorization': ('Bearer ' + api_key)}

    # Crea la richiesta
    req = urllib.request.Request(url, body, headers)
    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read().decode("utf-8"))
        return result
    except urllib.error.HTTPError as error:
        # Legge il messaggio di errore dal server
        error_message = error.read().decode("utf8", 'ignore')

        print(f"❌ Richiesta fallita con status code: {error.code}")
        print(error.info())
        print(error_message)

        return {
            "error": "Errore durante la richiesta all'API",
            "status_code": error.code,
            "details": error_message
        }
    



def load_all_questions_from_excel(filepath):
    wb = openpyxl.load_workbook(filepath)
    ws = wb.active
    questions_by_list = defaultdict(list)
    for row in ws.iter_rows(min_row=2, values_only=True):
        list_name, question, level1, level2, level3, level4 = row
        questions_by_list[list_name].append({
            "question": question,
            "level1": level1,
            "level2": level2,
            "level3": level3,
            "level4": level4
        })
    return dict(questions_by_list)



def load_prompts_from_json(filepath):
    """
    Carica i campi dal file prompt.json e restituisce un dizionario.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    return prompts



def save_response_json(response: dict, filename_prefix: str = "test") -> str:
    """
    Salva la risposta JSON in un file nella cartella DATA_SAVE_RAWJSON_REPOSITORY_PATH.

    Args:
        response (dict): Il contenuto JSON da salvare.
        filename_prefix (str): Prefisso per il nome del file (default "response").

    Returns:
        str: Il percorso completo del file salvato.
    """
    # Assicurati che la cartella esista
    if filename_prefix == "prosp_analysis":
      os.makedirs(DATA_SAVE_RAWJSON_REPOSITORY_PATH_PROSP, exist_ok=True)
      DATE_SAVE_PATH_RAWJSON= DATA_SAVE_RAWJSON_REPOSITORY_PATH_PROSP
    elif filename_prefix == "BPM_analysis":
      os.makedirs(DATA_SAVE_RAWJSON_REPOSITORY_PATH_BPM, exist_ok=True) 
      DATE_SAVE_PATH_RAWJSON= DATA_SAVE_RAWJSON_REPOSITORY_PATH_BPM
    elif filename_prefix == "ARA_2024_private_equity" or filename_prefix == "ARA_smart_private_equity" :
      os.makedirs(DATA_SAVE_RAWJSON_REPOSITORY_PATH_ARA_PE, exist_ok=True)
      DATE_SAVE_PATH_RAWJSON= DATA_SAVE_RAWJSON_REPOSITORY_PATH_ARA_PE 
    elif filename_prefix == "data_extraction":
      os.makedirs(DATA_SAVE_RAWJSON_REPOSITORY_PATH_DATAEXTR, exist_ok=True) 
      DATE_SAVE_PATH_RAWJSON= DATA_SAVE_RAWJSON_REPOSITORY_PATH_DATAEXTR
    elif filename_prefix == "ARA_private_debt":
      os.makedirs(DATA_SAVE_RAWJSON_REPOSITORY_PATH_ARA_PD, exist_ok=True)  
      DATE_SAVE_PATH_RAWJSON= DATA_SAVE_RAWJSON_REPOSITORY_PATH_ARA_PD
    else:
      os.makedirs(DATA_SAVE_RAWJSON_REPOSITORY_PATH, exist_ok=True) 
      DATE_SAVE_PATH_RAWJSON= DATA_SAVE_RAWJSON_REPOSITORY_PATH






    # Genera un nome file con timestamp, ad esempio response_20250523_153012.json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename_prefix}.json"
    full_path = os.path.join(DATE_SAVE_PATH_RAWJSON, filename)

    # Salva il file JSON
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=4)

    return full_path

def save_markdown_response(md_text:str, filename_prefix: str = "test") -> str:

    if filename_prefix == "prosp_analysis":
      os.makedirs(DATA_SAVE_MARKDOWN_REPOSITORY_PATH_PROSP, exist_ok=True)
      MARKDOWN_SAVE_PATH= DATA_SAVE_MARKDOWN_REPOSITORY_PATH_PROSP
    elif filename_prefix == "BPM_analysis":
      os.makedirs(DATA_SAVE_MARKDOWN_REPOSITORY_PATH_BPM, exist_ok=True) 
      MARKDOWN_SAVE_PATH= DATA_SAVE_MARKDOWN_REPOSITORY_PATH_BPM
    elif filename_prefix == "ARA_2024_private_equity" or filename_prefix == "ARA_smart_private_equity" :
      os.makedirs(DATA_SAVE_MARKDOWN_REPOSITORY_PATH_ARA_PE, exist_ok=True)
      MARKDOWN_SAVE_PATH= DATA_SAVE_MARKDOWN_REPOSITORY_PATH_ARA_PE
    elif filename_prefix == "data_extraction":
      os.makedirs(DATA_SAVE_MARKDOWN_REPOSITORY_PATH_DATAEXTR, exist_ok=True) 
      MARKDOWN_SAVE_PATH= DATA_SAVE_MARKDOWN_REPOSITORY_PATH_DATAEXTR
    elif filename_prefix == "ARA_private_debt":
      os.makedirs(DATA_SAVE_MARKDOWN_REPOSITORY_PATH_ARA_PD, exist_ok=True)  
      MARKDOWN_SAVE_PATH= DATA_SAVE_MARKDOWN_REPOSITORY_PATH_ARA_PD
    else:
      os.makedirs(DATA_SAVE_MARKDOWN_REPOSITORY_PATH, exist_ok=True)
      MARKDOWN_SAVE_PATH= DATA_SAVE_MARKDOWN_REPOSITORY_PATH
   

    # Genera un nome file con timestamp, ad esempio response_20250523_153012.json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename_prefix}.md"
    full_path = os.path.join(MARKDOWN_SAVE_PATH, filename)

    # Salva il file Markdown
    with open(full_path, "w", encoding="utf-8") as f:
       f.write(md_text)

    return full_path


#def save_window_to_html(window, msg_html, filename_prefix:str = "conversation"):  
#    """  
#    Save all text from window.conversation_area (QTextBrowser) into a DOCX file.  
#    """  
#    # Ensure directory exists  
#    os.makedirs(DATA_SAVE_WORD_REPOSITORY_PATH, exist_ok=True)  
#    # Compose filename  
#    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
#    html_filename = f"{filename_prefix}_{timestamp}.html"  
#    html_fullpath = os.path.join(DATA_SAVE_WORD_REPOSITORY_PATH, html_filename)  
#    with open(html_fullpath, "w", encoding="utf-8") as f:
#            f.write(msg_html) 
#    try:  
#        window.conversation_area.append(f"📄 Saved conversation to <code>{html_fullpath}</code>")  
#    except Exception:  
#        pass  
#    
#    return html_fullpath 



#senza HTML HEADER
#def save_window_to_html( conversation_area: QTextBrowser, msg_html: str, filename_prefix: str = "conversation" ):  
#    """  
#    Save all text from conversation_area (QTextBrowser) into an HTML file.  
#    """  
#
#    os.makedirs(DATA_SAVE_WORD_REPOSITORY_PATH, exist_ok=True)  
#    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
#    html_filename = f"{filename_prefix}_{timestamp}.html"  
#    html_fullpath = os.path.join(DATA_SAVE_WORD_REPOSITORY_PATH, html_filename)  
#  
#    with open(html_fullpath, "w", encoding="utf-8") as f:  
#        f.write(msg_html)  
#  
#    try:  
#        conversation_area.append(f"📄 Saved conversation to <code>{html_fullpath}</code>")  
#    except Exception:  
#        pass  
#    return html_fullpath 
#
#from datetime import datetime  
#import os  
  
def save_window_to_html(  
    conversation_area,  # QTextBrowser  
    msg_html: str,  
    filename_prefix: str = "conversation",
    language: str= "en",
    name_client: str = "Generic_Client"  
):  
    """  
    Save all text from conversation_area (QTextBrowser) into a styled HTML file.  
    """  
  
    FAVICON = "https://s3.eu-central-1.amazonaws.com/euc-cdn.freshdesk.com/data/helpdesk/attachments/production/204004181218/fav_icon/kFFhk_YySkVXMWXoz42yxHjdaC9C4ir4SA.png"  
    LOGO = "https://s3.eu-central-1.amazonaws.com/euc-cdn.freshdesk.com/data/helpdesk/attachments/production/204004181001/logo/UNkejhd-Gwglg_kyy2Yr7YxukNpvcb458g.png"  
  
    CSS = f"""  
    <style>  
      body {{  
        font-family: 'Lato', Arial, Helvetica, sans-serif;  
        color: #13344D;  
        background: #f6faff;  
        margin: 0;  
        padding: 0;  
      }}  
      .titlelogo {{  
        margin: 20px 0;  
        display:flex;  
        align-items:center;  
        gap:10px;  
      }}  
      .titlelogo img {{  
        height:44px; border-radius:6px;  
      }}  
      h1 {{  
        color:#2a7369;  
        margin-bottom:0;  
      }}  
      table {{  
        border-collapse: collapse;  
        background: #FFF;  
        border: 2px solid #277CB7;  
        border-radius: 8px;  
        margin-bottom: 1.5em;  
        width: 100%;  
        overflow: auto;  
      }}  
      th, td {{  
        border: 1px solid #277CB7;  
        padding: 8px 12px;  
        text-align: center;  
      }}  
      th {{  
        background: #277CB7;  
        color: white;  
        font-size: 16px;  
      }}  
      tr:nth-child(even) {{ background: #F3F6FC; }}  
      code, pre {{  
        background: #eaeef5;  
        color: #277CB7;  
        border-radius: 4px;  
        padding:2px 4px;  
      }}  
      a {{ color:#2a7369; }}  
    </style>  
    """  
  
    # === 2. Header HTML completo ===  
    HTML_TOP = f"""<!DOCTYPE html>  
        <html lang="it">  
        <head>  
          <meta charset="utf-8">  
          <title>Quantyx Test</title>  
          <link rel="shortcut icon" href="{FAVICON}">  
          <link rel="icon" href="{FAVICON}">  
          <link rel="apple-touch-icon" href="{FAVICON}">  
          <meta name="viewport" content="width=device-width, initial-scale=1.0">  
          <link href="https://fonts.googleapis.com/css?family=Lato:regular,italic,700,900,900italic" rel="stylesheet" type="text/css">  
        {CSS}  
        </head>  
        <body>  
        <div class="titlelogo">  
          <img src="{LOGO}" alt="Quantyx Logo">  
          <h1>Quantyx Test</h1>  
        </div>  
        <div class="conversation-content">  
        {msg_html}  
        </div>  
        </body>  
        </html>  
        """  
    
    language_code_map = {
        "english": "en",
        "italian": "it",
        "french": "fr",
        "spanish": "es",
        "german": "de",
        "portuguese": "pt",
        "chinese": "zh",
        "japanese": "ja",
        "russian": "ru",
        "arabic": "ar",

        "italiano": "it",
        "francese": "fr",
        "spagnolo": "es",
        "tedesco": "de",
        "portoghese": "pt",
        "cinese": "zh",
        "giapponese": "ja",
        "russo": "ru",
        "arabo": "ar",

        # Aggiungi altre varianti se necessarie, sempre in minuscolo
        "eng": "en",
        "ita": "it",

    }
    
    # Converte il nome esteso della lingua in sigla.
    # Se la lingua non è nella mappa, usa il valore originale (o un default come 'un' per unknown)
    language_code = language_code_map.get(language.lower(), language.lower())


    if filename_prefix == "ARA_smart_venture_capital":
      os.makedirs(DATA_SAVE_HTML_REPOSITORY_PATH_PROSP, exist_ok=True) 
      DATA_SAVE_PATH = DATA_SAVE_HTML_REPOSITORY_PATH_PROSP
    elif filename_prefix == "BPM_analysis":
      os.makedirs(DATA_SAVE_HTML_REPOSITORY_PATH_BPM, exist_ok=True) 
      DATA_SAVE_PATH = DATA_SAVE_HTML_REPOSITORY_PATH_BPM
    elif filename_prefix == "ARA_2024_private_equity" or filename_prefix == "ARA_smart_private_equity" :
      os.makedirs(DATA_SAVE_HTML_REPOSITORY_PATH_ARA_PE, exist_ok=True) 
      DATA_SAVE_PATH= DATA_SAVE_HTML_REPOSITORY_PATH_ARA_PE
    elif filename_prefix == "data_extraction":
      os.makedirs(DATA_SAVE_HTML_REPOSITORY_PATH_DATAEXTR, exist_ok=True) 
      DATA_SAVE_PATH= DATA_SAVE_HTML_REPOSITORY_PATH_DATAEXTR
    elif filename_prefix == "ARA_private_debt":
      os.makedirs(DATA_SAVE_HTML_REPOSITORY_PATH_ARA_PD, exist_ok=True)  
      DATA_SAVE_PATH= DATA_SAVE_HTML_REPOSITORY_PATH_ARA_PD  
    else:
      os.makedirs(DATA_SAVE_HTML_REPOSITORY_PATH, exist_ok=True) 
      DATA_SAVE_PATH= DATA_SAVE_HTML_REPOSITORY_PATH
    #os.makedirs(DATA_SAVE_WORD_REPOSITORY_PATH, exist_ok=True)  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
    html_filename = f"{filename_prefix}_{timestamp}_{name_client}_{language_code}.html"  
    html_fullpath = os.path.join(DATA_SAVE_PATH, html_filename)  
  
    # SCRIVI su file  
    with open(html_fullpath, "w", encoding="utf-8") as f:  
        f.write(HTML_TOP)  
  
    try:  
        conversation_area.append(f"📄 Saved conversation to <code>{html_fullpath}</code>")  
    except Exception:  
        pass  
    return html_fullpath  