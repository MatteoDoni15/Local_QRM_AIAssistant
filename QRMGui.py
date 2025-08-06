from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTextBrowser, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QLabel, QLineEdit
)
#from indexing import upload_index_pdf, get_embedding
#from pdf_processing_gpt41Ocr import generate_chat_id
#from pdf_processing_noOcr import generate_chat_id
#####NOTA: PER USARE MISTRAL DEVO RIMPOSTARE LA FUNZIONE upload file ma non e? importante ora
#from pdf_processing_MistralOcr import generate_chat_id

from questioning import ask_question_with_token_context, save_markdown_response, save_response_json,save_window_to_html, load_all_questions_from_excel, load_prompts_from_json # save_window_to_docx,add_json_table_to_docx   #, , save_html_to_docx, append_html_to_docx,
from index_indexer_manager import run_indexer
from blob_storage_manager import upload_file_to_blob, move_old_files_to_archive, generate_chat_id, update_blob_metadata, set_isdeleted_true_on_all_blobs
from collections_json_schema_str import *



import re
import markdown2
import sys
import time
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.identity import DefaultAzureCredential
from typing import List, Any, Dict
from openai import AzureOpenAI 
import json
import html
import os
from dotenv import load_dotenv




from collections_json_schema_str import JSON_SCHEMA_EXTRACTEDDATA,JSON_SCHEMA_COMPANY_INCOME_STATEMENT,JSON_SCHEMA_D,JSON_SCHEMA_EASY

from call_azure_function_processing_pdf import call_AIProcessPdfUploadIndexFunction_azure_function,build_json_value
from AIProcessPdfUploadIndexFunction import *
from questions_editor_dialog import QuestionsEditorDialog



system = """You are a virtual assistant for a financial web application developed by the QRM company.
       Your primary responsibility is to provide structured, detailed, and informative responses to customers, ensuring clarity and completeness while maintaining a formal and courteous tone. 
       Provide concise, structured answers focusing on key insights, relevant details, and actionable considerations. 
       Avoid unnecessary length and keep responses focused on the essentials. 
       When applicable, use bullet points or numbered lists to improve readability. 
       If the response involves financial data, forecasts, or trends, present the information using markdown-formatted tables and structured breakdowns for clarity. 
       If you are unsure of the answer or cannot find the information in the text, reply with "I couldn't find any information in the text." translated into the appropriate language.
       When solving formulas, guide the user through the solution step by step.\n

       ---

       The user will submit a list of independent financial questions in a single message,
        each question associated with up to four heading levels (Heading 1, Heading 2, Heading 3, Heading 4).
        If fewer than four heading levels are specified, it means only those levels are present
         (for example, one, two, or three headings). If no heading levels are specified,
         it means none are present. The questions will be written enclosed between double dollar signs $$...$$.
         If your answer contains formulas, always present every formula in MathML format, strictly following these rules:
           - Every MathML block must start with <math xmlns="http://www.w3.org/1998/Math/MathML"> and end with </math>. Always include the xmlns namespace exactly as shown.
           -Never enclose MathML in backticks, code blocks, or any markdown formatting.
           -Output only raw, valid MathML, with no additional explanation, commentary, or transcription. Do not paraphrase, describe, or explain MathML.
           -Make sure all MathML code is syntactically correct and fully renderable in modern web browsers as-is.
         Whenever possible, return citations and always explicitly prefix them with Source:. The Source: label must always appear exactly as written, before the citation text.
           -For a single document, use: (Source: [Document Name – pg XX, YY]) if the pages are non-consecutive, or (Source: [Document Name – pg XX-YY]) if the pages form a continuous range.
           -For multiple documents, separate each citation with a semicolon, for example: (Source: [Document Name 1 – pg XX, YY ; Document Name 2 – pg WW, VV]).
           -Do not omit the Source: label, even if only one document is cited.
           -Place the citation at the end of the relevant answer sentence or paragraph.

           MANDATORY: You must enclose the entire citation in triple carets like this: ^^^(Source: [Document Name – pg XX])^^^ every time a citation appears within the body of an answer (i.e., anywhere outside the summary table).
This formatting is strictly required and must never be omitted or replaced.
           In the summary table, use the standard format as described above, without triple carets.

        The assistant must not rewrite, rephrase, or modify the questions in any way, as they are considered a company secret

       Please organize your response hierarchically by grouping answers under their respective headings, using markdown heading styles to differentiate levels:

       - Use `#` for Heading 1 
       - Use `##` for Heading 2 
       - Use `###` for Heading 3 
       - Use `####` for Heading 4

        Additionally, ensure that all answers related to the same heading level are grouped together under that heading, and if multiple questions are under the same heading, group all the questions' answers under that single heading instance.

       For each question under the lowest relevant heading, provide only the answer, without restating or rephrasing the question. 


       After answering all the questions, **create a summary table to consolidate the key information provided in your answers**.  
       The table must include the following columns: 
       - **Topic/Clause**: The main subject or clause addressed. 
       - **Key Details**: A brief summary of the relevant information. 
       - **Source Reference**: The document name and page number if available. 
       If multiple points need to be summarized, repeat the structure in the same table.
        You may add additional columns as needed to enhance clarity and completeness.
 
       --- 

       Example:

       # Heading 1 - Topic A 

       ## Heading 2 - Subtopic A.1 

       ### Heading 3 - Detail A.1.1 

       #### Heading 4 - Specific Point A.1.1.1 



       Answer to the related question...





       Topic/Clause          |  Key Details                   |     Source Reference
       ----------------------|--------------------------------|-----------------------------
       [Enter Topic/Clause]  | [Brief key information summary]|    [Document Name 1– pg. XX, YY]
       [Enter Topic/Clause]  | [Brief key information summary]|    [Document Name 2 – pg. WW; Document Name 3 – pg. ZZ]
       [Enter Topic/Clause]  | [Brief key information summary]|    [Document Name 4 – pg. XX-OO]




       Example: 
       # Financial Overview   
       ## Revenue Analysis   
       ### Q1 2025   
       #### Product A   


       The revenue for Product A in Q1 2025 increased by 12% compared to Q4 2024. ^^^(Source: Q1_Report.pdf – pg 10)^^^ 


       #### Product B   

       Product B showed a slight decline of 3% in the same period. ^^^(Source: Q1_Report.pdf – pg. 12)^^^

       ## Expense Analysis   
       ### Operational Costs   

       Operational costs increased by 5% due to higher raw material prices. ^^^(Source: Expenses_2025.pdf – pg. 7)^^^


       | Topic/Clause         | Key Details                                             | Source Reference                                              | 
       |----------------------|---------------------------------------------------------|---------------------------------------------------------------| 
       | Product A Q1 2025    | Revenue increased by 12% vs Q4 2024                     | Q1_Report.pdf – pg. 10; Expenses_2025.pdf – pg. 20            | 
       | Product B Q1 2025    | Revenue declined by 3%                                  | Q1_Report.pdf – pg. 12                                        | 
       | Operational Costs    | Increased by 5% due to higher raw material prices       | Expenses_2025.pdf – pg. 7-10                                  | 


       ---
       The context below is composed of multiple document excerpts retrieved from the knowledge base, with each excerpt
       separated by a line of equal signs (=====) to clearly distinguish information from different sources and pages.

       ---
       """



#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
###################################################################################################################################################################


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





COST_MILLION_TOKEN_INPUT= 1.70
COST_MILLION_TOKEN_OUTPUT=7.04


#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################


class AIFrontEnd(QMainWindow):
    def __init__(self):
        super().__init__()

        prompts = load_prompts_from_json("prompt.json")
        self.text_input_pre =  system # prompts["text_input_pre"]
        self.text_input_post = "" # prompts["text_input_post"]

        #self.all_questions = load_all_questions_from_excel("questions.xlsx")
        self.all_questions = load_all_questions_from_excel("questions_newsPE_VC.xlsx")

        self.indexsearch_topn_chunk= 5
        self.chat_id =  generate_chat_id() #"bb11f717-c571-4fe7-b209-dc8a994db1ae" "40f1a16d-0c39-4a72-8f3b-eda15ebd63fb" #
        print(self.chat_id)
        self.language = "English"
        self.client_id = "default_client_007E"
        self.json_schema =JSON_SCHEMA_EXTRACTEDDATA #JSON_SCHEMA_D #JSON_SCHEMA_EXTRACTEDDATA #JSON_SCHEMA_EASY #JSON_SCHEMA_COMPANY_INCOME_STATEMENT #JSON_SCHEMA_D,JSON_SCHEMA_EASY,  JSON_SCHEMA_EXTRACTEDDATA
                                                    ###NON FUNZIONANO BENE TUTTI I JSON_SCHEMA
        self.uploaded_file_paths = []
        self.uploaded_json_values = []          
        #self.ram_tables = {}  # DIZIONARIO per tenere tabelle in RAM, serve per stampare bene il file docx
        self.analysisType= ""
        #### Markdown used for rendering the output in the QTextBrowser
        self.markdown_to_save= ""
        #### Final Markdown in real generation on QRM
        self.markdown_to_save_simil_real_generation= ""
        self.setWindowTitle("QRM-AIAssistant")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # Campi input lingua e nome cliente
        self.language_label = QLabel("Lingua:")
        self.language_input = QLineEdit()
        self.language_input.setPlaceholderText("Es. Italian, English, French")
        self.language_input.editingFinished.connect(lambda: self.set_language(self.language_input.text()))

        self.client_id_label = QLabel("Nome cliente (client_id):")
        self.client_id_input = QLineEdit()
        self.client_id_input.setPlaceholderText("Inserisci il nome del cliente")
        self.client_id_input.editingFinished.connect(lambda: self.set_client_id(self.client_id_input.text()))

        self.json_schema_label = QLabel("struttara dati da recuperare (json_schema_str):")
        self.json_schema_input = QLineEdit()
        self.json_schema_input.setPlaceholderText("Inserisci i dati da recuperare")
        self.json_schema_input.editingFinished.connect(lambda: self.set_data_to_extract(self.json_schema_input.text()))

        self.indexsearch_topn_chunk_label = QLabel("numero pagine da recuperare per domanda (indexsearch_topn_chunk):")
        self.indexsearch_topn_chunk_input = QLineEdit()
        self.indexsearch_topn_chunk_input.setPlaceholderText("Inserisci i dati da recuperare")
        self.indexsearch_topn_chunk_input.editingFinished.connect(lambda: self.set_data_to_extract(self.indexsearch_topn_chunk_input.text()))


  





        # Output conversazione e pulsantiIt
        self.conversation_area = QTextBrowser()
        self.upload_button = QPushButton("Select PDF  or Excel files to upload")
        self.elabora_pdf_button = QPushButton("Elabora PDF Batch su Azure")
        self.pe_2024_analysis_button = QPushButton("Analysis 2024 Private Equity")
        self.pe_smart_analysis_button = QPushButton("Analysis Smart Private Equity")
        self.pd_analysis_button = QPushButton("Analysis Private Debt")
        self.ARA_smart_venture_capital_button = QPushButton("Analysis Smart Venture Capital")
        self.BPM_analysis_button = QPushButton("Analysis BPM")
        self.data_extraction_button = QPushButton("Data Extraction")
        #self.save_docx_button = QPushButton("Save Conversation to DOCX")It
        ####questo serviva per l'indice per applicare una una soft delete con un metadato
        #self.IsDeleted_true_button = QPushButton("Set IsDeleted==true for all the BlobStorage Pdf")



        # Aggiunta al layout
        layout.addWidget(self.language_label)
        layout.addWidget(self.language_input)
        layout.addWidget(self.client_id_label)
        layout.addWidget(self.client_id_input)
        layout.addWidget(self.json_schema_label)
        layout.addWidget(self.json_schema_input) 
        layout.addWidget(self.indexsearch_topn_chunk_label) 
        layout.addWidget(self.indexsearch_topn_chunk_input) 
        layout.addWidget(self.conversation_area)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.elabora_pdf_button)  
        layout.addWidget(self.pe_2024_analysis_button)
        layout.addWidget(self.pe_smart_analysis_button)
        layout.addWidget(self.pd_analysis_button)
        layout.addWidget(self.ARA_smart_venture_capital_button)
        layout.addWidget(self.BPM_analysis_button)
        layout.addWidget(self.data_extraction_button)
        #layout.addWidget(self.save_docx_button) 
        #layout.addWidget(self.IsDeleted_true_button) 


        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Connessioni pulsanti
        self.upload_button.clicked.connect(self.upload_pdf)
        self.pe_2024_analysis_button.clicked.connect(self.ARA_2024_private_equity_analysis)
        self.pe_smart_analysis_button.clicked.connect(self.ARA_smart_private_equity_analysis)
        self.pd_analysis_button.clicked.connect(self.ara_private_debt_analysis)
        self.ARA_smart_venture_capital_button.clicked.connect(self.ARA_smart_venture_capital)
        self.BPM_analysis_button.clicked.connect(self.BPM_analysis)
        self.data_extraction_button.clicked.connect(self.data_extraction)
        #self.save_docx_button.clicked.connect(self.save_window_to_docx_gui)  
        #self.IsDeleted_true_button.clicked.connect(self.soft_delete_strategy_using_custom_metadata)
        self.elabora_pdf_button.clicked.connect(self._process_all_pdfs_and_index_handler)


    def soft_delete_strategy_using_custom_metadata(self):  
        if self.client_id == "Wolf_the_cleaner":  
            set_isdeleted_true_on_all_blobs(  
                connection_string=AZURE_STORAGE_CONNECTION_STRING,  
                container_name=CONTAINER_NAME  
            )  
            self.conversation_area.append("✅ IsDeleted=true set for all blobs in the container.")  
        else:  
            if not self.uploaded_file_paths:  
                self.conversation_area.append("⚠️ No files uploaded to soft-delete.")  
                return  
            for file_path in self.uploaded_file_paths:  
                blob_name = f"documents/{self.client_id}/{os.path.basename(file_path)}"  
                update_blob_metadata(  
                    connection_string=AZURE_STORAGE_CONNECTION_STRING,  
                    container_name=CONTAINER_NAME,  
                    blob_name=blob_name,  
                    metadata_key="IsDeleted",  
                    metadata_value="true"  
                )  
                self.conversation_area.append(f"✅ Soft delete applied to {blob_name}.") 



    def set_language(self, text):
        self.language = text
        self.conversation_area.append(f"🌐 Language set to: {text}")
        self.markdown_to_save += f"🌐 Language set to: {text} \n\n"

    def set_data_to_extract(self, text):
        self.json_schema = text
        self.setWindowTitle(f"AI Document Assistant - data to retriever set: {text}")
        self.conversation_area.append(f"🗓️ data to retriever set to:  {text}")
        self.markdown_to_save+= f"🗓️ data to retriever set to:  {text} \n\n"

    def set_client_id(self, text):
        self.client_id = text
        self.setWindowTitle(f"AI Document Assistant - Client: {text}")
        self.conversation_area.append(f"🆔 Client name set to:  {text} \n\n")
        self.markdown_to_save+= f"🆔 Client name set to:  {text}"

#    def upload_pdf(self):
#        file_path, _ = QFileDialog.getOpenFileName(self, "Seleziona un file PDF", "", "PDF Files (*.pdf)")
#        self.uploaded_file_paths.append(file_path)
#        if file_path:
#            self.conversation_area.append(f"📂  Selected file: {file_path}")
#            self.process_file(file_path)

#    def save_window_to_docx_gui(self):  
#        try:  
#            save_window_to_docx(self)  
#            self.conversation_area.append("✅ Conversation saved to DOCX.")  
#        except Exception as e:  
#            self.conversation_area.append(f"❌ Error saving to DOCX: {e}")  

    #def save_window_to_docx_gui(self):  
    #    try:  
    #        docx_path = save_window_to_docx(self, self.ram_tables)  
    #        self.conversation_area.append(f"✅ Conversation saved to DOCX: {docx_path}")  
    #    except Exception as e:  
    #        self.conversation_area.append(f"❌ Error saving to DOCX: {e}")  

    def upload_pdf(self):  
        files, _ = QFileDialog.getOpenFileNames(  
                        self,  
                        "Select one or more PDF or Excel files",  
                        "",  
                        "PDF o Excel (*.pdf *.xls *.xlsx *.xlsm *.xlsb *.xltx *.xltm)"  
                        ) 
        if not files:  
            self.conversation_area.append("⚠️ No File selected.")  
            return  
    
        for file_path in files:  
            self.uploaded_file_paths.append(file_path)  
            self.conversation_area.append(f"📂 Selected file: {file_path}")  
    
            try:  
                upload_result = upload_file_to_blob(file_path=file_path, user_id=self.client_id, chat_id=self.chat_id)  
                if not upload_result.get("blob_name"):  
                    raise Exception(upload_result.get("message", "Unknown upload error"))  
                value_json = build_json_value(  
                    record_id=generate_safe_id(),  
                    chat_id=self.chat_id,  
                    blob_url=upload_result["blob_name"],  
                    is_processed=False,  
                    pdf_name= upload_result["pdf_name"]  
                )  
                self.uploaded_json_values.append(value_json)  
                self.conversation_area.append("✅ File uploaded and added to batch for processing.")  
            except Exception as e:  
                self.conversation_area.append(f"❌ Error uploading file {file_path}: {e}") 


#######
#######  QUESTO MEDOTO E' UTILE PER USARE L'INDICE 
#######
#    def process_file(self, file_path):
#        # Upload the file and get details back, including the exact blob_name used
#        upload_result = upload_file_to_blob(file_path=file_path, user_id=self.client_id, chat_id=self.chat_id)
#
#        if not upload_result.get("success"):
#            self.conversation_area.append(f"❌ File upload error: {upload_result.get('message', 'Unknown error')}")
#            return # Stop processing if upload failed
#
#        actual_blob_name = upload_result["blob_name"]
#        self.conversation_area.append(f"📂 File uploaded to Azure Blob Storage: {actual_blob_name}")
#        
#        start_time = time.time()  # Registra il tempo di inizio
#        try:
#            run_indexer()
#            self.conversation_area.append("✅ Document processed successfully!")
#        except Exception as e:
#            self.conversation_area.append(f"❌ Error during indexing: {str(e)}")
#        
#        end_time = time.time()
#
#        total_time_indexer = end_time - start_time
#
#        #self.print_to_output(chat_output)
#        self.conversation_area.append( f"⏱️ Indexer completed in {total_time_indexer:.2f}" )



######
######        QUESTO CODICE SERVE PER QUANDO USIAMO PER LA AZURE FUNCTION DI ELABORAZIONE
######


#    def process_all_pdfs_and_index(self):  
#        self.conversation_area.append("🔄 Starting process PDFs...")
#        start_time = time.time() 
#        if not self.uploaded_json_values:  
#            self.conversation_area.append("⚠️ No PDFs in the batch to process.")  
#            return  
#    
#        self.conversation_area.append("🚀 Starting batch PDF indexing via Azure Function...")  
#    
#        try:  
#            json_body = {"values": self.uploaded_json_values}  
#    
#            result = call_AIProcessPdfUploadIndexFunction_azure_function(  
#                function_url=AZURE_FUNCTION_URL_LOCAL, #AZURE_FUNCTION_URL  
#                input_data=json_body,  
#                api_key=AZURE_FUNCTION_KEY_LOCAL  #AZURE_FUNCTION_KEY  
#            )  
#    
#            if "error" in result:  
#                self.conversation_area.append(f" Azure Function ERROR: {result['message']}")  
#                self.markdown_to_save += f" Azure Function ERROR: {result['message']}"
#                if result.get('details'):  
#                    self.conversation_area.append(f"<pre>{result['details']}</pre>")  
#                self.markdown_to_save +=f"<pre>{result['details']}</pre>"
#            else:  
#                self.conversation_area.append("✅ Azure Function indexing completed. Result:")  
#                self.conversation_area.append(f"<pre>{json.dumps(result, indent=2, ensure_ascii=False)}</pre>") 
#                self.markdown_to_save += "✅ Local PDF indexing completed. Result: \n" + f"<pre>{json.dumps(result, indent=2, ensure_ascii=False)}</pre>\n\n"
#                self.markdown_to_save += "Files:\n "
#                self.conversation_area.append("Files::\n ")
#                for value in self.uploaded_json_values:
#                    name_file= value["pdf_name"]
#                    self.conversation_area.append("Files::\n\n ")
#                    self.markdown_to_save +=  f"> {name_file} \n\n" 
#    
#            # Svuota il batch per evitare invii doppi  
#            self.uploaded_json_values.clear()  
#    
#        except Exception as e:  
#            self.conversation_area.append(f" Exception while calling Azure Function: {str(e)}")  
#        end_time = time.time()
#        total_time = end_time - start_time
#        self.conversation_area.append( f"Total time two process is: {total_time}")   
#        self.markdown_to_save += f"Total time two process is: {total_time} \n\n ---\n\n"


######
######        QUESTO CODICE SERVE PER QUANDO USIAMO LA LIBRERIA AIPROCESSPDFUPLOADINDEXFUNCTION
######

    #def process_all_pdfs_and_index(self):  
    async def process_all_pdfs_and_index(self):
        self.conversation_area.append("🔄 Starting process PDFs...")
        start_time = time.time() 
        if not self.uploaded_json_values:  
            self.conversation_area.append("⚠️ No PDFs in the batch to process.")  
            return  
    
        self.conversation_area.append("🚀 Starting batch PDF indexing (local processing)...")  
        try:  
            json_body = {"values": self.uploaded_json_values}

            print(f'Inizia la elaborazione dei PDF in batch: {json_body}')  
    
            # CHIAMA LA FUNZIONE LOCALE (no azure function remota)  
            result = await process_pdf_indexing_request(json_body)  
    
            if isinstance(result, dict) and "error" in result:  
                self.conversation_area.append(f"❌ Local indexing ERROR: {result.get('message')}")  
                if result.get('details'):  
                    self.conversation_area.append(f"<pre>{result['details']}</pre>")  
            else:  
                self.conversation_area.append("✅ Local PDF indexing completed. Result:")  
                self.conversation_area.append(f"<pre>{json.dumps(result, indent=2, ensure_ascii=False)}</pre>")  
                self.markdown_to_save += "✅ Local PDF indexing completed. Result: \n" + f"<pre>{json.dumps(result, indent=2, ensure_ascii=False)}</pre>\n\n"
                self.markdown_to_save += "Files:\n "
                self.conversation_area.append("Files::\n ")
                for value in self.uploaded_json_values:
                    name_file= value["pdf_name"]
                    self.conversation_area.append(f"> {name_file} \n\n")
                    self.markdown_to_save +=  f"> {name_file} \n\n"
                    

    
            # Svuota il batch per evitare invii doppi  
            self.uploaded_json_values.clear()  
        except Exception as e:  
            self.conversation_area.append(f"❌ Exception while processing PDFs locally: {str(e)}") 
        end_time = time.time()
        total_time = end_time - start_time
        self.conversation_area.append( f"Total time two process is: {total_time}")   
        self.markdown_to_save += f"Total time two process is: {total_time} \n\n ---\n\n"

    def _process_all_pdfs_and_index_handler(self):       
        try:  
            asyncio.get_running_loop()  
            asyncio.create_task(self.process_all_pdfs_and_index())  
        except RuntimeError:  
            # Se non c'è un event loop (caso start da thread PyQt), lo creiamo e lo eseguiamo in modo bloccante  
            asyncio.run(self.process_all_pdfs_and_index())  


    def ara_private_debt_analysis(self):
        self.analysisType= "ARA_private_debt"
        # Funzione per estrarre il numero dalla stringa
        def extract_number(group_name):
            match = re.search(r'_(\d+)$', group_name) # Cerca un underscore seguito da uno o più numeri alla fine della stringa
            if match:
                return int(match.group(1)) # Restituisce il numero come intero
            return float('inf') # Restituisce un valore grande se non trova il numero, per metterlo alla fine

        ARA_PRIVATE_DEBT_LIST = [
            self.all_questions[group_name]
            for group_name in sorted(self.all_questions.keys(), key=extract_number) # Usa la funzione extract_number come chiave
            if group_name.startswith("ARA_PRIVATE_DEBT_")
        ]




        self.conversation_area.append("🔄 I am initiating the analysis...")
        start_time = time.time()  # Registra il tempo di inizio
        chat_history= []
        input_tokens= 0
        output_tokens= 0
        for i in range(len(ARA_PRIVATE_DEBT_LIST)):
            data = {"chat_input":{              
                    "questions": ARA_PRIVATE_DEBT_LIST[i]
                    },
                    "text_input_pre": self.text_input_pre, #+ f"Answer in {self.language}.", # str, #system inizio # se vuota usa quello già caricato, altrimenti sovrascrive
                    "text_input_post":  self.text_input_post,# + f"Answer in {self.language}",           
                    "index_tags":{"chat_id": self.chat_id}, #{"chat_id":"dged53738" ,"user_id": "123fgdgd", etc} ### almeno un campo ci deve essere per forza altrimenti eccezione
                    "indexsearch_topn_chunk": self.indexsearch_topn_chunk,                 
                    "language": self.language,
                    "promptflow_mode": "questions",# questions= analysis, data_to_json_schema= extracteddata, single_question = query 
                    "chat_history": [],
                    "json_schema": "" # str
            }

            response = ask_question_with_token_context(data=data, url=API_ENDPOINT_PROMPT_FLOW, api_key=API_KEY_PROMPT_FLOW)

            save_response_json(response= response, filename_prefix=self.analysisType)



            chat_output = response.get("chat_output", "Errore: nessuna risposta valida ricevuta.")
            chat_output_json = json.loads(chat_output)

            # Accedi all'attributo choices
            content = chat_output_json['choices'][0]['message']['content']
            #self.print_to_output(content)  
            self.markdown_to_save += content + "\n\n---\n\n"   
            self.markdown_to_save_simil_real_generation += content + "\n\n---\n\n" # per la generazione simil realistica         
            
            input_tokens+= chat_output_json["usage"]["prompt_tokens"]
            output_tokens += chat_output_json["usage"]["completion_tokens"]
            #chat_output = response.get("chat_output", "Errore: nessuna risposta valida ricevuta.")

            print(ARA_PRIVATE_DEBT_LIST[i])
            time.sleep(1.5) 
   
        end_time = time.time()
        total_time = end_time - start_time

        price_inputs = input_tokens * (COST_MILLION_TOKEN_INPUT / 1_000_000)
        price_outputs = output_tokens * (COST_MILLION_TOKEN_OUTPUT / 1_000_000)
        total_price = price_inputs + price_outputs


        final_count= f"\n \n ⏱️ Analysis completed in {total_time:.2f} seconds, costing €{total_price:.4f}, using only the LLM input tokens {input_tokens} and output tokens {output_tokens}."
        #self.print_to_output(chat_output)
        #self.conversation_area.append(final_count
            #f"⏱️ Analisi completata in {total_time:.2f} secondi, costata {total_price:.4f} €, solamente l'uso del LLM"
            #f"token di input {input_tokens} e output {output_tokens}.")
        #    )
        
        self.markdown_to_save += final_count
        save_markdown_response(
            self.markdown_to_save_simil_real_generation,
            filename_prefix=self.analysisType
        )
        for i in range(len(ARA_PRIVATE_DEBT_LIST)):
            questions_for_this_api_call = " § ".join(q_dict["question"] for q_dict in ARA_PRIVATE_DEBT_LIST[i])
            self.markdown_to_save += f"\n\n API_CALL {i}:\n\n{questions_for_this_api_call}"
        self.print_to_output(self.markdown_to_save)
        
        
    def ARA_2024_private_equity_analysis(self):
        self.analysisType= "ARA_2024_private_equity"
                # Funzione per estrarre il numero dalla stringa
        def extract_number(group_name):
            match = re.search(r'_(\d+)$', group_name) # Cerca un underscore seguito da uno o più numeri alla fine della stringa
            if match:
                return int(match.group(1)) # Restituisce il numero come intero
            return float('inf') # Restituisce un valore grande se non trova il numero, per metterlo alla fine

        ARA_PRIVATE_EQUITY_LIST = [  
            self.all_questions[group_name]  
            for group_name in sorted(self.all_questions.keys(), key=extract_number)  
            if group_name.startswith("ARA_PRIVATE_EQUITY_") or group_name.startswith("ARA_2024_PRIVATE_EQUITY_")  
        ]  


        self.conversation_area.append("🔄 I am initiating the analysis...")
        start_time = time.time()  # Registra il tempo di inizio
        chat_history= []
        input_tokens= 0
        output_tokens= 0
        for i in range(len(ARA_PRIVATE_EQUITY_LIST)):
            data = {"chat_input":{              
                    "questions": ARA_PRIVATE_EQUITY_LIST[i]
                    },
                    "text_input_pre": self.text_input_pre,# + f"Answer in {self.language}.", # str, #system inizio # se vuota usa quello già caricato, altrimenti sovrascrive
                    "text_input_post": self.text_input_post,# + f"Answer in {self.language}",             
                    "index_tags":{"chat_id": self.chat_id}, #{"chat_id":"dged53738" ,"user_id": "123fgdgd", etc} ### almeno un campo ci deve essere per forza altrimenti eccezione
                    # qrm_client_id" = client (gruppo persone), 
                    #"qrm_user_id" l'utente singola persona, chat_id la chat aperta
                    "indexsearch_topn_chunk": self.indexsearch_topn_chunk,                 
                    "language": self.language,
                    "promptflow_mode": "questions",# questions= analysis, data_to_json_schema= extracteddata, single_question = query 
                    #"chat_input":{"user_input": "come sono bla bla "}
                    "chat_history": [],
                    "json_schema": "" # str
            }
            print(ARA_PRIVATE_EQUITY_LIST[i])
            response = ask_question_with_token_context(data=data, url=API_ENDPOINT_PROMPT_FLOW, api_key=API_KEY_PROMPT_FLOW)

            save_response_json(response= response, filename_prefix=self.analysisType)



            chat_output = response.get("chat_output", "Errore: nessuna risposta valida ricevuta.")
            chat_output_json = json.loads(chat_output)

            # Accedi all'attributo choices
            content = chat_output_json['choices'][0]['message']['content']
            #self.print_to_output(content)  
            self.markdown_to_save += content + "\n\n---\n\n"  
            self.markdown_to_save_simil_real_generation += content + "\n\n---\n\n" # per la generazione simil realistica          
            
            input_tokens+= chat_output_json["usage"]["prompt_tokens"]
            output_tokens += chat_output_json["usage"]["completion_tokens"]
            #chat_output = response.get("chat_output", "Errore: nessuna risposta valida ricevuta.")
        
            # Delay per evitare 429
            time.sleep(1.5)  

   
        end_time = time.time()
        total_time = end_time - start_time

        price_inputs = input_tokens * (COST_MILLION_TOKEN_INPUT / 1_000_000)
        price_outputs = output_tokens * (COST_MILLION_TOKEN_OUTPUT / 1_000_000)
        total_price = price_inputs + price_outputs
        final_count= f"\n \n ⏱️ Analysis completed in {total_time:.2f} seconds, costing €{total_price:.4f}, using only the LLM input tokens {input_tokens} and output tokens {output_tokens}."
        #self.print_to_output(chat_output)
        #self.conversation_area.append(final_count
            #f"⏱️ Analisi completata in {total_time:.2f} secondi, costata {total_price:.4f} €, solamente l'uso del LLM"
            #f"token di input {input_tokens} e output {output_tokens}.")
        #    )
        self.markdown_to_save += final_count

        save_markdown_response(
            self.markdown_to_save_simil_real_generation,
            filename_prefix=self.analysisType
        )
        for i in range(len(ARA_PRIVATE_EQUITY_LIST)):
            questions_for_this_api_call = " § ".join(q_dict["question"] for q_dict in ARA_PRIVATE_EQUITY_LIST[i])
            self.markdown_to_save += f"\n\n API_CALL {i}:\n\n{questions_for_this_api_call}"
        self.print_to_output(self.markdown_to_save)
                

    def ARA_smart_private_equity_analysis(self):
        self.analysisType= "ARA_smart_private_equity"
                # Funzione per estrarre il numero dalla stringa
        def extract_number(group_name):
            match = re.search(r'_(\d+)$', group_name) # Cerca un underscore seguito da uno o più numeri alla fine della stringa
            if match:
                return int(match.group(1)) # Restituisce il numero come intero
            return float('inf') # Restituisce un valore grande se non trova il numero, per metterlo alla fine

        ARA_PRIVATE_EQUITY_LIST = [  
            self.all_questions[group_name]  
            for group_name in sorted(self.all_questions.keys(), key=extract_number)  
            if group_name.startswith("ARA_PRIVATE_EQUITY_") or group_name.startswith("ARA_SMART_PRIVATE_EQUITY_")  
        ]  



        self.conversation_area.append("🔄 I am initiating the analysis...")
        start_time = time.time()  # Registra il tempo di inizio
        chat_history= []
        input_tokens= 0
        output_tokens= 0
        for i in range(len(ARA_PRIVATE_EQUITY_LIST)):
            data = {"chat_input":{              
                    "questions": ARA_PRIVATE_EQUITY_LIST[i]
                    },
                    "text_input_pre": self.text_input_pre,# + f"Answer in {self.language}.", # str, #system inizio # se vuota usa quello già caricato, altrimenti sovrascrive
                    "text_input_post": self.text_input_post,# + f"Answer in {self.language}",             
                    "index_tags":{"chat_id": self.chat_id}, #{"chat_id":"dged53738" ,"user_id": "123fgdgd", etc} ### almeno un campo ci deve essere per forza altrimenti eccezione
                    # qrm_client_id" = client (gruppo persone), 
                    #"qrm_user_id" l'utente singola persona, chat_id la chat aperta
                    "indexsearch_topn_chunk": self.indexsearch_topn_chunk,                 
                    "language": self.language,
                    "promptflow_mode": "questions",# questions= analysis, data_to_json_schema= extracteddata, single_question = query 
                    #"chat_input":{"user_input": "come sono bla bla "}
                    "chat_history": [],
                    "json_schema": "" # str
            }
            print(ARA_PRIVATE_EQUITY_LIST[i])
            response = ask_question_with_token_context(data=data, url=API_ENDPOINT_PROMPT_FLOW, api_key=API_KEY_PROMPT_FLOW)

            save_response_json(response= response, filename_prefix=self.analysisType)



            chat_output = response.get("chat_output", "Errore: nessuna risposta valida ricevuta.")
            chat_output_json = json.loads(chat_output)

            # Accedi all'attributo choices
            content = chat_output_json['choices'][0]['message']['content']
            #self.print_to_output(content)  
            self.markdown_to_save += content + "\n\n---\n\n"            
            self.markdown_to_save_simil_real_generation += content + "\n\n---\n\n" # per la generazione simil realistica

            input_tokens+= chat_output_json["usage"]["prompt_tokens"]
            output_tokens += chat_output_json["usage"]["completion_tokens"]
            #chat_output = response.get("chat_output", "Errore: nessuna risposta valida ricevuta.")
        
            # Delay per evitare 429
            time.sleep(1.5)  

   
        end_time = time.time()
        total_time = end_time - start_time

        price_inputs = input_tokens * (COST_MILLION_TOKEN_INPUT / 1_000_000)
        price_outputs = output_tokens * (COST_MILLION_TOKEN_OUTPUT / 1_000_000)
        total_price = price_inputs + price_outputs
        final_count= f"\n \n ⏱️ Analysis completed in {total_time:.2f} seconds, costing €{total_price:.4f}, using only the LLM input tokens {input_tokens} and output tokens {output_tokens}."
        #self.print_to_output(chat_output)
        #self.conversation_area.append(final_count
            #f"⏱️ Analisi completata in {total_time:.2f} secondi, costata {total_price:.4f} €, solamente l'uso del LLM"
            #f"token di input {input_tokens} e output {output_tokens}.")
        #    )
        self.markdown_to_save += final_count
        save_markdown_response(
            self.markdown_to_save_simil_real_generation,
            filename_prefix=self.analysisType
        )

        for i in range(len(ARA_PRIVATE_EQUITY_LIST)):
            questions_for_this_api_call =" § ".join(q_dict["question"]  for q_dict in ARA_PRIVATE_EQUITY_LIST[i])
            self.markdown_to_save += f"\n\n API_CALL {i}:\n\n{questions_for_this_api_call}"


        self.print_to_output(self.markdown_to_save)




    def ARA_smart_venture_capital(self):
        self.analysisType= "ARA_smart_venture_capital" 
        self.conversation_area.append("🔄 I am initiating the analysis......")
        
        # Funzione per estrarre il numero dalla stringa
        def extract_number(group_name):
            match = re.search(r'_(\d+)$', group_name) # Cerca un underscore seguito da uno o più numeri alla fine della stringa
            if match:
                return int(match.group(1)) # Restituisce il numero come intero
            return float('inf') # Restituisce un valore grande se non trova il numero, per metterlo alla fine

        ARA_SMART_VENTURE_CAPITAL_LIST = [
            self.all_questions[group_name]
            for group_name in sorted(self.all_questions.keys(), key=extract_number) # Usa la funzione extract_number come chiave
            if group_name.startswith("ARA_SMART_VENTURE_CAPITAL_")
        ]
        start_time = time.time()
        chat_history= []
        input_tokens= 0
        output_tokens= 0
        for i in range(len(ARA_SMART_VENTURE_CAPITAL_LIST)):
            data = {"chat_input":{              
                    "questions": ARA_SMART_VENTURE_CAPITAL_LIST[i]
                    },
                    "text_input_pre": self.text_input_pre,# + f"Answer in {self.language}.", # str, #system inizio # se vuota usa quello già caricato, altrimenti sovrascrive
                    "text_input_post": self.text_input_post,# + f"Answer in {self.language}",             
                    "index_tags":{"chat_id": self.chat_id}, #{"chat_id":"dged53738" ,"user_id": "123fgdgd", etc} ### almeno un campo ci deve essere per forza altrimenti eccezione
                    # qrm_client_id" = client (gruppo persone), 
                    #"qrm_user_id" l'utente singola persona, chat_id la chat aperta
                    "indexsearch_topn_chunk": self.indexsearch_topn_chunk,                 
                    "language": self.language,
                    "promptflow_mode": "questions",# questions= analysis, data_to_json_schema= extracteddata, single_question = query 
                    #"chat_input":{"user_input": "come sono bla bla "}
                    "chat_history": [],
                    "json_schema": "" # str
            }
            print(ARA_SMART_VENTURE_CAPITAL_LIST[i])
            response = ask_question_with_token_context(data=data, url=API_ENDPOINT_PROMPT_FLOW, api_key=API_KEY_PROMPT_FLOW)

            save_response_json(response= response, filename_prefix=self.analysisType)



            chat_output = response.get("chat_output", "Errore: nessuna risposta valida ricevuta.")
            chat_output_json = json.loads(chat_output)

            # Accedi all'attributo choices
            content = chat_output_json['choices'][0]['message']['content']
            #self.print_to_output(content)  
            self.markdown_to_save += content + "\n\n---\n\n"            
            self.markdown_to_save_simil_real_generation += content + "\n\n---\n\n" # per la generazione simil realistica

            input_tokens+= chat_output_json["usage"]["prompt_tokens"]
            output_tokens += chat_output_json["usage"]["completion_tokens"]
            #chat_output = response.get("chat_output", "Errore: nessuna risposta valida ricevuta.")
        
            # Delay per evitare 429
            time.sleep(1.5)  

      
        end_time = time.time()
        total_time = end_time - start_time

        price_inputs = input_tokens * (COST_MILLION_TOKEN_INPUT / 1_000_000)
        price_outputs = output_tokens * (COST_MILLION_TOKEN_OUTPUT / 1_000_000)
        total_price = price_inputs + price_outputs
        final_count= f"\n \n ⏱️ Analysis completed in {total_time:.2f} seconds, costing €{total_price:.4f}, using only the LLM input tokens {input_tokens} and output tokens {output_tokens}."
        #self.print_to_output(chat_output)
        #self.conversation_area.append(final_count
            #f"⏱️ Analisi completata in {total_time:.2f} secondi, costata {total_price:.4f} €, solamente l'uso del LLM"
            #f"token di input {input_tokens} e output {output_tokens}.")
        #    )
        self.markdown_to_save += final_count
        save_markdown_response(
            self.markdown_to_save_simil_real_generation,
            filename_prefix=self.analysisType
        )

        for i in range(len(ARA_SMART_VENTURE_CAPITAL_LIST)):
            questions_for_this_api_call = " § ".join(q_dict["question"] for q_dict in ARA_SMART_VENTURE_CAPITAL_LIST[i])
            self.markdown_to_save += f"\n\n API_CALL {i}:\n\n{questions_for_this_api_call}"



        self.print_to_output(self.markdown_to_save)
         
        
       


    def BPM_analysis(self):
        self.analysisType= "BPM_analysis" 
        
        # Funzione per estrarre il numero dalla stringa
        def extract_number(group_name):
            match = re.search(r'_(\d+)$', group_name) # Cerca un underscore seguito da uno o più numeri alla fine della stringa
            if match:
                return int(match.group(1)) # Restituisce il numero come intero
            return float('inf') # Restituisce un valore grande se non trova il numero, per metterlo alla fine


        BPM_QUESTIONS_LIST = [
        self.all_questions[group_name]
        for group_name in sorted(self.all_questions.keys(), key=extract_number) # Usa la funzione extract_number come chiave
        if group_name.startswith("BPM_QUESTIONS_")
        ]
    
        self.conversation_area.append("🔄 I am initiating the analysis...")
        start_time = time.time()
        chat_history= []
        input_tokens= 0
        output_tokens= 0
        for i in range(len(BPM_QUESTIONS_LIST)):
            data = {"chat_input":{              
                    "questions": BPM_QUESTIONS_LIST[i]
                    },
                    "text_input_pre": self.text_input_pre,# + f"Answer in {self.language}.", # str, #system inizio # se vuota usa quello già caricato, altrimenti sovrascrive
                    "text_input_post": self.text_input_post,# + f"Answer in {self.language}.",                   
                    "index_tags":{"chat_id": self.chat_id}, #{"chat_id":"dged53738" ,"user_id": "123fgdgd", etc} ### almeno un campo ci deve essere per forza altrimenti eccezione
                    # qrm_client_id" = client (gruppo persone), 
                    #"qrm_user_id" l'utente singola persona, chat_id la chat aperta
                    "indexsearch_topn_chunk": self.indexsearch_topn_chunk,                 
                    "language": self.language,
                    "promptflow_mode": "questions",# questions= analysis, data_to_json_schema= extracteddata, single_question = query 
                    #"chat_input":{"user_input": "come sono bla bla "}
                    "chat_history": [],
                    "json_schema": "" # str
            }

            response = ask_question_with_token_context(data=data, url=API_ENDPOINT_PROMPT_FLOW, api_key=API_KEY_PROMPT_FLOW)

            save_response_json(response= response, filename_prefix=self.analysisType)



            chat_output = response.get("chat_output", "Errore: nessuna risposta valida ricevuta.")
            chat_output_json = json.loads(chat_output)

            # Accedi all'attributo choices
            content = chat_output_json['choices'][0]['message']['content']
            self.markdown_to_save += content + "\n\n---\n\n"
            self.markdown_to_save_simil_real_generation += content + "\n\n---\n\n" # per la generazione simil realistica
             
            
            
            input_tokens+= chat_output_json["usage"]["prompt_tokens"]
            output_tokens += chat_output_json["usage"]["completion_tokens"]

        
            # Delay per evitare 429
            time.sleep(1.5) 
        
        end_time = time.time()
        total_time = end_time - start_time



        input_tokens = chat_output_json["usage"]["prompt_tokens"]
        output_tokens = chat_output_json["usage"]["completion_tokens"]
        price_inputs = input_tokens * (COST_MILLION_TOKEN_INPUT / 1_000_000)
        price_outputs = output_tokens * (COST_MILLION_TOKEN_OUTPUT / 1_000_000)
        total_price = price_inputs + price_outputs
        final_count= f"\n \n ⏱️ Analysis completed in {total_time:.2f} seconds, costing €{total_price:.4f}, using only the LLM input tokens {input_tokens} and output tokens {output_tokens}."

        #self.print_to_output(chat_output)
        #self.conversation_area.append(final_count
            #f"⏱️ Analisi completata in {total_time:.2f} secondi, costata {total_price:.4f} €, solamente l'uso del LLM"
            #f"token di input {input_tokens} e output {output_tokens}.")
        #    )
        self.markdown_to_save += final_count
        save_markdown_response(
            self.markdown_to_save_simil_real_generation,
            filename_prefix=self.analysisType
        )

        for i in range(len(BPM_QUESTIONS_LIST)):
            questions_for_this_api_call = " § ".join(q_dict["question"] for q_dict in BPM_QUESTIONS_LIST[i])
            self.markdown_to_save += f"\n\n API_CALL {i}:\n\n{questions_for_this_api_call}"




        self.print_to_output(self.markdown_to_save)
        
 



    def convert_markdown_table_to_html(self, markdown_text):  
        table_pattern = r"((\|.*\|(\n|$))+)"  
        matches = list(re.finditer(table_pattern, markdown_text))  
        for match in matches:  
            md_table = match.group(0).strip().split('\n')  
            if len(md_table) < 2:  
                print("DEBUG: Table too short:", md_table)  
                continue  
            sep_line = md_table[1].strip()  
            if len(sep_line) > 100:  
                print("DEBUG: Riga separatore troppo lunga ({})".format(len(sep_line)))  
                continue  
            if "|" not in sep_line or "-" not in sep_line:  
                print("DEBUG: Riga separatore NON valida:", repr(sep_line))  
                continue  
            
            # Qui presumi che la prima riga sia sempre intestazione, la seconda il separatore.  
            headers = [html.escape(cell.strip()) for cell in md_table[0].strip('|').split('|')]  
            alignment_line = md_table[1].strip('|').split('|')  
            start_row = 2  
            alignments = []  
            for align in alignment_line:  
                align = align.strip()  
                if align.startswith(':') and align.endswith(':'):  
                    alignments.append("center")  
                elif align.endswith(':'):  
                    alignments.append("right")  
                elif align.startswith(':'):  
                    alignments.append("left")  
                else:  
                    alignments.append("center")  
            def format_markdown(cell):  
                cell = html.escape(cell)  
                cell = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", cell)  
                cell = re.sub(r"\*(.*?)\*", r"<em>\1</em>", cell)  
                return cell  
            rows = []  
            for line in md_table[start_row:]:  
                row = [format_markdown(cell.strip()) for cell in line.strip('|').split('|')]  
                while len(row) < len(headers):  
                    row.append("")  
                rows.append(row)  
            html_table = '<table border="1" cellspacing="0" cellpadding="4" style="border-collapse:collapse;text-align:center;">'  
            html_table += "<thead><tr>" + "".join(f"<th style='text-align:{align};'>{h}</th>" for h, align in zip(headers, alignments)) + "</tr></thead><tbody>"  
            for row in rows:  
                html_table += "<tr>" + "".join(f"<td style='text-align:{align};'>{cell}</td>" for cell, align in zip(row, alignments)) + "</tr>"  
            html_table += "</tbody></table>"  
            markdown_text = markdown_text.replace(match.group(0), html_table)  
        return markdown_text  


    def data_extraction(self):
        self.analysisType== "data_extraction"
        self.conversation_area.append("🔄 starting data extraction...")
        start_time = time.time()


        data = {"chat_input":{  },
                "text_input_pre": "", # Se fosse diverso da "" bisogna aggiungere + f"Answer in {self.language}.", # str, #system inizio # se vuota usa quello già caricato, altrimenti sovrascrive
                "text_input_post":"", #str system fine
                "index_tags":{"chat_id": self.chat_id},                   
                "user_informations":[],#[{"key": "qrm_client_id","value":"1234"},{"key": "qrm_user_id","value":"1234"}] 
                # qrm_client_id" = client (gruppo persone), 
                #"qrm_user_id" l'utente singola persona, chat_id la chat aperta
                "indexsearch_topn_chunk": self.indexsearch_topn_chunk,               
                "language": self.language,
                "promptflow_mode": "data_to_json_schema",# questions= analysis, data_to_json_schema= extracteddata, single_question = query 
                #"chat_input":{"user_input": "come sono bla bla "}
                "chat_history": [],
                "json_schema": self.json_schema #json_schema_extractedData # json_schema_Company_income_statement #json_schema_easy, json_schema_d
        }

        #response = ask_question_with_token_context(data=data, url=API_ENDPOINT_PROMPT_FLOW, api_key=API_KEY_PROMPT_FLOW)
        #input_tokens = response.get("input_tokens", 0)
        #output_tokens = response.get("output_tokens", 0)
        #chat_output = response.get("chat_output", "Errore: nessuna risposta valida ricevuta.")


        response = ask_question_with_token_context(data=data, url=API_ENDPOINT_PROMPT_FLOW, api_key=API_KEY_PROMPT_FLOW)       

        
        chat_output = response.get("chat_output", "Errore: nessuna risposta valida ricevuta.")
        chat_output_json = json.loads(chat_output)

        save_response_json(response= chat_output_json, filename_prefix=self.analysisType)
        # Accedi all'attributo choices
        #content = chat_output_json['choices'][0]['message']['content']
        #self.print_to_output(content)  

        # CHIAMA LA NUOVA FUNZIONE PER GENERARE LA TABELLA HTML
        table_html = self.json_to_html_table(chat_output_json)
        self.conversation_area.append(table_html) # 

        end_time = time.time()
        total_time = end_time - start_time



        input_tokens = chat_output_json["usage"]["prompt_tokens"]
        output_tokens = chat_output_json["usage"]["completion_tokens"]
        price_inputs = input_tokens * (COST_MILLION_TOKEN_INPUT / 1_000_000)
        price_outputs = output_tokens * (COST_MILLION_TOKEN_OUTPUT / 1_000_000)
        total_price = price_inputs + price_outputs

        #self.print_to_output(chat_output)
        self.conversation_area.append(
            #f"⏱️ Analisi completata in {total_time:.2f} secondi, costata {total_price:.4f} €, solamente l'uso del LLM"
            #f"token di input {input_tokens} e output {output_tokens}.")
            f"⏱️ Extraction completed in {total_time:.2f} seconds, costing €{total_price:.4f}, using only the LLM input tokens {input_tokens} and output tokens {output_tokens}.")


  
    def clean_separators(self, text):  
        """  
        Sostituisce tutte le occorrenze di righe composte da 20, 21 o 22 tilde consecutive con un a capo.  
        """  
        # ^ e $ ancorano l'inizio/fine riga; [~]{20,22} indica 20-22 tilde  
        return re.sub(r'^[~]{20,22}$', '\n', text, flags=re.MULTILINE) 

    def json_to_html_table(self, chat_output_json: dict) -> str:
        """
        Converte il JSON di output del modello in una tabella HTML,
        usando self.json_schema per determinare le colonne.

        Args:
            chat_output_json (dict): L'intero JSON di output ricevuto dal modello.

        Returns:
            str: La tabella HTML generata.
        """
        try:
            # Naviga nel JSON per trovare il contenuto desiderato
            # Il contenuto "reale" è una stringa JSON dentro "content"
            content_str = chat_output_json['choices'][0]['message']['content']

            # Effettua un secondo parsing per ottenere l'oggetto JSON dal content_str
            parsed_content = json.loads(content_str)

        except (KeyError, json.JSONDecodeError) as e:
            return f"<p>⚠️ Errore nel parsing del JSON di input: {html.escape(str(e))}</p>"
        
        json_schema_dict= json.loads(self.json_schema)

        # Determina la chiave principale dal json_schema_dict
        main_key = None
        if "properties" in json_schema_dict:
            for key, value in json_schema_dict["properties"].items():
                if value.get("type") == "array" and "items" in value:
                    main_key = key
                    break
                
        if not main_key:
            return "<p>⚠️ Chiave principale dell'array non trovata in json_schema_dict.</p>"

        data_list = parsed_content.get(main_key, [])

        if not data_list:
            return "<p>Nessun dato trovato per la chiave principale: {html.escape(main_key)}.</p>"

        # Determina l'ordine e i nomi delle colonne dallo schema
        field_order = []
        if main_key in json_schema_dict.get("properties", {}):
            items_schema = json_schema_dict["properties"][main_key].get("items", {})

            # Caso con $ref (come nel tuo esempio)
            if "$ref" in items_schema:
                def_name = items_schema["$ref"].split("/")[-1]
                if def_name in json_schema_dict.get("$defs", {}):
                    field_order = list(json_schema_dict["$defs"][def_name].get("properties", {}).keys())

            # Caso senza $ref, proprietà definite direttamente sotto "items"
            elif "properties" in items_schema:
                field_order = list(items_schema["properties"].keys())

        if not field_order:
            return "<p>⚠️ Impossibile determinare l'ordine delle colonne da json_schema_dict.</p>"

        # Costruisci la tabella HTML
        html_table = '<table border="1" cellspacing="0" cellpadding="4" style="border-collapse:collapse;text-align:center;">'
        html_table += "<thead><tr>" + "".join(f"<th>{html.escape(field)}</th>" for field in field_order) + "</tr></thead><tbody>"

        for item in data_list:
            html_table += "<tr>"
            for field in field_order:
                value = item.get(field, "")
                html_table += f"<td>{html.escape(str(value))}</td>"
            html_table += "</tr>"

        html_table += "</tbody></table>"
        return html_table


######
###### vecchia stampa funzionava senza striscia di ######################## tra le tabelle di riassunto 
######
#    def print_to_output(self, message):
#        # Converti l'intero testo in HTML con markdown2
#        message = markdown2.markdown(message, extras=["fenced-code-blocks", "tables"])
#
#        # Converti solo le tabelle presenti nel messaggio HTML
#        message = self.convert_markdown_table_to_html(message)
#
#        # Aggiungi il risultato nell'area di conversazione
#        self.conversation_area.append(message)
#


    def find_and_replace_summary_tables(self, text, convert_table_func):  
        """  
        Replaces all summary tables (delimited by 24 '#' lines) with HTML.  
        """  
        pattern = r"^########################\n([\s\S]+?)^########################$"  
        matches = list(re.finditer(pattern, text, re.MULTILINE))  
        for m in reversed(matches):  
            markdown_table = m.group(1).strip()  
            html_table = convert_table_func(markdown_table)  
            start, end = m.span()  
            text = text[:start] + html_table + text[end:]  
        return text  
    
    def find_and_replace_other_markdown_tables(self, text, json_to_html_table_func, context_json=None):  
        """  
        Finds all markdown tables not inside summary (##########) blocks  
        and replaces them with HTML via json_to_html_table.  
        context_json is an optional mapping from markdown (or positional index) to the JSON structure for the related table.  
        """  
        # Remove summary tables temporarily and mark their positions  
        summary_pattern = r"^########################\n[\s\S]+?^########################$"  
        summary_markers = []  
    
        def summary_replacer(match):  
            idx = len(summary_markers)  
            summary_markers.append(match.group(0))  
            return f"[[SUMMARY_MARKER_{idx}]]"  
    
        temp_text = re.sub(summary_pattern, summary_replacer, text, flags=re.MULTILINE)  
    
        # Pattern to find markdown tables not inside summary blocks  
        table_pattern = re.compile(r"((?:^\|.*\n)+)", re.MULTILINE)  
        matches = list(table_pattern.finditer(temp_text))  
    
        for idx, m in enumerate(reversed(matches)):  
            md_table = m.group(1)  
            # Attempt to retrieve the corresponding JSON payload for this table  
            json_table = None  
            if context_json is not None:  
                # Here you might use the content as key, or position idx, or another logic  
                json_table = context_json.get(md_table) or context_json.get(len(matches) - 1 - idx)  
            if json_table:  
                html_table = json_to_html_table_func(json_table)  
                start, end = m.span()  
                temp_text = temp_text[:start] + html_table + temp_text[end:]  
    
        # Restore summary tables  
        for idx, original in enumerate(summary_markers):  
            temp_text = temp_text.replace(f"[[SUMMARY_MARKER_{idx}]]", original)  
    
        return temp_text  
    
    # Example integration in your print_to_output method  
    
    def print_to_output(self, message, json_tables_context=None):  
        """  
        Prints the assistant's message to the conversation area:  
          - Summary tables (delimited by 24 #) are rendered using convert_markdown_table_to_html  
          - All other Markdown tables are rendered using json_to_html_table (if context JSON is provided)  
          - The rest is rendered as Markdown/HTML  
        Args:  
            message: str, the markdown message  
            json_tables_context: dict, mapping table index (int) to the JSON data expected for that table  
        """  
        # Step 0: Replase 
        message = self.clean_separators(message)  
        # Step 1: Replace summary tables (delimited by 24 #) with HTML tables  
        msg_stage1 = self.find_and_replace_summary_tables(  
            message, self.convert_markdown_table_to_html  
        )  
    
        # Step 2: Replace other markdown tables (not summary) with json_to_html_table (if JSON is available)  
        msg_stage2 = self.find_and_replace_other_markdown_tables(  
            msg_stage1, self.json_to_html_table, context_json=json_tables_context  
        )  
    
        # Step 3: Convert the full message (possibly mixed HTML/markdown) to HTML  
        msg_html = markdown2.markdown(  
            msg_stage2, extras=["fenced-code-blocks", "tables"]  
        )  

        if self.analysisType== "ARA_smart_venture_capital" or self.analysisType== "BPM_analysis"  or self.analysisType== "ARA_2024_private_equity" or self.analysisType=="ARA_smart_private_equity"or self.analysisType== "data_extraction"or self.analysisType== "ARA_private_debt":
            print(save_window_to_html(self.conversation_area, msg_html, filename_prefix=self.analysisType, name_client= self.client_id, language=self.language) ) 
            self.analysisType = ""
     

        # Step 4: Append the result to the conversation area  
        self.conversation_area.append(msg_html)  










if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AIFrontEnd()

    window.show()
    sys.exit(app.exec())




