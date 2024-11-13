from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_huggingface import HuggingFaceEmbeddings
import vertexai
import os
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from fastapi.middleware.cors import CORSMiddleware

# Instancia de FastAPI
app2 = FastAPI()


# Habilitar CORS
app2.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite solicitudes de todos los dominios. Cambia "*" si deseas restringirlo.
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.).
    allow_headers=["*"],  # Permite todas las cabeceras.
)



##########################################################################
######    CONFIG LLM GEMINI 1.5 PRO  #####################################
##########################################################################

vertexai.init(
    project="gen-lang-client-0730997933",
    location="us-central1"
)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gen-lang-client-0730997933-9e99e8511560.json'

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

llm = ChatVertexAI(model="gemini-1.5-pro-002")

##########################################################################

##########################################################################
################             Prompts             #########################
##########################################################################

########  1. Prompt Agent 1  #############################################
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ID para guardar el historial de consultas anteriores
MEMORY_KEY = "chat_history"

chat_history = []

# Template configurado para el prompt
prompt_agent_1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """System Prompt:

                You are a data processing agent. Your task is to:

                1. Open an Excel file with path provided by the user.
                2. Extract all tables from the Excel sheets on a dictionary.
                3. Interpretate the dictionary and create a sql script to create these tables on database. To interpretate first get name of columns and then get the type of variable.
                4. Finally give the sql script as response.
            """,
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
##########################################################################
########  2. Prompt Agent 2  #############################################
prompt_agent_2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un experto en bases de datos, lectura de script sql y programador html. Tu tarea es a traves de un script poder entender la estructura de las tablas y crear formularios en html a partir de estos",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
##########################################################################

##########################################################################
################              TOOLS              #########################
##########################################################################
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.tools import tool
from typing import List, Type, Dict
from langchain_community.document_loaders import UnstructuredExcelLoader
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage

########  1. Tools Agent 1  #############################################
@tool
def get_tables(file_path: str) -> Dict[str, List[Dict[str, str]]]:
    """Extrae tablas desde un archivo Excel, guarda cada tabla en un CSV y devuelve un diccionario con columnas y tipos de datos."""
    # Cargar todas las hojas del archivo Excel
    xls = pd.ExcelFile(file_path)
    
    tables_info = {}

    # Recorrer cada hoja, guardarla como CSV y extraer información sobre columnas y tipos de datos
    for sheet_name in xls.sheet_names:
        # Convertir la hoja a DataFrame
        df = xls.parse(sheet_name)
        
        # Guardar el DataFrame como CSV
        csv_file_name = f"{sheet_name}.csv"
        df.to_csv(csv_file_name, index=False)
        
        # Extraer información de las columnas y tipos de datos
        columns_info = [
            {"name": column, "dtype": str(dtype)}
            for column, dtype in df.dtypes.items()
        ]
        
        # Guardar esta información en el diccionario
        tables_info[sheet_name] = columns_info
    
    return tables_info

tools = [get_tables]

llm_with_tools = llm.bind_tools(tools)
##########################################################################

########  2. Tools Agent 2  #############################################
@tool
def get_sql(url: str) -> str:
    """Get script from input file."""
    loader = TextLoader(url)
    sql = loader.load()
    sql_content = sql[0].page_content 

    return str(sql_content)

@tool
def get_data(url: str) -> str:
    """Get data example"""
    loader = CSVLoader(file_path=url)
    data = loader.load()
    data_content = data[0].page_content 

    return data_content

tools_2 = [get_sql, get_data]

llm_with_tools_2 = llm.bind_tools(tools_2)
##########################################################################

##########################################################################
################             Agents              #########################
##########################################################################
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentType
from langchain.agents import AgentExecutor

########    1. Agent 1       #############################################
agent_1 = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt_agent_1
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
##########################################################################

########    2. Agent 2       #############################################
agent_2 = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt_agent_2
    | llm_with_tools_2
    | OpenAIToolsAgentOutputParser()
)
##########################################################################

### Executors ############################################################
agent_executor = AgentExecutor(agent=agent_1, tools = tools, verbose=True)
agent_executor2 = AgentExecutor(agent=agent_2, tools = tools_2, verbose=True)
##########################################################################

### Guardar historial  ################################################### 
def save_history(result, input):
    chat_history.extend(
        [
            HumanMessage(content=input),
            AIMessage(content=result["output"]),
        ]
    )
##########################################################################

### Save files function  #################################################
def save_script(result, file_name):
    """save files with input data and name file."""
    
    with open(file_name, 'w') as file:
        file.write(result)
##########################################################################

# Modelo para la entrada del API
class QueryRequest(BaseModel):
    question: str


# Ruta principal para hacer consultas en lenguaje natural
@app2.post("/get_sql_query/")
async def run_query(request: QueryRequest):
    input_prompt = f"Genera un script sql desde el archivo ubicado en '{request.question}', motor de base de datos postgresql"
    result = agent_executor.invoke({"input": input_prompt, "chat_history": chat_history })
    save_script(result.get('output'), 'output.sql')
    save_history(result, input_prompt)
    return {"result": result}

@app2.post("/sql_to_html/")
async def run_query2(request: QueryRequest):
    input_prompt2 = f"Genera un html a partir del archivo '{request.question}', dame ademas botones funcionales con javascript"
    result = agent_executor2.invoke({"input": input_prompt2, "chat_history": chat_history })
    save_history(result, input_prompt2)
    save_script(result.get('output'), 'forms.html')
    print(result)
    return {"html": result.get('output')}

# Levantar servidor de FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app2, host="0.0.0.0", port=8001)