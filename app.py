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
app = FastAPI()


# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite solicitudes de todos los dominios. Cambia "*" si deseas restringirlo.
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.).
    allow_headers=["*"],  # Permite todas las cabeceras.
)


vertexai.init(
    project="gen-lang-client-0730997933",
    location="us-central1"
)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gen-lang-client-0730997933-9e99e8511560.json'

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

llm = ChatVertexAI(model="gemini-1.5-pro-002")




db = SQLDatabase.from_uri("sqlite:///Chinook.db")


# Configurar modelo de embeddings de Hugging Face
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

from langchain_community.agent_toolkits import create_sql_agent

agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

# Modelo para la entrada del API
class QueryRequest(BaseModel):
    question: str

# Ruta principal para hacer consultas en lenguaje natural
@app.post("/query/")
async def run_query(request: QueryRequest):
    result = agent_executor.invoke({"input": request.question})
    return {"result": result}

# Levantar servidor de FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)