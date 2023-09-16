from gradio_client import Client
from pydantic import BaseModel
import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse

from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from pydantic import BaseModel

class Item(BaseModel):
    q: str

app = FastAPI()
@app.get("/")
async  def getResult(item: Item):
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    llm = OpenAI(openai_api_key="nv-v88q46u4Ha4Q6Qrb1bPRN0V4x0SSOvL3Ue3CvK9Wi8PqG8QM",
                 openai_api_base='https://api.nova-oss.com/v1')
    loader = CSVLoader(
        file_path="/data/excel_file_example.csv",
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": ["P", "Q", "M"],
        },
    )

    data = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    red_alert_texts = text_splitter.split_documents(data)

    red_alert_db = Chroma.from_documents(red_alert_texts, embeddings)
    red_alert_retriever = red_alert_db.as_retriever()
    red_alert_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=red_alert_retriever)

    query = item.q+'"?'
    red_alert_qa.run(query)

    return {red_alert_qa}


