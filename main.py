from typing import Union

from fastapi import FastAPI,Request,File, UploadFile,Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
import speech_recognition as sr 
from pydub import AudioSegment
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.vectorstores import Qdrant
import os
import pinecone
from typing import Union

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ["COHERE_API_KEY"] = "hv5YTaV6oUo5T9LGOY8F4bBtalGflhTU2FdPtEk3"

@app.get("/")
def read_root():
    return {"Hello": "World"}


""" @app.post("/data")
async def get_data(request: Request):
    # Retrieve data from the frontend
    data = await  request.json()
    
    # Process the data or perform any desired operations
    print(data)
    rd = "recieved"
    # Return a response
    return rd """

#Voice
@app.post("/upload-voice")
async def upload_image(data: UploadFile = File(...)):

    return {"results": "hello"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    sound = file
    print(file)
    with open(sound,"rb"):
        sound.read()
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}


# load our document here 
from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path="./gtbank-data-web.csv", csv_args={'delimiter': ','})

data = loader.load()

# split texts into chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1200,
    chunk_overlap  = 200,
    length_function = len,
)

data_chunks = text_splitter.split_documents(data)

# initialize embeddings 
embeddings = CohereEmbeddings(model = "multilingual-22-12")

# vector storage

'''
    There is an instance we can create an online vector storage using pincone but api restrictions so cant move forward with that.
'''

'''index_name = "faqs-dbase"

docsearch = Pinecone.from_documents(data_chunks, embeddings, index_name=index_name)'''

# local vector storage
db = Qdrant.from_documents(data_chunks, embeddings, location=":memory:", collection_name="my_documents", distance_func="Cosine")


# make our prompt 
prompt_template = """

generate response to the question based on the text provided.

Change instances where the bank is HDFC Bank to GTbank, 

If the text doesn't contain the answer, reply that the answer is not available and can request for more assistance by contacting us by telephone or sending a mail to customer service representative.

the Telephone Numbers:Tel: (+233) 302 611 560 Toll free: 0800 124 000 and the mail is gh.customersupport@gtbank.com

Text: {context}

Question: {question}
"""


PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}


# This function takes the prompt as a parameter and returns the answer based on our documents on our vector storage
def question_and_answer(question):
    qa = RetrievalQA.from_chain_type(llm=Cohere(model="command-nightly", temperature=0), 
                                 chain_type="stuff", 
                                 retriever=db.as_retriever(search_type="mmr"), 
                                 chain_type_kwargs=chain_type_kwargs, 
                                 return_source_documents=True)
                                 

    answer = qa({"query": question})
   
    return answer['result']

#print(question_and_answer("Hi"))


@app.post("/data")
async def get_data(request: Request):
    # Retrieve data from the frontend
    data = await request.json()
    
    # Process the data or perform any desired operations
    print(data['body'])
    rd = "recieved"
    # Return a response
    return rd

@app.post("/chat")
async def get_data(request: Request):
    # Retrieve data from the frontend
    data = await request.json()
    chatMsg= data['body']
    print(chatMsg)
    qa = RetrievalQA.from_chain_type(llm=Cohere(model="command-nightly", temperature=0), 
                                 chain_type="stuff", 
                                 retriever=db.as_retriever(search_type="mmr"), 
                                 chain_type_kwargs=chain_type_kwargs, 
                                 return_source_documents=True)
                                 

    answer = qa({"query": chatMsg})

    #return answer['result']
    # Process the data or perform any desired operations
    #print(data['body'])
    rd = "recieved"
    # Return a response
    return answer['result']