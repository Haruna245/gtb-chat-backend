import os
import io

from typing import Union

from fastapi import FastAPI,Request,File, UploadFile,Form
from fastapi.middleware.cors import CORSMiddleware

import speech_recognition as sr 
from pydub import AudioSegment

from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant

from qdrant_client import QdrantClient

from typing import Annotated
from typing import Union


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ["COHERE_API_KEY"] = "P5qlLVKqvPGIixGiGVmrKe1yXEQIbrcFoNMJn5ax"

client = QdrantClient(
   "https://5f76228b-aa4a-4bd9-887a-6cae33a53bfc.us-east4-0.gcp.cloud.qdrant.io:6333", 
api_key="1FKRDKZlC7W8BcMeayUaxE3nhnMPkcmTB2jGzWcjNKH-_e_9iP4ozQ"
)

# initialize embeddings 
embeddings = CohereEmbeddings(model = "multilingual-22-12")

doc_store = Qdrant(
    client=client, collection_name="my_documents", 
    embeddings=embeddings,
)

r = sr.Recognizer()


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
    mf = io.BytesIO(file)
    print(type(mf))
    #audio = AudioSegment.from_file(mf)
    #audio = AudioSegment.from_bytes()
    return {"file_size": len(file)}


""" @app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    print(type(file))
    #myfile = file.read()
    #audio = AudioSegment.from_file(file)
    return {"filename": file.filename}
 """

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    print(type(file))
    myfile = await file.read()
    
    #contents = base64.b64decode(myfile) 
    #print(contents)
    """ with open(myfile, 'rb') as f:
        contents = f.read()
    audio = AudioSegment.from_file(contents) """
    audio_segment = AudioSegment.from_file(io.BytesIO(myfile), format="m4a")
    output_filename = "output.wav"
    audio_segment.export(output_filename, format="wav")
    # open the file
    with sr.AudioFile(output_filename) as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        text = r.recognize_google(audio_data)
        print(text)
    #myfile = file.read()
    #audio = AudioSegment.from_file(contents)
    #return {"filename": file.filename}
    return text



# make our prompt 
prompt_template = """

generate response to the question based on the text provided.

If the text doesn't contain the answer, reply that the answer is not available and can request for more assistance by contacting us by telephone or sending a mail to customer service representative.

Text: {context}

Question: {question}
"""


PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}


PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}


# This function takes the prompt as a parameter and returns the answer based on our documents on our vector storage
def question_and_answer(question):
    qa = RetrievalQA.from_chain_type(llm=Cohere(model="command-nightly", temperature=0), 
                                 chain_type="stuff", 
                                 retriever = doc_store.as_retriever(search_type="mmr"), 
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
                                 retriever=doc_store.as_retriever(search_type="mmr"), 
                                 chain_type_kwargs=chain_type_kwargs, 
                                 return_source_documents=True)
                                 

    answer = qa({"query": chatMsg})

    #return answer['result']
    # Process the data or perform any desired operations
    #print(data['body'])
    rd = "recieved"
    # Return a response
    return answer['result']