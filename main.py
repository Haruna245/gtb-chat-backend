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
import io
import pinecone
from typing import Union
from langchain.document_loaders.csv_loader import CSVLoader

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ["COHERE_API_KEY"] = "hv5YTaV6oUo5T9LGOY8F4bBtalGflhTU2FdPtEk3"
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




loader = CSVLoader(file_path="./BankFAQs.csv", encoding='utf8')

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
db = Qdrant.from_documents(data_chunks, embeddings, location=":memory:", collection_name="my_documents", distance_func="Dot")


# make our prompt 
# our prompt can be fined tuned as well, finding a way around that
prompt_template = """Text: {context}

Question: {question}

Answer the question based on the text provided.Change instances where the bank is HDFC Bank to GTbank If the text doesn't contain the answer, reply that the answer is not available."""


PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}


# This function takes the prompt as a parameter and returns the answer based on our documents on our vector storage
def question_and_answer(question):
    qa = RetrievalQA.from_chain_type(llm=Cohere(model="command-nightly", temperature=0), 
                                 chain_type="stuff", 
                                 retriever=db.as_retriever(), 
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
                                 retriever=db.as_retriever(), 
                                 chain_type_kwargs=chain_type_kwargs, 
                                 return_source_documents=True)
                                 

    answer = qa({"query": chatMsg})

    #return answer['result']
    # Process the data or perform any desired operations
    #print(data['body'])
    rd = "recieved"
    # Return a response
    return answer['result']