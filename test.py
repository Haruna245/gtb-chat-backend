import base64
from typing import Union

from fastapi import FastAPI,Request,File, UploadFile,Form,HTTPException,Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated
import io
from pydub import AudioSegment
import requests
import speech_recognition as sr

from sqlalchemy.orm import Session

import crud, models, schemas
from database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

r = sr.Recognizer()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def read_root():
    return {"Hello": "World"}



@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return crud.create_user(db=db, user=user)

@app.get("/users/{user_id}", response_model=schemas.User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@app.post("/users/login", response_model=schemas.User)
async def read_user(request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    print(data)
    email= data['email']
    password = data['password']
    password1 = password + "notreallyhashed"
    db_user = crud.get_user_login(db, user_email=email,user_password=password1)
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@app.post("/users/items/", response_model=schemas.Item)
def create_item_for_user(
     item: schemas.ItemCreate, db: Session = Depends(get_db)
):
    return crud.create_user_item(db=db, item=item,)

@app.get("/items/", response_model=list[schemas.Item])
def read_items(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    items = crud.get_items(db, skip=skip, limit=limit)
    return items

@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    #sound = file
    mf = io.BytesIO(file)
    print(type(mf))
    #audio = AudioSegment.from_file(mf)
    #audio = AudioSegment.from_bytes()
    return {"file_size": len(file)}


## I am using this
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



@app.post("/upload-voice")
async def upload_image(data: UploadFile = File(...)):
    
    return {"results": "hello"}


@app.post("/testItem")
async def read_root(request: Request):
    data =await request.json() 
    question= data['question']
    answer = data['answer']
    items =dict(data)
    print(type(items))
    Session = Depends(get_db)
    db = Session
    crud.create_user_item(db=db,item=items)
    #res = requests.post("http://127.0.0.1:8000/users/items/",{'question':question,'answer':answer})
    return {"Hello": "World"}


app.post("/feedback", response_model=schemas.FeedBack)
def create_user_FeedBack(
     item: schemas.FeedbackCreate, db: Session = Depends(get_db)
):
    return crud.create_user_FeedBack(db=db, item=item,)


@app.post("/Feedback")
def feedback(item: schemas.FeedbackCreate, db: Session = Depends(get_db)):
    return crud.create_user_FeedBack(db=db, item=item,)