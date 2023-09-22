from pydantic import BaseModel



class UserBase(BaseModel):
    email: str
    first_name:str
    last_name:str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool

    class Config:
        orm_mode = True



class ItemBase(BaseModel):
    question: str
    answer: str | None = None


class ItemCreate(ItemBase):
    pass


class Item(ItemBase):
    id: int
    

    class Config:
        orm_mode = True
