from sqlalchemy.orm import Session
import models, schemas
from fastapi import FastAPI,Request

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def create_user(db: Session, user: schemas.UserCreate):
    fake_hashed_password = user.password + "notreallyhashed"
    db_user = models.User(email=user.email, hashed_password=fake_hashed_password,first_name=user.first_name,last_name=user.last_name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user




def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_login(db: Session,user_email,user_password):
    
    return db.query(models.User).filter(models.User.email == user_email,models.User.hashed_password == user_password).first()


def get_items(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Item).offset(skip).limit(limit).all()


def create_user_item(db: Session, item: schemas.ItemCreate):
    db_item = models.Item(**item.dict())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def create_user_item1(db: Session, item:dict):
    db_item = models.Item(item)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


def create_user_FeedBack(db: Session, item: schemas.FeedbackCreate):
    db_feedback = models.Feedback(**item.dict())
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return db_feedback




