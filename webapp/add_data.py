from database import UserInDB, engine, LanguageInDB
from sqlmodel import Session
import bcrypt
from google.cloud import translate_v2 as google_translate # type: ignore
import pydantic
from typing import List

def add_sample_user():
    password = 'password'.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_pw = bcrypt.hashpw(password=password, salt=salt)
    with Session(engine) as session:
        user = UserInDB(
            username='username',
            firstname='user',
            email='user@example.com',
            hashed_password=hashed_pw,
            salt=salt,
            primary_lang='en'
        )
        user.favourite_languages.append(session.get(LanguageInDB, 'ja')) # type: ignore
        session.add(user)
        session.commit()

def add_languages():
    class InputLang(pydantic.BaseModel):
        language: str
        name: str
    
    with Session(bind=engine) as session:
        langs: List[InputLang] = [InputLang(**lang) for lang in google_translate.Client().get_languages()] #type: ignore
        languages: List[LanguageInDB] = [LanguageInDB(code=lang.language) for lang in langs]
        session.add_all(languages)
        session.commit()

def add_data():
    add_languages()
    add_sample_user()