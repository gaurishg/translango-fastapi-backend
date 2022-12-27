from database import UserInDB, engine, Language
from sqlmodel import Session
import bcrypt
import json
import pydantic

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
        user.favourite_languages.append(session.get(Language, 'ja')) # type: ignore
        session.add(user)
        session.commit()

def add_languages():
    class InputLang(pydantic.BaseModel):
        code: str
        name: str
        nativeName: str
    
    with Session(engine) as session:
        with open('iso639_1.json') as f_handle:
            data = [InputLang(**lang) for lang in json.load(f_handle)]
            data = [Language(iso639_1code=lang.code, name_in_en=lang.name) for lang in data]
            session.add_all(data)
            session.commit()

def add_data():
    add_languages()
    add_sample_user()