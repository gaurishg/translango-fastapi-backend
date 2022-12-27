# app.py

import sys
import os
import time

current_path = sys.path[0]
sys.path.append(os.path.abspath(os.path.join(current_path, "..")))
sys.path.append(os.path.abspath(os.path.join(current_path, "..", "yolov7")))
# print(sys.path)
# import yolov7.translango
from typing import List, Dict, Tuple, Optional  # type: ignore

from fastapi import FastAPI, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session
# from passlib.context import CryptContext
from jose import jwt, JWTError
from starlette.exceptions import HTTPException as StartletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import bcrypt
# import secrets
from pydantic import BaseModel

from database import User, UserFromFrontend, UserInDB, engine, SQLModel, UserToFrontend
import add_data

app: FastAPI = FastAPI()

SECRET_KEY = os.environ.get("SECRET_KEY", "")
ALGORITHM = os.environ.get("ALGORITHM", "")
if SECRET_KEY == "" or ALGORITHM == "":
    print(f"SECRET_KEY or ALGORITHM environment variable(s) not set")
    exit()

# pwd_context = CryptContext(schemes=['bcrypt'])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str
    email: str


def get_session():
    with Session(bind=engine) as db:
        yield db


def authenticate_user(username: str, password: str, session: Session=Depends(get_session)):
    user = session.get(UserInDB, username)
    if user is None:
        return False
    if not bcrypt.checkpw(
        password=password.encode("utf-8"), hashed_password=user.hashed_password
    ):
        return False
    return user

def get_user_from_token_string(token_string: str = Depends(oauth2_scheme), session: Session=Depends(get_session)):
    credentials_exception = StartletteHTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload: Dict[str, str] = jwt.decode(
            token=token_string, key=SECRET_KEY, algorithms=[ALGORITHM]
        )
        username = payload.get("username")
        email = payload.get("email")
        if username is None or email is None:
            raise credentials_exception
        user = session.get(UserInDB, username)
        if user is None:
            raise credentials_exception
        return user
    except JWTError:
        raise credentials_exception


class SampleMiddleWare(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response: # type: ignore
        # return await super().dispatch(request, call_next)
        start_time = time.time()
        response: Response = await call_next(request)
        process_time = time.time() - start_time
        response.headers['X-Process-Time'] = str(process_time)
        return response

app.add_middleware(SampleMiddleWare)
app.add_middleware(CORSMiddleware, allow_origins=['*'])

@app.on_event("startup") # type: ignore
def on_startup():
    SQLModel.metadata.create_all(bind=engine)
    add_data.add_data()

@app.get("/admin/users", tags=["Admin"], response_model=List[UserToFrontend])
def admin_get_all_users(session: Session=Depends(get_session)):
    result: List[UserInDB] = session.query(UserInDB).all()
    # return [UserToFrontend.from_orm(user) for user in result]
    return result


@app.post("/admin/create-user", tags=["Admin"])
def admin_create_user(user_with_pass: UserFromFrontend, session: Session=Depends(get_session)):
    password = user_with_pass.password
    salt = bcrypt.gensalt()
    password_hash = bcrypt.hashpw(password=password.encode("utf-8"), salt=salt)
    new_user = UserInDB(
        **user_with_pass.dict(),
        hashed_password=password_hash,
        salt=salt,
    )
    session.add(new_user)
    session.commit()
    return User.from_orm(new_user)


@app.post("/token", response_model=Token)
def get_access_token(formdata: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(formdata.username, formdata.password)
    if not user:
        raise StartletteHTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = jwt.encode(
        {"username": user.username, "email": user.email},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )

    return Token(access_token=access_token, token_type="bearer")

@app.get('/users/me', response_model=User)
def get_me(current_user: User = Depends(get_user_from_token_string)):
    return current_user