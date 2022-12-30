# app.py

from google.cloud import translate_v2 as google_translate  # type: ignore
import gcp_api_helpers
import aws_api_helpers
import add_data
from database import (
    User,
    UserFromFrontend,
    UserInDB,
    engine,
    SQLModel,
    UserToFrontend,
    LanguageInDB,
    Language,
)
from PIL import Image, ImageOps
from pydantic import BaseModel, Field  # type: ignore
import bcrypt
from starlette.responses import Response
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.exceptions import HTTPException as StartletteHTTPException
from jose import jwt, JWTError
from sqlmodel import Session
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import FastAPI, Depends, status, File, UploadFile, Body  # type: ignore
from typing import List, Dict, Tuple, Optional  # type: ignore
import sys
import os
import time

current_path = sys.path[0]
sys.path.append(os.path.abspath(os.path.join(current_path, "..")))
sys.path.append(os.path.abspath(os.path.join(current_path, "..", "yolov7")))
# print(sys.path)
# import yolov7.translango

# from passlib.context import CryptContext
# import secrets

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


def authenticate_user(
    username: str, password: str, session: Session = Depends(get_session)
):
    user = session.get(UserInDB, username)
    if user is None:
        return False
    if not bcrypt.checkpw(
        password=password.encode("utf-8"), hashed_password=user.hashed_password
    ):
        return False
    return user


def get_user_from_token_string(
    token_string: str = Depends(oauth2_scheme), session: Session = Depends(get_session)
):
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
    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore
        # return await super().dispatch(request, call_next)
        start_time = time.time()
        response: Response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response


# app.add_middleware(SampleMiddleWare)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")  # type: ignore
def on_startup():
    SQLModel.metadata.create_all(bind=engine)
    add_data.add_data()
    # print(google_translate.Client().translate("Hello", target_language='hi')) #type: ignore
    # print(google_translate.Client().get_languages(target_language='hi')) #type: ignore


@app.get("/testing/users", tags=["Testing"], response_model=List[UserToFrontend])
def testing_get_all_users(session: Session = Depends(get_session)):
    result: List[UserInDB] = session.query(UserInDB).all()
    return result


@app.post("/testing/create-user", tags=["Testing"])
def testing_create_user(
    user_with_pass: UserFromFrontend, session: Session = Depends(get_session)
):
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


@app.get(
    "/available-languages", tags=["No Authentication"], response_model=List[Language]
)
def testing_get_all_languages(
    session: Session = Depends(get_session), target_lang: str = "en"
):
    return gcp_api_helpers.get_languages(target_lang=target_lang)


@app.post(
    "/testing/image-object-detection",
    tags=["Testing"],
    response_model=gcp_api_helpers.CloudVisionAndTranslation,
)
def testing_upload_image(
    *,
    session: Session = Depends(get_session),
    file: UploadFile = File(),
    target_languages: List[str],
):
    target_languages = target_languages[0].split(",")
    image = Image.open(file.file).convert("RGB")
    image = ImageOps.contain(image=image, size=(1280, 720))
    uploaded_image = aws_api_helpers.upload_PIL_Image_to_s3(image=image)
    image_name = uploaded_image.key
    detections = gcp_api_helpers.object_detection_from_s3(image_name)
    detections_with_translation = (
        gcp_api_helpers.add_translation_to_CloudVisionDetections(
            detections=detections,
            target_languages=[LanguageInDB(code=l) for l in target_languages],
        )
    )
    return detections_with_translation


@app.post(
    "/testing/text-translate",
    tags=["Testing"],
    response_model=gcp_api_helpers.TextTranslateResponseToFrontend,
)
def testing_text_translate(
    *,
    text: str = Body(),
    target_lang: str = Body(),
    source_lang: Optional[str] = Body(default=None),
):
    response = gcp_api_helpers.TextTranslateResponseToFrontend.parse_obj(
        gcp_api_helpers.text_translate(text, target_lang, source_lang)
    )
    if response.detectedSourceLanguage is None:
        response.detectedSourceLanguage = source_lang
    return response


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


@app.get("/users/me", response_model=User)
def get_me(current_user: User = Depends(get_user_from_token_string)):
    return current_user
