# app.py

import gcp_api_helpers
import add_data
from database import (
    User,
    UserFromFrontend,
    UserInDB,
    UserToFrontendWithToken,
    engine,
    SQLModel,
    UserToFrontend,
    LanguageInDB,
    Language,
    Token,
    TokenData,
)
from PIL import Image, ImageOps
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
import os
import time


app: FastAPI = FastAPI()

SECRET_KEY = os.environ.get("SECRET_KEY", "")
ALGORITHM = os.environ.get("ALGORITHM", "")
if SECRET_KEY == "" or ALGORITHM == "":
    print(f"SECRET_KEY or ALGORITHM environment variable(s) not set")
    exit()

# pwd_context = CryptContext(schemes=['bcrypt'])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_session():
    with Session(bind=engine) as db:
        yield db


def authenticate_user(username: str, password: str, session: Session):
    user = session.get(UserInDB, username)
    if user is None:
        return False
    if not bcrypt.checkpw(
        password=password.encode("utf-8"), hashed_password=user.hashed_password
    ):
        return False
    return user


def get_access_token_from_username_password(
    username: str, password: str, session: Session
):
    user = authenticate_user(username, password, session)
    if not user:
        raise StartletteHTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_data = jwt.encode(
        {"username": user.username, "email": user.email},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    return Token(access_token_data=access_token_data, token_type="bearer")


def get_user_from_token_string(
    token_string: str = Depends(oauth2_scheme), session: Session = Depends(get_session)
):
    credentials_exception = StartletteHTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    print(f"Token string received: {token_string}")
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
        return UserToFrontendWithToken.from_orm(
            user, {"token": Token(access_token_data=token_string, token_type="Bearer")}
        )
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
    allow_origins=[
        "http://localhost:3000",
        "https://frontend.d2e7dlzpkoas0a.amplifyapp.com",
        "https://gaurishg.github.io",
        "http://gaurishg.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")  # type: ignore
def on_startup():
    if os.path.exists("db.sqlite"):
        os.remove("db.sqlite")
    SQLModel.metadata.create_all(bind=engine)
    add_data.add_data()
    # print(google_translate.Client().translate("Hello", target_language='hi')) #type: ignore
    # print(google_translate.Client().get_languages(target_language='hi')) #type: ignore


@app.get("/testing/users", tags=["Testing"], response_model=List[UserToFrontend])
def testing_get_all_users(session: Session = Depends(get_session)):
    result: List[UserInDB] = session.query(UserInDB).all()
    return result


@app.post("/signup", response_model=UserToFrontendWithToken)
def testing_create_user(
    user_with_pass: UserFromFrontend, session: Session = Depends(get_session)
):
    password = user_with_pass.password
    fav_lang_codes = user_with_pass.favourite_langs
    salt = bcrypt.gensalt()
    password_hash = bcrypt.hashpw(password=password.encode("utf-8"), salt=salt)
    if session.get(UserInDB, user_with_pass.username):
        raise StartletteHTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail="Username already exists",
        )

    new_user = UserInDB(
        **user_with_pass.dict(),  # type: ignore
        hashed_password=password_hash,
        salt=salt,
    )

    for code in fav_lang_codes:
        new_user.favourite_languages.append(session.get(LanguageInDB, code))  # type: ignore
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    access_token = get_access_token_from_username_password(
        username=user_with_pass.username, password=password, session=session
    )
    response = UserToFrontendWithToken.from_orm(
        new_user, update={"token": access_token}
    )
    return response


@app.post("/login", response_model=UserToFrontendWithToken)
def user_login(
    username: str = Body(),
    password: str = Body(),
    session: Session = Depends(get_session),
):
    user = authenticate_user(username=username, password=password, session=session)
    if not user:
        raise StartletteHTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username or password wrong",
        )
    response = UserToFrontendWithToken.from_orm(
        user,
        update={
            "token": get_access_token_from_username_password(
                username=username, password=password, session=session
            )
        },
    )
    return response


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
    source_language: str = Body(default="en"),
    target_languages: List[str] = Body(),
):
    target_languages = target_languages[0].split(",")
    image = Image.open(file.file).convert("RGB")
    image = ImageOps.contain(image=image, size=(1280, 720))
    uploaded_image = gcp_api_helpers.upload_PIL_Image(image=image)
    image_name: str = uploaded_image.name  # type: ignore
    return gcp_api_helpers.object_detection_with_translation_from_url(
        url=gcp_api_helpers.filename_to_gs_bucket_uri(image_name),
        target_languages=[LanguageInDB(code=l) for l in target_languages],
        source_language=LanguageInDB(code=source_language),
    )


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
def get_access_token(
    formdata: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session),
):
    return get_access_token_from_username_password(
        formdata.username, formdata.password, session=session
    )


@app.get("/users/me", response_model=UserToFrontend)
def get_me(current_user: UserToFrontend = Depends(get_user_from_token_string)):
    return current_user
