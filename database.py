from sqlmodel import Field # type: ignore
from sqlmodel import Relationship # type: ignore
# from sqlmodel import CHAR
from sqlmodel import SQLModel, create_engine
from sqlalchemy.orm import declared_attr
from sqlmodel.pool import StaticPool
from typing import Optional, List

engine = create_engine(
    'sqlite:///', 
    echo=True,
    connect_args={'check_same_thread': False},
    poolclass=StaticPool
    )


class UserFavLangs(SQLModel, table=True):
    username: str = Field(primary_key=True, foreign_key="users.username")
    code: str = Field(primary_key=True, foreign_key="languages.code")

class LanguageInDB(SQLModel, table=True):
    code: str = Field(primary_key=True, nullable=False, min_length=2)

    users_with_primary_language: List["UserInDB"] = Relationship(back_populates="primary_language")
    users_with_favourite_languages: List["UserInDB"] = Relationship(back_populates="favourite_languages", link_model=UserFavLangs)

    @declared_attr
    def __tablename__(cls) -> str:
        return "languages"

class Language(LanguageInDB):
    name: str

class User(SQLModel): 
    username: str = Field(primary_key=True)
    firstname: str
    middlename: Optional[str] = Field(default=None)
    lastname: Optional[str] = Field(default=None)
    email: str = Field(unique=True, nullable=False)
    primary_lang: str = Field(nullable=False, foreign_key="languages.code", min_length=2)
    

class UserFromFrontend(User):
    password: str

class UserInDB(User, table=True):
    @declared_attr
    def __tablename__(cls) -> str:
        return "users"
    
    hashed_password: bytes
    salt: bytes

    primary_language: LanguageInDB = Relationship(back_populates="users_with_primary_language")
    favourite_languages: List[LanguageInDB] = Relationship(back_populates="users_with_favourite_languages", link_model=UserFavLangs)

class UserToFrontend(User):
    primary_language: LanguageInDB
    favourite_languages: List[LanguageInDB] = Field(min_items=1)


if __name__ == '__main__':
    SQLModel.metadata.create_all(bind=engine)