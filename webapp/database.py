from sqlmodel import Field # type: ignore
# from sqlmodel import CHAR
from sqlmodel import SQLModel, create_engine
from sqlalchemy.orm import declared_attr
from sqlmodel.pool import StaticPool
from typing import Optional

engine = create_engine(
    'sqlite:///', 
    echo=True,
    connect_args={'check_same_thread': False},
    poolclass=StaticPool
    )

class Language(SQLModel, table=True):
    iso639_1code: str = Field(primary_key=True, nullable=False, min_length=2, max_length=2)
    name_in_en: str

    def __repr__(self) -> str:
        return f"Language({self.iso639_1code}: {self.name_in_en})"

class User(SQLModel): 
    username: str = Field(primary_key=True)
    firstname: str
    middlename: Optional[str] = Field(default=None)
    lastname: Optional[str] = Field(default=None)
    email: str = Field(unique=True, nullable=False)

class UserFromFrontend(User):
    password: str

class UserInDB(User, table=True):
    @declared_attr
    def __tablename__(cls) -> str:
        return "users"
    
    hashed_password: bytes
    salt: bytes

if __name__ == '__main__':
    SQLModel.metadata.create_all(bind=engine)