from sqlmodel import Field # type: ignore
from sqlmodel import SQLModel, create_engine
from sqlalchemy.orm import declared_attr
from typing import Optional

engine = create_engine('sqlite:///db.sqlite', echo=True)

class User(SQLModel): 
    username: str = Field(primary_key=True)
    firstname: str
    middlename: Optional[str]
    lastname: Optional[str]
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