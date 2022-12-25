from sqlmodel import Field # type: ignore
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.orm import declared_attr
from typing import Optional

engine = create_engine('sqlite:///db.sqlite', echo=True)

class User(SQLModel): 
    username: str = Field(primary_key=True)
    firstname: str
    middlename: Optional[str]
    lastname: Optional[str]
    email: str = Field(unique=True, nullable=False)

    @staticmethod
    def from_user_in_db(u: "UserInDB") -> "User":
        return User(
            username=u.firstname,
            firstname=u.firstname,
            middlename=u.middlename,
            lastname=u.lastname,
            email=u.email
        )

class UserFromFrontend(User):
    password: str

class UserInDB(User, table=True):
    @declared_attr
    def __tablename__(cls) -> str:
        return "users"
    
    hashed_password: bytes
    salt: bytes

    @staticmethod
    def get_from_username(username: str) -> Optional["UserInDB"]:
        user: Optional[UserInDB] = db.query(UserInDB).where(UserInDB.username == username).first()
        return user

SQLModel.metadata.create_all(bind=engine)
db = Session(bind=engine)