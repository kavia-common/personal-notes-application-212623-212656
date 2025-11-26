import os
from datetime import datetime, timedelta
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, Depends, HTTPException, status, Body, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel, Field, EmailStr, ConfigDict
from starlette.responses import JSONResponse

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from bson import ObjectId

# =========================
# Configuration and Constants
# =========================

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    # The running preview origin may be injected by env in CI; allow wildcard subdomains via regex below if needed
]

JWT_ALGORITHM_DEFAULT = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES_DEFAULT = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

db_client: Optional[AsyncIOMotorClient] = None
db: Optional[AsyncIOMotorDatabase] = None

# =========================
# Utils - ObjectId helpers
# =========================

class PyObjectId(ObjectId):
    """Helper type for Pydantic to handle MongoDB ObjectId."""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v: Any):
        if isinstance(v, ObjectId):
            return v
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        schema = handler(core_schema)
        schema.update(type="string")
        return schema


# =========================
# Schemas
# =========================

class Token(BaseModel):
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")

class TokenData(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None

class UserBase(BaseModel):
    email: EmailStr = Field(..., description="User email address")

class UserCreate(UserBase):
    password: str = Field(..., min_length=6, description="Plaintext password to register")

class UserPublic(UserBase):
    id: PyObjectId = Field(..., alias="_id", description="User identifier")
    # Allow population by field name so 'id' works, and alias usage for '_id'
    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        str_strip_whitespace=True,
        json_encoders={ObjectId: str, PyObjectId: str},
        ser_json_inf_nan="allow",
    )

class UserDB(UserBase):
    # Use public field name with alias to MongoDB '_id'
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id", description="User identifier")
    password_hash: str

    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        json_encoders={ObjectId: str, PyObjectId: str},
    )

class NoteBase(BaseModel):
    title: str = Field(..., min_length=1, description="Note title")
    content: str = Field("", description="Note body/content")
    tags: List[str] = Field(default_factory=list, description="List of tags")
    archived: bool = Field(False, description="Archived status")

class NoteCreate(NoteBase):
    pass

class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    archived: Optional[bool] = None

class NotePublic(NoteBase):
    id: PyObjectId = Field(..., alias="_id", description="Note identifier")
    user_id: PyObjectId = Field(..., description="Owner user id")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    model_config = ConfigDict(
        populate_by_name=True,
        from_attributes=True,
        json_encoders={ObjectId: str, PyObjectId: str},
    )

class NotesListResponse(BaseModel):
    items: List[NotePublic]
    total: int
    page: int
    page_size: int


# =========================
# Serialization helpers
# =========================

def _stringify_object_id(value: Any) -> Any:
    """Return string version for ObjectId-like values; otherwise unchanged."""
    if isinstance(value, ObjectId):
        return str(value)
    return value

# PUBLIC_INTERFACE
def mongo_to_api_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a MongoDB document dict to an API-friendly dict mapping _id->id and stringifying ObjectIds."""
    if not doc:
        return doc
    out = {}
    for k, v in doc.items():
        if k == "_id":
            out["id"] = _stringify_object_id(v)
        else:
            out[k] = _stringify_object_id(v)
    return out

# PUBLIC_INTERFACE
def mongo_to_api_list(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert a list of MongoDB documents to API-friendly dicts."""
    return [mongo_to_api_doc(d) for d in docs]

# =========================
# App initialization
# =========================

app = FastAPI(
    title="Notes API",
    description="FastAPI backend for personal notes with user authentication",
    version="0.1.0",
    openapi_tags=[
        {"name": "Health", "description": "Service health and connectivity"},
        {"name": "Auth", "description": "Authentication and user endpoints"},
        {"name": "Notes", "description": "CRUD for notes"},
    ],
)
# Verification note:
# from src.api.main import app, UserPublic, NotePublic
# UserPublic.model_validate({'_id': '65e...', 'email': 'a@b.com'})
# NotePublic.model_validate({'_id': '65e...', 'user_id': '65e...', 'title': 't', 'content': '', 'tags': [], 'archived': False, 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()})

# CORS setup from env
def get_cors_origins() -> List[str]:
    """
    Resolve allowed CORS origins from the CORS_ORIGINS environment variable.
    Falls back to DEFAULT_CORS_ORIGINS if not set.
    """
    raw = os.getenv("CORS_ORIGINS", "")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if not origins:
        origins = DEFAULT_CORS_ORIGINS
    return origins

# Apply CORS middleware using resolved origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Configuration loader
# =========================

class Settings(BaseModel):
    mongodb_url: str = Field(..., description="MongoDB connection URL")
    mongodb_db: str = Field(..., description="MongoDB database name")
    jwt_secret: str = Field(..., description="JWT secret for signing")
    jwt_alg: str = Field(default=JWT_ALGORITHM_DEFAULT, description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=ACCESS_TOKEN_EXPIRE_MINUTES_DEFAULT, description="Access token expiry in minutes")

# PUBLIC_INTERFACE
def get_settings() -> Settings:
    """Load settings from environment variables."""
    mongodb_url = os.getenv("MONGODB_URL")
    mongodb_db = os.getenv("MONGODB_DB")
    jwt_secret = os.getenv("JWT_SECRET")
    jwt_alg = os.getenv("JWT_ALG", JWT_ALGORITHM_DEFAULT)
    access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", str(ACCESS_TOKEN_EXPIRE_MINUTES_DEFAULT)))

    if not mongodb_url or not mongodb_db or not jwt_secret:
        raise RuntimeError("Missing required environment variables: MONGODB_URL, MONGODB_DB, JWT_SECRET")

    return Settings(
        mongodb_url=mongodb_url,
        mongodb_db=mongodb_db,
        jwt_secret=jwt_secret,
        jwt_alg=jwt_alg,
        access_token_expire_minutes=access_token_expire_minutes,
    )


# =========================
# Database wiring
# =========================

async def init_indexes(database: AsyncIOMotorDatabase):
    # Users: email unique
    await database["users"].create_index("email", unique=True)
    # Notes: user_id and text index
    await database["notes"].create_index([("user_id", 1)])
    await database["notes"].create_index([("title", "text"), ("content", "text")])

@app.on_event("startup")
async def on_startup():
    settings = get_settings()
    # Assign to module-level vars declared above
    global db_client
    db_client = AsyncIOMotorClient(settings.mongodb_url)
    globals()["db"] = db_client[settings.mongodb_db]
    await init_indexes(globals()["db"])

@app.on_event("shutdown")
async def on_shutdown():
    if db_client:
        db_client.close()

# PUBLIC_INTERFACE
async def get_db() -> AsyncIOMotorDatabase:
    """Dependency that returns the MongoDB database instance."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return db


# =========================
# Security utils (JWT + password)
# =========================

def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# PUBLIC_INTERFACE
def create_access_token(data: Dict[str, Any], settings: Optional[Settings] = None, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token with provided claims."""
    if settings is None:
        settings = get_settings()
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_alg)
    return encoded_jwt

# PUBLIC_INTERFACE
async def get_current_user(token: str = Depends(oauth2_scheme), database: AsyncIOMotorDatabase = Depends(get_db)) -> Dict[str, Any]:
    """Return current user document from token; raises 401 if invalid."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        settings = get_settings()
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_alg])
        user_id: Optional[str] = payload.get("sub")
        email: Optional[str] = payload.get("email")
        if user_id is None or email is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id, email=email)
    except JWTError:
        raise credentials_exception

    user = await database["users"].find_one({"_id": ObjectId(token_data.user_id)})
    if not user:
        raise credentials_exception
    return user


# =========================
# Health Endpoint
# =========================

@app.get("/", tags=["Health"], summary="Health Check", operation_id="health_check")
def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.

    Returns:
    - message: service status string
    - db_connected: whether database is reachable (best-effort)
    """
    db_connected = False
    try:
        if db is not None:
            # A lightweight check: access a collection name (doesn't hit server)
            _ = db.name
            db_connected = True
    except Exception:
        db_connected = False
    return {"message": "Healthy", "db_connected": db_connected}


# =========================
# Auth Routes
# =========================

@app.post("/auth/register", response_model=UserPublic, tags=["Auth"], summary="Register", operation_id="auth_register")
async def register(user_in: UserCreate = Body(...), database: AsyncIOMotorDatabase = Depends(get_db)) -> Any:
    """
    Register a new user.

    Body:
    - email: EmailStr
    - password: str

    Returns:
    - UserPublic
    """
    existing = await database["users"].find_one({"email": user_in.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email is already registered")

    password_hash = get_password_hash(user_in.password)
    doc = {"email": user_in.email, "password_hash": password_hash}
    result = await database["users"].insert_one(doc)
    created = await database["users"].find_one({"_id": result.inserted_id})
    # Ensure password hash is not returned
    created.pop("password_hash", None)
    return mongo_to_api_doc(created)

@app.post("/auth/login", response_model=Token, tags=["Auth"], summary="Login", operation_id="auth_login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), database: AsyncIOMotorDatabase = Depends(get_db)) -> Token:
    """
    Login to obtain JWT.

    Form fields:
    - username: email
    - password: password

    Returns:
    - Token: access_token (JWT bearer)
    """
    user = await database["users"].find_one({"email": form_data.username})
    if not user or not verify_password(form_data.password, user.get("password_hash", "")):
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    token = create_access_token({"sub": str(user["_id"]), "email": user["email"]})
    return Token(access_token=token, token_type="bearer")

@app.get("/auth/me", response_model=UserPublic, tags=["Auth"], summary="Get current user", operation_id="auth_me")
async def me(current_user: Dict[str, Any] = Depends(get_current_user)) -> Any:
    """
    Get the authenticated user's profile.
    """
    sanitized = {k: v for k, v in current_user.items() if k != "password_hash"}
    return mongo_to_api_doc(sanitized)


# =========================
# Notes Routes
# =========================

def build_notes_query(
    user_id: ObjectId,
    search: Optional[str],
    tag: Optional[str],
    archived: Optional[bool],
) -> Dict[str, Any]:
    q: Dict[str, Any] = {"user_id": user_id}
    if archived is not None:
        q["archived"] = archived
    if tag:
        q["tags"] = tag
    if search:
        # Use text index when possible, fallback to regex if text score not usable
        q["$text"] = {"$search": search}
    return q

def build_sort(sort: Optional[str]) -> List[tuple]:
    # sort can be: created_at, -created_at, updated_at, -updated_at, title, -title
    mapping = {
        "created_at": ("created_at", 1),
        "-created_at": ("created_at", -1),
        "updated_at": ("updated_at", 1),
        "-updated_at": ("updated_at", -1),
        "title": ("title", 1),
        "-title": ("title", -1),
    }
    field, direction = mapping.get(sort or "-updated_at", ("updated_at", -1))
    return [(field, direction)]

@app.get(
    "/notes",
    response_model=NotesListResponse,
    tags=["Notes"],
    summary="List notes",
    operation_id="list_notes",
)
async def list_notes(
    search: Optional[str] = Query(None, description="Full-text search across title and content"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    archived: Optional[bool] = Query(None, description="Filter by archived flag"),
    sort: Optional[str] = Query("-updated_at", description="Sort field (e.g., -updated_at, created_at)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    database: AsyncIOMotorDatabase = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> NotesListResponse:
    """
    List notes owned by the current user with optional filters, search, sorting, and pagination.
    """
    user_id = current_user["_id"]
    query = build_notes_query(user_id, search, tag, archived)

    total = await database["notes"].count_documents(query)
    cursor = database["notes"].find(query)

    # Apply sort; prefer $meta textScore sort when searching
    if search:
        cursor = cursor.sort([("score", {"$meta": "textScore"})])
        cursor = cursor.project({"score": {"$meta": "textScore"}, "*": 1})
    else:
        cursor = cursor.sort(build_sort(sort))

    skip = (page - 1) * page_size
    items = await cursor.skip(skip).limit(page_size).to_list(length=page_size)

    api_items = mongo_to_api_list(items)
    return NotesListResponse(items=api_items, total=total, page=page, page_size=page_size)

@app.post(
    "/notes",
    response_model=NotePublic,
    tags=["Notes"],
    summary="Create note",
    operation_id="create_note",
    status_code=201,
)
async def create_note(
    note_in: NoteCreate = Body(...),
    database: AsyncIOMotorDatabase = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> NotePublic:
    """
    Create a new note for the current user.
    """
    now = datetime.utcnow()
    doc = {
        "title": note_in.title,
        "content": note_in.content,
        "tags": note_in.tags,
        "archived": note_in.archived,
        "user_id": current_user["_id"],
        "created_at": now,
        "updated_at": now,
    }
    result = await database["notes"].insert_one(doc)
    created = await database["notes"].find_one({"_id": result.inserted_id})
    return mongo_to_api_doc(created)  # type: ignore

@app.get(
    "/notes/{note_id}",
    response_model=NotePublic,
    tags=["Notes"],
    summary="Get note by id",
    operation_id="get_note",
)
async def get_note(
    note_id: str = Path(..., description="Note id"),
    database: AsyncIOMotorDatabase = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> NotePublic:
    """
    Retrieve a single note by id for the current user.
    """
    if not ObjectId.is_valid(note_id):
        raise HTTPException(status_code=400, detail="Invalid note id")
    note = await database["notes"].find_one({"_id": ObjectId(note_id), "user_id": current_user["_id"]})
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return mongo_to_api_doc(note)  # type: ignore

@app.put(
    "/notes/{note_id}",
    response_model=NotePublic,
    tags=["Notes"],
    summary="Replace note",
    operation_id="update_note",
)
async def replace_note(
    note_id: str = Path(..., description="Note id"),
    note_in: NoteCreate = Body(...),
    database: AsyncIOMotorDatabase = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> NotePublic:
    """
    Replace an entire note document with provided content.
    """
    if not ObjectId.is_valid(note_id):
        raise HTTPException(status_code=400, detail="Invalid note id")
    now = datetime.utcnow()
    result = await database["notes"].find_one_and_update(
        {"_id": ObjectId(note_id), "user_id": current_user["_id"]},
        {
            "$set": {
                "title": note_in.title,
                "content": note_in.content,
                "tags": note_in.tags,
                "archived": note_in.archived,
                "updated_at": now,
            }
        },
        return_document=True,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Note not found")
    return mongo_to_api_doc(result)  # type: ignore

@app.patch(
    "/notes/{note_id}",
    response_model=NotePublic,
    tags=["Notes"],
    summary="Patch note",
    operation_id="patch_note",
)
async def patch_note(
    note_id: str = Path(..., description="Note id"),
    note_in: NoteUpdate = Body(...),
    database: AsyncIOMotorDatabase = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> NotePublic:
    """
    Partially update a note fields.
    """
    if not ObjectId.is_valid(note_id):
        raise HTTPException(status_code=400, detail="Invalid note id")

    updates: Dict[str, Any] = {k: v for k, v in note_in.model_dump(exclude_unset=True).items()}
    if not updates:
        note = await database["notes"].find_one({"_id": ObjectId(note_id), "user_id": current_user["_id"]})
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        return mongo_to_api_doc(note)  # type: ignore

    updates["updated_at"] = datetime.utcnow()
    result = await database["notes"].find_one_and_update(
        {"_id": ObjectId(note_id), "user_id": current_user["_id"]},
        {"$set": updates},
        return_document=True,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Note not found")
    return result  # type: ignore

@app.delete(
    "/notes/{note_id}",
    status_code=204,
    tags=["Notes"],
    summary="Delete note",
    operation_id="delete_note",
)
async def delete_note(
    note_id: str = Path(..., description="Note id"),
    database: AsyncIOMotorDatabase = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Delete a note owned by the current user.
    """
    if not ObjectId.is_valid(note_id):
        raise HTTPException(status_code=400, detail="Invalid note id")
    res = await database["notes"].delete_one({"_id": ObjectId(note_id), "user_id": current_user["_id"]})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Note not found")
    return JSONResponse(status_code=204, content=None)

@app.patch(
    "/notes/{note_id}/archive",
    response_model=NotePublic,
    tags=["Notes"],
    summary="Archive/unarchive note",
    operation_id="archive_note",
)
async def archive_note(
    note_id: str = Path(..., description="Note id"),
    archived: bool = Body(..., embed=True, description="Archive state to set"),
    database: AsyncIOMotorDatabase = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> NotePublic:
    """
    Toggle archive state of a note.
    """
    if not ObjectId.is_valid(note_id):
        raise HTTPException(status_code=400, detail="Invalid note id")
    result = await database["notes"].find_one_and_update(
        {"_id": ObjectId(note_id), "user_id": current_user["_id"]},
        {"$set": {"archived": archived, "updated_at": datetime.utcnow()}},
        return_document=True,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Note not found")
    return result  # type: ignore
