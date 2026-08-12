import os
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
import jwt
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError, PyMongoError

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sensor_forecast")
USERS_COLLECTION = os.getenv("MONGO_USERS_COLLECTION", "users")

AUTH_SECRET = os.getenv(
    "AUTH_SECRET", "dev-secret-change-me-in-production"
)
TOKEN_EXPIRE_HOURS = int(os.getenv("TOKEN_EXPIRE_HOURS", "72"))
ALGORITHM = "HS256"

router = APIRouter()
security = HTTPBearer(auto_error=False)

_client: Optional[MongoClient] = None


def _get_users_collection():
	global _client
	try:
		if _client is None:
			_client = MongoClient(
				MONGO_URI,
				serverSelectionTimeoutMS=int(os.getenv("MONGO_TIMEOUT_MS", "5000")),
			)
		_col = _client[MONGO_DB_NAME][USERS_COLLECTION]
		_col.create_index([("email", ASCENDING)], unique=True)
		return _col
	except HTTPException:
		raise
	except Exception as exc:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=f"Database unavailable: {exc}",
		) from exc


def _serialize_user(doc: dict) -> dict:
	return {
		"id": str(doc["_id"]),
		"name": doc.get("name", ""),
		"email": doc["email"],
		"created_at": (
			doc["created_at"].isoformat()
			if isinstance(doc.get("created_at"), datetime)
			else doc.get("created_at")
		),
	}


class SignupRequest(BaseModel):
	name: str = Field(..., min_length=2, max_length=100)
	email: EmailStr
	password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
	email: EmailStr
	password: str = Field(..., min_length=1)


class UserResponse(BaseModel):
	id: str
	name: str
	email: str
	created_at: Optional[str] = None


class AuthResponse(BaseModel):
	token: str
	user: UserResponse


def _hash_password(password: str) -> str:
	return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
	try:
		return bcrypt.checkpw(
			password.encode("utf-8"), password_hash.encode("utf-8")
		)
	except ValueError:
		return False


def _create_token(user_id: str) -> str:
	expires = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
	payload = {"sub": str(user_id), "exp": expires}
	return jwt.encode(payload, AUTH_SECRET, algorithm=ALGORITHM)


def _decode_token(token: str) -> Optional[str]:
	try:
		payload = jwt.decode(token, AUTH_SECRET, algorithms=[ALGORITHM])
		return payload.get("sub")
	except jwt.PyJWTError:
		return None


def get_current_user(
	credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
	if credentials is None:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Not authenticated",
			headers={"WWW-Authenticate": "Bearer"},
		)
	user_id = _decode_token(credentials.credentials)
	if user_id is None:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Invalid or expired token",
			headers={"WWW-Authenticate": "Bearer"},
		)
	try:
		user = _get_users_collection().find_one({"_id": ObjectId(user_id)})
	except (ObjectId.InvalidId, PyMongoError) as exc:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Invalid or expired token",
		)
	if user is None:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="User no longer exists",
		)
	return user


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(req: SignupRequest):
	try:
		collection = _get_users_collection()
		user = {
			"name": req.name.strip(),
			"email": req.email.lower().strip(),
			"password_hash": _hash_password(req.password),
			"created_at": datetime.utcnow(),
		}
		try:
			result = collection.insert_one(user)
		except DuplicateKeyError:
			raise HTTPException(
				status_code=status.HTTP_409_CONFLICT,
				detail="An account with this email already exists",
			)
		user["_id"] = result.inserted_id
		return AuthResponse(
			token=_create_token(str(result.inserted_id)),
			user=UserResponse(**_serialize_user(user)),
		)
	except HTTPException:
		raise
	except Exception as exc:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=f"Signup failed: {exc}",
		) from exc


@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest):
	try:
		user = _get_users_collection().find_one(
			{"email": req.email.lower().strip()}
		)
		if user is None or not _verify_password(req.password, user.get("password_hash", "")):
			raise HTTPException(
				status_code=status.HTTP_401_UNAUTHORIZED,
				detail="Invalid email or password",
			)
		return AuthResponse(
			token=_create_token(str(user["_id"])),
			user=UserResponse(**_serialize_user(user)),
		)
	except HTTPException:
		raise
	except Exception as exc:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=f"Login failed: {exc}",
		) from exc


@router.get("/me", response_model=UserResponse)
async def me(current_user: dict = Depends(get_current_user)):
	return UserResponse(**_serialize_user(current_user))