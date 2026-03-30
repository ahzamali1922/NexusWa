# main.py
import sys
import os
from pathlib import Path

# 1. Tell Python to look in the parent directory (the NexusWatch root folder)
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

# 2. Now standard imports will work!
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(title="NexusWatch AML API", version="1.0")

# ... (the rest of your main.py code stays exactly the same)

# Allow your React frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "NexusWatch Backend is live."}