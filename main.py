from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from pydantic import Field
from enum import Enum
import uuid
import os

from src.main import run 

class SolverType(str, Enum):
    cpsat = "cpsat"
    cbc = "cbc"


class Job(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    tsv: float
    p: int
    solver: SolverType
    

app = FastAPI(
    title="UrBreath - OptimalSnow - API",
    description="",
    version="0.1",
    contact={
        "name": "Nefeli Kousta",
        "email": "nefelikousta@gmail.com",
    },
    license_info={
        "name": "BSD License",
    }
)


jobs_db = {}

@app.post("/jobs/", status_code=201)
async def create_job(job_data: Job): 
    
    job_dict = job_data.model_dump()
    job_id_str = str(job_dict["id"])
    
    job_dict["created_at"] = datetime.now()
    job_dict["status"] = "pending"
    
    jobs_db[job_id_str] = job_dict
    
    return job_dict


@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: uuid.UUID):
    
    str_id = str(job_id)
    
    if str_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[str_id]
    
    return {
        "job_id": job["id"],
        "status": job["status"],
        "created_at": job["created_at"]
    }