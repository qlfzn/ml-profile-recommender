from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services import DBHandler
from application import FreelancerProjectMatcher
import uuid
import json

app = FastAPI()
matcher = FreelancerProjectMatcher()
db = DBHandler()

class RequestModel(BaseModel):
    project_id: str

@app.post("/match")
async def match_freelancers(request: RequestModel):
    try:
        project_id = request.project_id
        matched_results = matcher.process_matches_data(project_id=project_id)

        if not matched_results:
            raise HTTPException(status_code=404, detail="No matches found")
        
        result_data = {"project_id": project_id, "matches": matched_results}
        db.store_match_results(json.dumps(result_data))
        
        return {"message": "Matching completed successfully", "data": result_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
