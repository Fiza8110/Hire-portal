from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer
from config.config import JObs_COL,APPLICATION_COL,REGISTER_COL,fs
from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.responses import RedirectResponse, JSONResponse
from bson import ObjectId
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from datetime import datetime, timedelta
import os
from routes.home import   send_application_success_email
from datetime import datetime
import json
from pydantic import BaseModel, EmailStr, Field
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi import Body
import fitz  # PyMuPDF
import re
import requests 
from fastapi.responses import StreamingResponse
import mimetypes
from dotenv import load_dotenv
load_dotenv()
route = APIRouter() # route = APIRouter(): Creates a modular group of routes.
class JobUpdate(BaseModel):#Defines validation rules for form data submitted while applying for a job.
    Job_Title: str
    Job_Description: str
    Experience: str
    Skills: str
    Location: str

oauth2_schema = OAuth2PasswordBearer(tokenUrl="token") # Setup for token-based authentication.

templates = Jinja2Templates(directory='templates') # Jinja2Templates: Tells FastAPI to render HTML from the templates directory.

route.mount("/static", StaticFiles(directory = "static"), name = "static") # mount("/static", ...): Serves static files (CSS/JS/images) from /static.

HF_TOKEN = os.getenv("HF_TOKEN")

@route.post("/ask-ai")
async def ask_ai(message: dict = Body(...)):
    user_input = message.get("message", "")
    if not user_input:
        return {"reply": "Please enter a message."}

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    payload = {
        "inputs": user_input
    }

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",  # ✅ Verified model
            headers=headers,
            json=payload
        )

        print("➡️ Status Code:", response.status_code)
        print("➡️ Response Text:", repr(response.text))

        if response.status_code != 200:
            return JSONResponse(status_code=500, content={"reply": f"❌ Error {response.status_code}: {response.text}"})

        output = response.json()

        if isinstance(output, list) and "generated_text" in output[0]:
            return {"reply": output[0]["generated_text"]}
        elif isinstance(output, dict) and "generated_text" in output:
            return {"reply": output["generated_text"]}
        else:
            return {"reply": str(output)}

    except Exception as e:
        return JSONResponse(status_code=500, content={"reply": f"❌ Exception: {str(e)}"})
#route to render dashboard html page
@route.get("/dashboard")
async def dashboard(request: Request):
    user_email = request.session.get("user_email")
    jobs_list = list(JObs_COL.find({}))
    applied_jobs = list(APPLICATION_COL.find({"email": user_email}))

    applied_job_titles = {job["job_title"] for job in applied_jobs}
    print(jobs_list)
    print(applied_jobs)
    print(applied_job_titles)
    return templates.TemplateResponse("Dashboard.html", {
        "request": request,
        "jobs_list": jobs_list,
        "applied_job_titles": applied_job_titles
    })

# route to render HR dashboard html page
@route.get("/hrDashboard", response_class=HTMLResponse)
async def hrDashboard(request: Request):
    # Total Candidates
    total_candidates = APPLICATION_COL.count_documents({})

    # Total Admin Users (assuming role field exists as 'admin')
    total_users = REGISTER_COL.count_documents({"role": "admin"})

    # Total Job Postings
    total_jobs = JObs_COL.count_documents({})

    # New Applications in last 7 days
    last_week = datetime.now() - timedelta(days=7)
    new_applications = APPLICATION_COL.count_documents({
       
        "status": "Applied"
    })
   
    # Pending Approvals with status = "in-progress"
    pending_approvals = APPLICATION_COL.count_documents({"status": "in-progress"})

    rejected_applications = APPLICATION_COL.count_documents({"status": "Rejected"})

    return templates.TemplateResponse("HRDashboard.html", {
        "request": request,
        "total_candidates": total_candidates,
        "total_users": total_users,
        "total_jobs": total_jobs,
        "new_applications": new_applications,
        "pending_approvals": pending_approvals,
        "rejected_applications": rejected_applications
    })
# route to render forgot password html page
@route.get("/forgotPassword")
def forgotPassword(request: Request):
  
    return templates.TemplateResponse("ForgotPassword.html", {"request": request})


# route to render job apply html page
@route.get("/applayJob")
def hrDashboard(request: Request):
    return templates.TemplateResponse("JobApply.html", {"request": request})

class JobApplication(BaseModel):
    job_title: str
    first_name: str
    last_name: str
    email: EmailStr
    mobile: str = Field(..., min_length=10, max_length=15)
    current_employee: str
    company_name: str | None = None
    company_location: str | None = None
    experience: str | None = None
    current_ctc: str | None = None
    expected_ctc: str | None = None
    notice_period: str | None = None

@route.post("/applyJob")
async def apply_job(
    job_title: str = Form(...),
    first_name: str = Form(...),
    last_name: str = Form(...),
    email: EmailStr = Form(...),
    mobile: str = Form(...),
    current_employee: str = Form(...),
    company_name: str = Form(None),
    company_location: str = Form(None),
    experience: str = Form(None),
    current_ctc: str = Form(None),
    expected_ctc: str = Form(None),
    notice_period: str = Form(None),
    resume: UploadFile = File(...),
):
    try:
        resume_data = await resume.read()
        resume_id = fs.put(resume_data, filename=resume.filename, content_type=resume.content_type)

        application_data = {
            "job_title": job_title,
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "mobile": mobile,
            "current_employee": current_employee,
            "company_name": company_name,
            "company_location": company_location,
            "experience": experience,
            "current_ctc": current_ctc,
            "expected_ctc": expected_ctc,
            "notice_period": notice_period,
            "resume_id": resume_id,  # Save file ID
            "status": "Applied",
            "applied_on": datetime.utcnow().isoformat(),
        }

        if APPLICATION_COL.find_one({"email": email}):
            raise HTTPException(status_code=400, detail="Email already exists.")

        result = APPLICATION_COL.insert_one(application_data)
        application_data["_id"] = str(result.inserted_id)
        full_name = f"{first_name} {last_name}"
        send_application_success_email(email, full_name, job_title)
        return JSONResponse(status_code=200, content={"message": "Application submitted successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
# Helper function to convert ObjectId to string
def serialize_objectid(obj):
    if isinstance(obj, ObjectId):# convert it to a string using str(obj).
        return str(obj)
    elif isinstance(obj, dict):#recursively call serialize_objectid on each value of the dictionary
        return {key: serialize_objectid(value) for key, value in obj.items()}
    elif isinstance(obj, list):# recursively process each item in the list.
        return [serialize_objectid(item) for item in obj]
    else:
        return obj

@route.get("/viewApplications")
def view_applications(request: Request):#This function handles the request.
    # apps = list(APPLICATION_COL.find({}))#Fetches all applications from the db
    # Fetch only applications with status "Applied" or "in-progress"
    apps = list(APPLICATION_COL.find({"status": {"$in": ["Applied", "in-progress"]}}))
    # convert ObjectIds
    for a in apps:
        a["_id"] = str(a["_id"])#uses _id fields of type ObjectId ,converts each _id to a string 
    return templates.TemplateResponse("HRcards.html", {
        "request": request,
        "applications_list": apps,
        "show_search": True
    })

@route.get("/candidateApplications")
def hrDashboard(request: Request):
    user_email = request.session.get("user_email")
    applications = list(APPLICATION_COL.find({"email": user_email}))
    return templates.TemplateResponse("CandidateStatus.html", {"request": request,"applications": applications})
# route to view details of a specific application
@route.get("/viewDetails/{application_id}", response_class=HTMLResponse)
async def view_details(request: Request, application_id: str):
    obj_id = ObjectId(application_id)

    # Fetch the application details first
    application_data = APPLICATION_COL.find_one({"_id": obj_id})
    print(application_data)

    if application_data:
        # Update status if it is 'Applied'
        if application_data.get("status") == "Applied":
            APPLICATION_COL.update_one(
                {"_id": obj_id},
                {"$set": {"status": "in-progress"}}
            )
            # Re-fetch updated data after update
            application_data = APPLICATION_COL.find_one({"_id": obj_id})

        application_data = serialize_objectid(application_data)  # Convert ObjectId
        return templates.TemplateResponse("ViewApplication.html", {
            "request": request,
            "application_data": application_data
        })

    else:
        return {"message": "Application not found!"}


# to get post new job application
@route.get("/postnewjob")
def hrDashboard(request: Request):

    return templates.TemplateResponse("Postnewjob.html", {"request": request})
@route.get("/joblist") #defines a GET endpoint at the URL path /joblist
# Renders the main dashboard showing all jobs by fetching them from JObs_COL
def joblist(request: Request):#Shows a list of all jobs currently posted.
    jobs_list = list(JObs_COL.find({}))
    return templates.TemplateResponse("Joblist.html", {"request": request, "jobs_list": jobs_list})
@route.get("/jobDetails/{job_id}")
def job_details(request: Request, job_id: str):#{job_id} is a path parameter
    job = JObs_COL.find_one({"_id": ObjectId(job_id)})#ObjectId(job_id) to convert the string into a MongoDB ObjectId.
    if job:
        job = serialize_objectid(job)  # Convert ObjectId to string
        return templates.TemplateResponse("Jobdetails.html", {"request": request, "job": job})
    else:
        raise HTTPException(status_code=404, detail="Job not found")
# Route to render edit job page
@route.get("/editJob/{job_id}", response_class=HTMLResponse)
async def edit_job(request: Request, job_id: str):
    job = JObs_COL.find_one({"_id": ObjectId(job_id)})
    if job:
        job = serialize_objectid(job)
        return templates.TemplateResponse("EditJob.html", {"request": request, "job": job})
    else:
        raise HTTPException(status_code=404, detail="Job not found")
# Route to handle editing and saving the job details
@route.put("/updateJob/{job_id}", response_class=JSONResponse)
async def update_job(job_id: str, job_data: JobUpdate = Body(...)):
    # Convert the Pydantic model to a plain dict
    updated_fields = job_data.dict()

    result = JObs_COL.update_one(
        {"_id": ObjectId(job_id)},
        {"$set": updated_fields}
    )

    if result.modified_count:
        # Add the ID to the response so you can verify on the frontend
        response_data = updated_fields.copy()
        response_data["_id"] = job_id
        return JSONResponse({"message": "Job updated successfully", "data": response_data}, status_code=200)

    # If no document was modified, either nothing changed or the ID wasn't found
    raise HTTPException(status_code=400, detail="No job found or no changes made")
# Route to delete a job
@route.delete("/deleteJob/{job_id}", response_class=JSONResponse)
async def delete_job(job_id: str):
    result = JObs_COL.delete_one({"_id": ObjectId(job_id)})
    
    if result.deleted_count:
        return JSONResponse(content={"message": "Job deleted successfully"}, status_code=200)
    else:
        raise HTTPException(status_code=404, detail="Job not found")
@route.get("/.well-known/appspecific/com.chrome.devtools.json")
async def devtools_stub():
    return JSONResponse(content={"status": "not implemented"}, status_code=200)
# to post new job application post action
@route.post("/postnewjob")
async def hrDashboard(request: Request, data: dict):#Defines an asynchronous function to handle the request.
    print(data)
   
    # Create new job data
    newjob = {#A dictionary newjob is created using values extracted from the data dictionary.
        "Job_Title": data["Job_Title"],
        "Job_Description": data["Job_Description"],
        "Experience": data["Experience"],
        "Skills": data["Skills"],
        "Location": data["Location"]
    }

    # Insert the new job into the database
    result = JObs_COL.insert_one(newjob)

    # Convert ObjectId to string
    newjob['_id'] = str(result.inserted_id)

    # Return JSON response with the new job data
    return JSONResponse(content={"message": "Job posted successfully", "data": newjob}, status_code=200)

from fastapi.responses import FileResponse

@route.get("/add-candidate")
def get_add_candidate(request: Request):
    user_email = request.session.get("user_email")
    user = REGISTER_COL.find_one({"email": user_email})
    user_role = user.get("role") if user else "user"
    request.session["user_role"] = user_role
    job_titles = [j["Job_Title"] for j in JObs_COL.find({}, {"_id": 0, "Job_Title": 1})]
    return templates.TemplateResponse("AddCandidate.html", {
        "request": request,
        "job_titles": job_titles,
        "user_role": user_role
    })

@route.post("/add-candidate")
async def post_add_candidate(
    job_title: str = Form(...),
    first_name: str = Form(...),
    last_name: str = Form(...),
    email: EmailStr = Form(...),
    mobile: str = Form(...),
    current_employee: str = Form(...),
    company_name: str = Form(None),
    company_location: str = Form(None),
    experience: str = Form(None),
    current_ctc: str = Form(None),
    expected_ctc: str = Form(None),
    notice_period: str = Form(None),
    resume: UploadFile = File(...)
):
    try:
        resume_data = await resume.read()
        resume_id = fs.put(resume_data, filename=resume.filename, content_type=resume.content_type)

        if APPLICATION_COL.find_one({"email": email}):
            raise HTTPException(400, "Email already exists.")

        APPLICATION_COL.insert_one({
            "job_title": job_title,
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "mobile": mobile,
            "current_employee": current_employee,
            "company_name": company_name,
            "company_location": company_location,
            "experience": experience,
            "current_ctc": current_ctc,
            "expected_ctc": expected_ctc,
            "notice_period": notice_period,
            "resume_id": resume_id,
            "status": "Applied",
            "added_on": datetime.utcnow().isoformat(),
        })

        return RedirectResponse(url="/viewApplications", status_code=303)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# List of all candidates
@route.get("/listCandidates", response_class=HTMLResponse)
async def list_candidates(request: Request):
    user_email = request.session.get("user_email")
    user = REGISTER_COL.find_one({"email": user_email})
    user_role = user.get("role") if user else "user"
    request.session["user_role"] = user_role

    candidates = list(APPLICATION_COL.find({}, {
        "_id": 1,
        "first_name": 1,
        "last_name": 1,
        "job_title": 1,
        "email": 1,
        "status": 1
    }))

    # Add full name here
    candidates = [
        {
            "_id": str(candidate["_id"]),
            "full_name": f"{candidate.get('first_name', '')} {candidate.get('last_name', '')}".strip(),
            "job_title": candidate.get("job_title", ""),
            "email": candidate.get("email", ""),
            "status": candidate.get("status", "")
        }
        for candidate in candidates
    ]

    candidates_json = json.dumps(candidates)
    return templates.TemplateResponse("ListCandidates.html", {
        "request": request,
        "candidates_json": candidates_json,
        "user_role": user_role
    })
@route.get("/downloadResume/{application_id}")
async def download_resume(application_id: str):
    try:
        if not ObjectId.is_valid(application_id):
            raise HTTPException(status_code=400, detail="Invalid ID")
        
        application = APPLICATION_COL.find_one({"_id": ObjectId(application_id)})
        if not application or not application.get("resume_id"):
            raise HTTPException(status_code=404, detail="Resume not found")

        grid_out = fs.get(application["resume_id"])
        return StreamingResponse(
            grid_out,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{grid_out.filename}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")



@route.delete("/deleteApplication/{application_id}")
async def delete_application(application_id: str):
    """
    Delete a job application and its associated resume file
    """
    try:
        # Find the application first
        application = APPLICATION_COL.find_one({"_id": ObjectId(application_id)})
        
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        
        # Delete the resume file if it exists
        if "resume" in application and os.path.exists(application["resume"]):
            try:
                os.remove(application["resume"])
            except OSError as e:
                print(f"Error deleting resume file: {e}")
        
        # Delete from database
        result = APPLICATION_COL.delete_one({"_id": ObjectId(application_id)})
        
        if result.deleted_count == 1:
            return {"message": "Application deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete application")
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting application: {str(e)}"
        )
@route.post("/parse-resume")
async def parse_resume(resume: UploadFile = File(...)):
    try:
        contents = await resume.read()
        with open("temp_resume.pdf", "wb") as f:
            f.write(contents)

        text = ""
        with fitz.open("temp_resume.pdf") as doc:
            for page in doc:
                text += page.get_text()

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        lower_text = text.lower()

        # === NAME (top 5 lines with letters only) ===
        full_name = ""
        for line in lines[:5]:
            clean_line = line.strip()
            if len(clean_line.split()) in [2, 3] and all(word.isalpha() for word in clean_line.split()):
                full_name = clean_line        
                break

        name_parts = full_name.split()
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[-1] if len(name_parts) > 1 else ""

        # === MOBILE ===
        phone_match = re.search(r"\+?\d[\d\s\-()]{9,14}", text)
        mobile = phone_match.group().strip() if phone_match else ""

       

        # === EXPERIENCE ===
        exp_match = re.search(r"\b(\d+)\+?\s*(years?|yrs?|year)\b|\bexperience\b", lower_text)
        # exp_match = re.search(r"\bexperience\b", line_lower.lower())
        experience = exp_match.group() if exp_match else ""

        return JSONResponse(content={
            "parsed": {
                "first_name": first_name,
                "last_name": last_name,
                "mobile": mobile,
                "experience": experience
            }
        })

    except Exception as e:
        print("Resume parsing error:", e)
        return JSONResponse(status_code=500, content={"error": "Failed to parse resume"})
@route.get("/viewResume/{application_id}")
async def view_resume(application_id: str):
    try:
        if not ObjectId.is_valid(application_id):
            raise HTTPException(status_code=400, detail="Invalid ID")
        
        application = APPLICATION_COL.find_one({"_id": ObjectId(application_id)})
        if not application or not application.get("resume_id"):
            raise HTTPException(status_code=404, detail="Resume not found")

        grid_out = fs.get(application["resume_id"])
        return StreamingResponse(
            grid_out,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{grid_out.filename}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
@route.get("/getJobDetailsByTitle/{job_title}")
async def get_job_details_by_title(job_title: str):
    try:
        job = JObs_COL.find_one({"Job_Title": job_title})

        if job:
            return {
                "job_title": job.get("Job_Title", "No title"),
                "experience": job.get("Experience", "No experience specified"),
                "skills": job.get("Skills", "No skills listed"),
                "description": job.get("Job_Description", "No description available"),
                "location": job.get("Location", "Location not specified")
            }
        else:
            return {"error": "Job not found"}
    except Exception as e:
        return {"error": f"Server error: {str(e)}"}


def extract_skills(text):
    url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    prompt = f"""
    Extract a list of technical and soft skills from the following resume text, designed for ATS compatibility.
    Skills may be listed explicitly in bullets (e.g., • Java, - SQL, * Communication), bubbles (e.g., ● Python), 
    comma-separated (e.g., Java, SQL, Python), or under sections like 'Skills:' or mentioned in context (e.g., 'Proficient in Java', 'Developed using SQL').
    Return the result as a Python list of strings, e.g., ['java', 'sql', 'communication'], using lowercase and comma-separated.
    Return [] if no skills are found. Do not include explanations.
    Resume text:
    {text[:4000]}
    """
    payload = {"inputs": prompt}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching skills from Hugging Face API: {e}")
        return fallback_skill_extraction(text)

    output = response.json()
    print("Hugging Face Raw Output:", output)

    if isinstance(output, list) and output and "generated_text" in output[0]:
        skills_text = output[0]["generated_text"]
    elif isinstance(output, dict) and "generated_text" in output:
        skills_text = output["generated_text"]
    else:
        skills_text = str(output)

    try:
        skills_text = skills_text.replace(prompt, "").strip()
        match = re.search(r'\[.*?\]', skills_text, re.DOTALL)
        if match:
            skills_str = match.group(0)
            skills = eval(skills_str)
        else:
            delimiters = r'[\n•●-*,\s]+'
            skills = re.split(delimiters, skills_text)
            skills = [s.strip('[]"\', ') for s in skills if s.strip()]

        if isinstance(skills, list):
            return [s.strip().lower() for s in skills if s.strip() and not s.isspace()]
        return fallback_skill_extraction(text)
    except Exception as e:
        print(f"Skill parsing error: {e}")
        return fallback_skill_extraction(text)

def fallback_skill_extraction(text):
    text = text.lower()
    skills = set()
    patterns = [
        r'(?:•|●|-|\*)\s*(\w+)',  # Bullet or bubble followed by word
        r',?\s*(\w+)(?:,|\n|$)',  # Comma-separated or newline-separated words
        r'proficient\s+in\s+(\w+)',  # Context like "Proficient in Java"
        r'developed\s+using\s+(\w+)',  # Context like "Developed using SQL"
        r'skills\s*:\s*([\w\s,]+)',  # Skills section
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            skills.update(re.split(r'[,;\s]+', match) if isinstance(match, str) else [match])
    return list(skills)

@route.get("/compareSkills/{application_id}/{job_title}")
async def compare_skills(application_id: str, job_title: str):
    try:
        print(f"Comparing Skills for Application: {application_id} | Job Title: {job_title}")

        if not ObjectId.is_valid(application_id):
            return JSONResponse(status_code=400, content={"error": "Invalid application ID"})

        application = APPLICATION_COL.find_one({"_id": ObjectId(application_id)})
        if not application:
            return JSONResponse(status_code=404, content={"error": "Application not found"})

        resume_id = application.get("resume_id")
        if not resume_id:
            return JSONResponse(status_code=404, content={"error": "Resume not found"})

        grid_out = fs.get(ObjectId(resume_id))
        doc = fitz.open(stream=grid_out.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        print(f"Extracted Resume Text (first 500 chars): {text[:500]}...")

        resume_skills = extract_skills(text)
        print(f"Extracted Resume Skills: {resume_skills}")

        job = JObs_COL.find_one({"Job_Title": job_title})
        if not job:
            return JSONResponse(status_code=404, content={"error": "Job not found"})

        job_skills = [skill.strip().lower() for skill in job.get("Skills", "").split(",")]
        critical_skills = [skill.strip().lower() for skill in job.get("Critical_Skills", "").split(",")] or []
        print(f"Job Skills: {job_skills}, Critical Skills: {critical_skills}")

        matched_skills = list(set(resume_skills) & set(job_skills))
        missing_skills = list(set(job_skills) - set(matched_skills))
        skill_match_percentage = (len(matched_skills) / len(job_skills) * 100) if job_skills else 0

        critical_matched = list(set(resume_skills) & set(critical_skills))
        ats_score = (skill_match_percentage * 0.7) + (len(critical_matched) / len(critical_skills) * 100 * 0.3 if critical_skills else 0)
        ats_score = min(max(ats_score, 0), 100)

        print(f"Matched Skills: {matched_skills}")
        print(f"Missing Skills: {missing_skills}")
        print(f"Skill Match Percentage: {skill_match_percentage:.2f}%")
        print(f"ATS Score: {ats_score:.2f}%")

        return JSONResponse({
            "job_title": job_title,
            "email": application["email"],
            "skill_match_percentage": round(skill_match_percentage, 2),
            "ats_score": round(ats_score, 2),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills
        })
    except Exception as e:
        print(f"Server error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Server error: {str(e)}"})