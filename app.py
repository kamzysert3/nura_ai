import os
import uuid
import pymongo
import nest_asyncio
import asyncio
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

# Apply nest_asyncio patch
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Get API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGODB_URI = os.getenv("MONGO_URI")

# Check for missing API keys
if not GOOGLE_API_KEY or not MONGODB_URI:
    raise EnvironmentError("Missing GOOGLE_API_KEY or MONGO_URI in .env")


chat_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# Define system prompt template
system_prompt_template = """
  You are a reliable, HIPAA-compliant healthcare AI assistant trained to
  support patients, doctors, and clinical staff by providing accurate,
  compassionate, and medically sound information. Your responses must be
  respectful, evidence-based, and clearly indicate when a question should be
  referred to a licensed medical professional. Prioritize patient safety,
  privacy, and clarity in communication at all times.
"""

@tool(
    name_or_callable="classify_specialty",
    description=(
        "Takes a free-text patient complaint (e.g. 'I have rashes on my arm') "
        "and returns exactly one medical specialty (e.g. 'Dermatologist')."
    )
)
def classify_specialty(complaint: str) -> str:
    prompt = (
        "You are an expert medical router. Given a patient's complaint, output the exact "
        "specialist (e.g., 'Dermatologist', 'Cardiologist') they should see. Just give one word or phrase.\n\n"
        "Examples:\n"
        "Complaint: 'I have chest pain.'\nSpecialist: Cardiologist\n"
        "Complaint: 'My skin is itchy and flaky.'\nSpecialist: Dermatologist\n"
        "Complaint: 'I have persistent heartburn.'\nSpecialist: Gastroenterologist\n\n"
        f"Complaint: '{complaint}'\nSpecialist:"
    )

    return chat_llm.invoke([("user", prompt)])

@tool(
    name_or_callable="find_doctors",
    description=(
        "Given a medical specialty (e.g. 'Dermatologist'), return up to 5 "
        "doctors from MongoDB matching that specialty as a markdown list."
    )
)
def find_doctors(specialty: str) -> str:
    client = pymongo.MongoClient(MONGODB_URI)
    col = client["test"]["doctors"]
    cursor = col.find(
        {"specialty": {"$regex": specialty, "$options": "i"}},
        {"_id": 0, "uid": 1, "name": 1, "hospital": 1, "licenseID": 1, "email": 1, "phone": 1}
    ).limit(5)
    docs = list(cursor)
    client.close()
    if not docs:
        return f"No doctors found for specialty '{specialty}'."

    lines = []
    for d in docs:
        contact = []
        if d.get('email'): contact.append(d['email'])
        if d.get('phone'): contact.append(str(d['phone']))
        contact_info = ", ".join(contact) if contact else 'no contact info'
        lines.append(
            f"- **{d.get('name','Unknown')}** at {d.get('hospital','Unknown Hospital')} "
            f"(License: {d.get('licenseID','N/A')}) — {contact_info}"
        )
    return "\n".join(lines)

@tool(
    name_or_callable="process_document",
    description=(
        "Given an instruction and the text of a document separated by '||',"
        " perform the instruction on the document. Input format: '<instruction>||<document_text>'."
    )
)
def process_document(input_str: str) -> str:
    instruction, doc = input_str.split('||',1)
    prompt = (
        "You are a document assistant. Follow the instruction on the document."
        f"\nInstruction: {instruction}\n\nDocument:\n{doc}"
    )
    resp = chat_llm.invoke([("user", prompt)])
    return resp["messages"][-1][1].strip()

# Populate tools list
tools = [classify_specialty, find_doctors, process_document]

# Initialize FastAPI
app = FastAPI(title="Nura Assistant API")

# Shared memory per thread (simulate per-user thread using thread_id)
thread_memories = {}

# Data models
class ChatRequest(BaseModel):
    thread_id: Optional[str] = None
    message: str
    document_text: Optional[str] = None

class ChatResponse(BaseModel):
    thread_id: str
    response: str

agent_cache = {}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    thread_id = request.thread_id or str(uuid.uuid4())

    # Load or create memory saver
    if thread_id not in thread_memories:
        # thread_memories[thread_id] = MongoDBSaver.from_conn_string(MONGODB_URI)
        thread_memories[thread_id] = MemorySaver()

    memory = thread_memories[thread_id]

    # Reuse agent if already created
    if thread_id not in agent_cache:
        agent_cache[thread_id] = create_react_agent(
            model=chat_llm,
            tools=tools,
            prompt=system_prompt_template,
            checkpointer=memory,
        )

    agent = agent_cache[thread_id]

    user_input = request.message
    if request.document_text:
        user_input += f"||{request.document_text}"

    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke({"messages": [("user", user_input)]}, config)
    response = result["messages"][-1].content

    return ChatResponse(thread_id=thread_id, response=response)

@app.get("/")
async def health_check():
    return {"status": "healthy"}

import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
