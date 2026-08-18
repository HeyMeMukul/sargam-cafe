import os
import shutil
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from agent_runner import transcribe_audio_agentic

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Receives the audio file from the frontend."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"filename": file.filename, "status": "Upload successful", "path": file_path}

@app.websocket("/ws/agent")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket connection to stream the Agent's thought process 
    and piano commands back to the frontend in real-time.
    """
    await websocket.accept()
    try:
        # Wait for the frontend to send the filename to process
        data = await websocket.receive_text()
        filepath = os.path.join(UPLOAD_DIR, data)
        
        if not os.path.exists(filepath):
            await websocket.send_text(f"[Error] File {data} not found on server.")
            return

        # Define the callback function that the Agent uses to send logs/commands to UI
        async def log_to_ui(message: str):
            try:
                await websocket.send_text(message)
            except Exception:
                pass

        # Start the Agent!
        await transcribe_audio_agentic(filepath, log_callback=log_to_ui)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_text(f"[Fatal Error] {str(e)}")

# Run with: uvicorn main:app --reload
