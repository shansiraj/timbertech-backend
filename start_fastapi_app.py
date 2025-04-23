import uvicorn
from fastapi_app.controller import app  # Import the FastAPI app from your app file

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)