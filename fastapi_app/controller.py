from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from fastapi_app.service import classify_wood_image

app = FastAPI()

@app.get("/test")
def test():
    response_data = {
        "final_grade": "A",
        "price": "233 LKR",
        "grade_probability": {
            "A": 0.6,
            "B": 0.2,
            "C": 0.1,
            "D": 0.1
        },
        "description": "High-quality teak wood with minimal defects, suitable for premium furniture."
    }
    return JSONResponse(content=response_data)

@app.post("/teak-wood-valuation")
async def evaluate_teak_wood(file: UploadFile = File(...)):
    # Read image from uploaded file
    contents = await file.read()
    
    # # Convert to Image
    image = Image.open(io.BytesIO(contents))
    
    # Perform classification (use the actual model here)
    final_grade, price, grade_probability, description = classify_wood_image(image)
    
    
    # Prepare the response
    result = {
        "final_grade": final_grade,
        "price": price,
        "grade_probability": grade_probability,
        "description": description
    }

    # Return the JSON response
    return JSONResponse(content=result)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)