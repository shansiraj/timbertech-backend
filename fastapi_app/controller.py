from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from fastapi_app.service import classify_wood_image

app = FastAPI()

#
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