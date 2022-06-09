from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from algorithms.valid_print import isPrintable
from io import BytesIO

class Grid(BaseModel):
    cells: List[List[int]]


app = FastAPI()


@app.post("/validPrint")
async def root(grid: Grid):
    return {"is_valid": isPrintable(grid.cells)}

@app.post("/isSmileyFace")
async def root():
    return {"similarity": 0.0}

@app.post("/generateSmiley")
async def root():
    image = BytesIO()
    # generate smiley
    img.save(image, format='JPEG', quality=85)
    image.seek(0)
    return StreamingResponse(image.read(), media_type='image/jpeg')