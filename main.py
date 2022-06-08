from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from algorithms.valid_print import isPrintable


class Grid(BaseModel):
    cells: List[List[int]]


app = FastAPI()


@app.post("/validPrint")
async def root(grid: Grid):
    return {"is_valid": isPrintable(grid.cells)}
