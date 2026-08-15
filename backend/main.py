from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form
)

from fastapi.middleware.cors import CORSMiddleware

from PIL import Image

import io

from inference import (
    predict_answer,
    predict_rationale
)


app = FastAPI(
    title="VCR AI"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def home():

    return {
        "status": "VCR AI running"
    }


# ============================================================
# ANSWER
# ============================================================

@app.post("/predict-answer")
async def answer(
    image: UploadFile = File(...),
    question: str = Form(...),
    answers: str = Form(...)
):

    image_bytes = await image.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    answer_list = [
        x.strip()
        for x in answers.split("|")
    ]

    result = predict_answer(
        image,
        question,
        answer_list
    )

    return result


# ============================================================
# RATIONALE
# ============================================================

@app.post("/predict-rationale")
async def rationale(
    image: UploadFile = File(...),
    question: str = Form(...),
    selected_answer: str = Form(...),
    rationales: str = Form(...)
):

    image_bytes = await image.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    rationale_list = [
        x.strip()
        for x in rationales.split("|")
    ]

    result = predict_rationale(
        image,
        question,
        selected_answer,
        rationale_list
    )

    return result