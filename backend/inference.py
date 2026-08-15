import torch
import re

from PIL import Image

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor
)


MODEL_PATH = "../model/Qwen3-VL-8B-VCR"

processor = AutoProcessor.from_pretrained(
    MODEL_PATH
)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto"
)

model.eval()

print("✅ VCR model loaded")


# ============================================================
# EXACT PROMPTS FROM YOUR NOTEBOOK
# ============================================================

def answer_prompt(question, answers):

    return f"""You are solving Visual Commonsense Reasoning.

Your job is to select the ONE answer that is best supported by the image and question.

Pay attention to:
1. Which people/objects are referenced.
2. Their positions and interactions.
3. Actions and body language.
4. What is actually visible versus what is merely plausible.

Do not choose an answer just because it sounds reasonable — it must be supported by the image.

Question:
{question}

Answer choices:
0. {answers[0]}
1. {answers[1]}
2. {answers[2]}
3. {answers[3]}

Return only the number of the best answer: 0, 1, 2, or 3."""


def rationale_prompt(
    question,
    answer,
    rationales
):

    return f"""You are solving Visual Commonsense Reasoning.

The answer has already been selected. Your job is to choose the ONE rationale that
actually explains why that answer is supported.

A good rationale must:
- agree with the image
- agree with the question
- agree with the selected answer
- avoid unsupported assumptions

Question:
{question}

Selected answer:
{answer}

Rationale choices:
0. {rationales[0]}
1. {rationales[1]}
2. {rationales[2]}
3. {rationales[3]}

Return only the number of the best rationale: 0, 1, 2, or 3."""


# ============================================================
# CHOICE EXTRACTION
# ============================================================

def extract_choice(output):

    match = re.search(
        r"\b([0-3])\b",
        output
    )

    return int(match.group(1)) if match else -1


# ============================================================
# MODEL CALL
# ============================================================

def ask_model(image, prompt):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    )

    inputs = {
        k: v.to(model.device)
        if hasattr(v, "to")
        else v
        for k, v in inputs.items()
    }

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )

    generated_ids = output_ids[
        0,
        inputs["input_ids"].shape[1]:
    ]

    output = processor.decode(
        generated_ids,
        skip_special_tokens=True
    )

    return output


# ============================================================
# ANSWER PREDICTION
# ============================================================

def predict_answer(
    image,
    question,
    answers
):

    prompt = answer_prompt(
        question,
        answers
    )

    raw_output = ask_model(
        image,
        prompt
    )

    prediction = extract_choice(
        raw_output
    )

    return {
        "prediction": prediction,
        "raw_output": raw_output
    }


# ============================================================
# RATIONALE PREDICTION
# ============================================================

def predict_rationale(
    image,
    question,
    selected_answer,
    rationales
):

    prompt = rationale_prompt(
        question,
        selected_answer,
        rationales
    )

    raw_output = ask_model(
        image,
        prompt
    )

    prediction = extract_choice(
        raw_output
    )

    return {
        "prediction": prediction,
        "raw_output": raw_output
    }