




"""Load the instruction-tuned decoder and generate answers using ONNX Runtime."""

import os
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import BertTokenizerFast

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
ONNX_PATH = BASE_DIR / "final_instruction_tuned.onnx"
DEFAULT_TOKENIZER_DIR = BASE_DIR / "tokenizer"
DEFAULT_TOKENIZER_NAME = "bert-base-uncased"
EXPECTED_VOCAB_SIZE = 30522


app = FastAPI()

tokenizer = None
model = None


class QuestionRequest(BaseModel):
    question: str
    max_gen: int = 250


def load_tokenizer(tokenizer_path=None):
    """Load the checkpoint's tokenizer from a local path or BERT fallback."""

    configured_path = (
        tokenizer_path
        or os.environ.get("NEW_MODEL_TOKENIZER_PATH")
    )

    if configured_path:
        tokenizer_source = Path(configured_path).expanduser()
        if not tokenizer_source.exists():
            raise FileNotFoundError(
                "Tokenizer path does not exist: "
                f"{tokenizer_source}. Set NEW_MODEL_TOKENIZER_PATH to the "
                "directory containing the trained tokenizer."
            )
    elif DEFAULT_TOKENIZER_DIR.is_dir():
        tokenizer_source = DEFAULT_TOKENIZER_DIR
    else:
        tokenizer_source = os.environ.get(
            "NEW_MODEL_TOKENIZER_NAME",
            DEFAULT_TOKENIZER_NAME,
        )

    tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_source))

    actual_vocab_size = len(tokenizer)

    if actual_vocab_size != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            "Tokenizer vocabulary does not match the model checkpoint: "
            f"expected {EXPECTED_VOCAB_SIZE}, got {actual_vocab_size} "
            f"from {tokenizer_source}."
        )

    return tokenizer


def load_onnx_model(onnx_path=ONNX_PATH):
    """Load the trained ONNX model."""

    onnx_path = Path(onnx_path)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"]
    )

    return session


def positional_encoding_np(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(
        10000,
        (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model)
    )

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return angle_rads[np.newaxis, ...].astype(np.float32)


def generate_answer(model, tokenizer, question, seq_len=256, max_gen=150):

    if seq_len < 2:
        raise ValueError("seq_len must be at least 2.")

    if max_gen < 1:
        return ""

    prompt = f"Question: {question}\nAnswer:"

    token_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False
    )

    if not token_ids:
        raise ValueError("Question produced no tokens.")

    context_limit = seq_len - 1
    token_ids = token_ids[-context_limit:]

    generated_ids = []

    end_ids = tokenizer.encode(
        "\nEnd of answer.",
        add_special_tokens=False,
    )

    inputs = model.get_inputs()

    # Identify both ONNX inputs.
    input_names = [inp.name for inp in inputs]

    print("ONNX inputs:")
    for inp in inputs:
        print(inp.name, inp.shape, inp.type)

    # Identify token input and positional encoding input.
    token_input = inputs[0]
    positional_input = inputs[1]

    token_dtype = (
        np.int64
        if token_input.type == "tensor(int64)"
        else np.int32
    )

    # Get positional encoding dimension from ONNX input shape.
    d_model = positional_input.shape[-1]

    if not isinstance(d_model, int):
        raise ValueError(
            f"Invalid positional encoding dimension: {d_model}"
        )

    pe = positional_encoding_np(seq_len, d_model)

    for _ in range(max_gen):

        input_seq = np.array(
            [token_ids],
            dtype=token_dtype
        )

        # Supply both required ONNX inputs.
        feed = {
            token_input.name: input_seq,
            positional_input.name: pe
        }

        logits = model.run(None, feed)[0]

        next_id_value = int(
            np.argmax(logits[0, -1, :])
        )

        generated_ids.append(next_id_value)

        if (
            end_ids
            and len(generated_ids) >= len(end_ids)
            and generated_ids[-len(end_ids):] == end_ids
        ):
            generated_ids = generated_ids[:-len(end_ids)]
            break

        token_ids.append(next_id_value)

        if len(token_ids) > context_limit:
            token_ids = token_ids[-context_limit:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()


# =========================================================
# STREAMING GENERATION
# =========================================================

def generate_answer_stream(
    model,
    tokenizer,
    question,
    seq_len=256,
    max_gen=150
):
    """Generate and yield decoded text as tokens are produced."""

    if seq_len < 2:
        raise ValueError("seq_len must be at least 2.")

    if max_gen < 1:
        return

    prompt = f"Question: {question}\nAnswer:"

    token_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False
    )

    if not token_ids:
        raise ValueError("Question produced no tokens.")

    context_limit = seq_len - 1
    token_ids = token_ids[-context_limit:]

    end_ids = tokenizer.encode(
        "\nEnd of answer.",
        add_special_tokens=False,
    )

    inputs = model.get_inputs()

    token_input = inputs[0]
    positional_input = inputs[1]

    token_dtype = (
        np.int64
        if token_input.type == "tensor(int64)"
        else np.int32
    )

    d_model = positional_input.shape[-1]

    if not isinstance(d_model, int):
        raise ValueError(
            f"Invalid positional encoding dimension: {d_model}"
        )

    pe = positional_encoding_np(seq_len, d_model)

    generated_ids = []

    for _ in range(max_gen):

        input_seq = np.array(
            [token_ids],
            dtype=token_dtype
        )

        feed = {
            token_input.name: input_seq,
            positional_input.name: pe
        }

        logits = model.run(None, feed)[0]

        next_id_value = int(
            np.argmax(logits[0, -1, :])
        )

        # Stop at SEP or EOS.
        if next_id_value in [
            tokenizer.sep_token_id,
            tokenizer.eos_token_id
        ]:
            break

        generated_ids.append(next_id_value)

        # Preserve the existing end phrase stopping logic.
        if (
            end_ids
            and len(generated_ids) >= len(end_ids)
            and generated_ids[-len(end_ids):] == end_ids
        ):
            generated_ids = generated_ids[:-len(end_ids)]
            break

        token_ids.append(next_id_value)

        if len(token_ids) > context_limit:
            token_ids = token_ids[-context_limit:]

        # Decode generated tokens and stream only newly completed text.
        decoded_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        if decoded_text:
            yield decoded_text

            # Keep the generated text in sync with the streamed text.
            # The complete decoded prefix is sent to the client.


# =========================================================
# FASTAPI ROUTES
# =========================================================

@app.on_event("startup")
def startup_event():

    global tokenizer, model

    tokenizer = load_tokenizer()
    model = load_onnx_model()

    print("Tokenizer and ONNX model loaded successfully.")


@app.post("/generate")
def generate(request: QuestionRequest):

    answer = generate_answer(
        model=model,
        tokenizer=tokenizer,
        question=request.question,
        max_gen=request.max_gen
    )

    return {
        "question": request.question,
        "answer": answer
    }


@app.post("/generate/stream")
def generate_stream(request: QuestionRequest):

    def stream():

        try:
            previous_text = ""

            for current_text in generate_answer_stream(
                model=model,
                tokenizer=tokenizer,
                question=request.question,
                max_gen=request.max_gen
            ):

                # Send only the newly generated portion.
                new_text = current_text[len(previous_text):]

                if new_text:
                    yield f"data: {json.dumps({'token': new_text})}\n\n"

                previous_text = current_text

            yield "data: [DONE]\n\n"

        except Exception as e:

            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
         reload=True
    )

    # uvicorn.run(
    #     app,
    #     host="0.0.0.0",
    #     port=8000,
    #     reload=True
    # )