"""Export the trained instruction-tuned transformer weights to ONNX.

Run from the repository root:
    python -m new_model.convert_to_onnx

The output defaults to ``new_model/final_instruction_tuned.onnx``. Install
``tf2onnx`` and ``onnx`` in the TensorFlow environment before conversion.
"""

import argparse
from pathlib import Path

import tensorflow as tf
from pathlib import Path
import os
import tensorflow as tf
from transformers import BertTokenizerFast
TRAIN_RATIO = 0.99
SEQ_LEN = 256
STRIDE = 256
# PER_GPU_BATCH_SIZE = 38
# GLOBAL_BATCH_SIZE  = 76
BATCH_SIZE = 24
BUFFER_SIZE = 10000
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 13
DFF = 1024
DROPOUT = 0.1
# T
EPOCHS =30
INITIAL_LR = 3e-5
MIN_LR = 8e-6
WARMUP_EPOCHS = 1
WEIGHT_DECAY = 0.01
try:
    from .transformer import DecoderOnlyTransformer, SEQ_LEN
except ImportError:
    from transformer import DecoderOnlyTransformer, SEQ_LEN

WEIGHTS_PATH =  "./models/final31_transformer_last_epoch.weights.h5"


BASE_DIR = Path(__file__).resolve().parent

# WEIGHTS_PATH = BASE_DIR / "final_instruction_tuned.weights.h5"

DEFAULT_TOKENIZER_DIR = BASE_DIR / "tokenizer"
DEFAULT_TOKENIZER_NAME = "bert-base-uncased"
EXPECTED_VOCAB_SIZE = 30522
# try:
#     from .main import BASE_DIR, WEIGHTS_PATH, build_model, load_tokenizer
#     from .transformer import SEQ_LEN
# except ImportError:
#     from main import BASE_DIR, WEIGHTS_PATH, build_model, load_tokenizer
#     from transformer import SEQ_LEN


def load_tokenizer(tokenizer_path=None):
    """
    Load an existing local tokenizer or create and save a new one
    if it does not exist.
    """

    if tokenizer_path is not None:
        tokenizer_dir = Path(tokenizer_path).expanduser().resolve()
    else:
        tokenizer_dir = DEFAULT_TOKENIZER_DIR

    if tokenizer_dir.exists() and any(tokenizer_dir.iterdir()):
        tokenizer = BertTokenizerFast.from_pretrained(
            str(tokenizer_dir)
        )
        print(f"Loaded existing tokenizer: {tokenizer_dir}")

    else:
        tokenizer_name = os.getenv(
            "NEW_MODEL_TOKENIZER_NAME",
            DEFAULT_TOKENIZER_NAME
        )

        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tokenizer_dir))

        print(f"Created and saved new tokenizer: {tokenizer_dir}")

    if len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"Unexpected tokenizer vocabulary size: {len(tokenizer)}. "
            f"Expected {EXPECTED_VOCAB_SIZE}."
        )

    return tokenizer

def build_model(tokenizer, weights_path=WEIGHTS_PATH):
    """
    Build the decoder-only Transformer and load trained weights.
    """

    model = DecoderOnlyTransformer(
        num_layers=NUM_LAYERS,
        d_model=EMBED_DIM,
        num_heads=NUM_HEADS,
        dff=DFF,
        vocab_size=len(tokenizer),
        maximum_position_encoding=SEQ_LEN,
        rate=DROPOUT
    )
    # Build model variables before loading weights.
    dummy_input = tf.zeros((1, SEQ_LEN), dtype=tf.int32)
    model(dummy_input, training=False)

    weights_path = Path(weights_path).expanduser().resolve()

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Trained weights not found: {weights_path}"
        )

    model.load_weights(str(weights_path))

    print(f"Loaded trained weights: {weights_path}")

    return model

DEFAULT_OUTPUT_PATH = BASE_DIR / "final_instruction_tuned.onnx"
DEFAULT_OPSET = 17


def convert_to_onnx(
    weights_path=WEIGHTS_PATH,
    output_path=DEFAULT_OUTPUT_PATH,
    tokenizer_path=None,
    opset=DEFAULT_OPSET,
):
    try:
        import onnx
        import tf2onnx
    except ImportError as exc:
        raise RuntimeError(
            "ONNX conversion requires tf2onnx and onnx. Install them in the "
            "same environment as TensorFlow, then rerun this script."
        ) from exc

    tokenizer = load_tokenizer(tokenizer_path)
    model = build_model(tokenizer, weights_path=weights_path)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_signature = [
        tf.TensorSpec(
            shape=(None, None),
            dtype=tf.int32,
            name="input_ids",
        )
    ]

    @tf.function(input_signature=input_signature)
    def serving_function(input_ids):
        return {"logits": model(input_ids, training=False)}

    _, _ = tf2onnx.convert.from_function(
        serving_function,
        input_signature=input_signature,
        opset=opset,
        output_path=str(output_path),
    )

    onnx.checker.check_model(str(output_path))
    print(f"ONNX model saved and checked: {output_path}")
    print(f"Input: int32 [batch, sequence], sequence length <= {SEQ_LEN}")
    print("Output: float32 [batch, sequence, vocabulary] logits")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=WEIGHTS_PATH,
        help="Path to the TensorFlow .weights.h5 checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output ONNX file (defaults inside new_model/).",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Optional local tokenizer directory; overrides the environment setting.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET,
        help="ONNX opset version (default: %(default)s).",
    )
    args = parser.parse_args()

    convert_to_onnx(
        weights_path=args.weights,
        output_path=args.output,
        tokenizer_path=args.tokenizer,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()



# """Export the trained instruction-tuned transformer weights to ONNX.

# Run from the repository root:
#     python -m new_model.convert_to_onnx

# The output defaults to ``new_model/final_instruction_tuned.onnx``. Install
# ``tf2onnx`` and ``onnx`` in the TensorFlow environment before conversion.
# """

# import argparse
# from pathlib import Path

# import tensorflow as tf

# try:
#     from .main import BASE_DIR, WEIGHTS_PATH, build_model, load_tokenizer
#     from .transformer import SEQ_LEN
# except ImportError:
#     from main import BASE_DIR, WEIGHTS_PATH, build_model, load_tokenizer
#     from transformer import SEQ_LEN


# DEFAULT_OUTPUT_PATH = BASE_DIR / "final_instruction_tuned.onnx"
# DEFAULT_OPSET = 17


# def convert_to_onnx(
#     weights_path=WEIGHTS_PATH,
#     output_path=DEFAULT_OUTPUT_PATH,
#     tokenizer_path=None,
#     opset=DEFAULT_OPSET,
# ):
#     try:
#         import onnx
#         import tf2onnx
#     except ImportError as exc:
#         raise RuntimeError(
#             "ONNX conversion requires tf2onnx and onnx. Install them in the "
#             "same environment as TensorFlow, then rerun this script."
#         ) from exc

#     tokenizer = load_tokenizer(tokenizer_path)
#     model = build_model(tokenizer, weights_path=weights_path)
#     output_path = Path(output_path).expanduser().resolve()
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     input_signature = [
#         tf.TensorSpec(
#             shape=(None, None),
#             dtype=tf.int32,
#             name="input_ids",
#         )
#     ]

#     @tf.function(input_signature=input_signature)
#     def serving_function(input_ids):
#         return {"logits": model(input_ids, training=False)}

#     _, _ = tf2onnx.convert.from_function(
#         serving_function,
#         input_signature=input_signature,
#         opset=opset,
#         output_path=str(output_path),
#     )

#     onnx.checker.check_model(str(output_path))
#     print(f"ONNX model saved and checked: {output_path}")
#     print(f"Input: int32 [batch, sequence], sequence length <= {SEQ_LEN}")
#     print("Output: float32 [batch, sequence, vocabulary] logits")
#     return output_path


# def main():
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument(
#         "--weights",
#         type=Path,
#         default=WEIGHTS_PATH,
#         help="Path to the TensorFlow .weights.h5 checkpoint.",
#     )
#     parser.add_argument(
#         "--output",
#         type=Path,
#         default=DEFAULT_OUTPUT_PATH,
#         help="Output ONNX file (defaults inside new_model/).",
#     )
#     parser.add_argument(
#         "--tokenizer",
#         type=Path,
#         default=None,
#         help="Optional local tokenizer directory; overrides the environment setting.",
#     )
#     parser.add_argument(
#         "--opset",
#         type=int,
#         default=DEFAULT_OPSET,
#         help="ONNX opset version (default: %(default)s).",
#     )
#     args = parser.parse_args()

#     convert_to_onnx(
#         weights_path=args.weights,
#         output_path=args.output,
#         tokenizer_path=args.tokenizer,
#         opset=args.opset,
#     )


# if __name__ == "__main__":
#     main()
