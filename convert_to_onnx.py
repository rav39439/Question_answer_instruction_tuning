"""Export the trained instruction-tuned transformer weights to ONNX.

Run from the repository root:
    python -m new_model.convert_to_onnx

The output defaults to ``new_model/final_instruction_tuned.onnx``. Install
``tf2onnx`` and ``onnx`` in the TensorFlow environment before conversion.
"""

import argparse
from pathlib import Path

import tensorflow as tf

try:
    from .main import BASE_DIR, WEIGHTS_PATH, build_model, load_tokenizer
    from .transformer import SEQ_LEN
except ImportError:
    from main import BASE_DIR, WEIGHTS_PATH, build_model, load_tokenizer
    from transformer import SEQ_LEN


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
