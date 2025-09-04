# --- paste below your imports or near the bottom of a file ---
import argparse
import sys
import torch

import warnings
warnings.filterwarnings(
    "ignore",
    message="`do_sample` is set to `False`.*",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="`do_sample` is set to `False`.*top_k.*",
    category=UserWarning
)


def _build_chat_prompt(tokenizer, user_text: str, system_text: str = "You are a helpful AI assistant."):
    """
    Build a prompt compatible with Qwen Instruct. If the tokenizer supports
    apply_chat_template, we’ll use it. Otherwise, fall back to a simple prefix.
    """
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # adds the assistant prefix
        )
    # Fallback format
    return f"{system_text}\n\nUser: {user_text}\nAssistant:"

def generate_reply(model, tokenizer, prompt: str, max_new_tokens: int = 256):
    """
    Deterministic generation for QA/chat testing. Returns only the new text.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # deterministic
            temperature=0.0,          # ignored when do_sample=False
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Only decode newly generated tokens
    gen_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0].strip()
    return text

def chat_loop(model, tokenizer, system_text: str, max_new_tokens: int):
    print("\nInteractive QA/chat with Qwen. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user_text.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_text:
            continue

        prompt = _build_chat_prompt(tokenizer, user_text, system_text=system_text)
        reply = generate_reply(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        print(f"Qwen: {reply}")

# ---- MAIN (run this file directly to chat) ----
if __name__ == "__main__":
    from model_setup import load_qwen_model, get_model_info  # uses your existing wrapper

    parser = argparse.ArgumentParser(description="Quick interactive QA/chat with Qwen.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HF model id or local path")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', 'cuda:0', or 'cpu'")
    parser.add_argument("--dtype", type=str, default="float16", help="'float16', 'bfloat16', or 'float32'")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--system", type=str, default="You are a helpful AI assistant.")
    args = parser.parse_args()

    # Load model/tokenizer with your wrapper (eval mode, left padding, pad/eos set)
    model, tokenizer = load_qwen_model(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
    )
    info = get_model_info(model)
    print(f"Loaded {args.model} on device (CUDA avail={info['is_cuda_available']}), dtype={info['dtype']}.")
    print(f"Parameters: {info['total_parameters']:,}\n")

    # Start interactive chat
    chat_loop(model, tokenizer, system_text=args.system, max_new_tokens=args.max_new_tokens)
