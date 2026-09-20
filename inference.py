import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "He-Tag/smallm-125m"
MAX_NEW_TOKENS = 200


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(name, device):
    model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True,
                                                 dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    return model.to(device).eval(), tokenizer


def encode(tokenizer, messages, device):
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True).to(device)


def answer(model, tokenizer, messages, device, max_new_tokens=MAX_NEW_TOKENS):
    room = model.config.max_seq_len - max_new_tokens
    inputs = encode(tokenizer, messages, device)
    while inputs["input_ids"].shape[1] > room and len(messages) > 1:
        messages = messages[2:]
        inputs = encode(tokenizer, messages, device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    reply = generated[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(reply, skip_special_tokens=True).strip()


def chat(model, tokenizer, device, system=None):
    history = [{"role": "system", "content": system}] if system else []
    print("Ctrl-D to quit, /reset to start a new conversation.\n")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            print()
            return
        if not question:
            continue
        if question == "/reset":
            history = history[:1] if system else []
            print("(new conversation)\n")
            continue
        history.append({"role": "user", "content": question})
        reply = answer(model, tokenizer, history, device)
        history.append({"role": "assistant", "content": reply})
        print(reply + "\n")


def main():
    parser = argparse.ArgumentParser(description="Chat with smallm-125m.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--system", default=None)
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()

    device = pick_device()
    model, tokenizer = load(args.model, device)
    print(f"{args.model} on {device}, context {model.config.max_seq_len} tokens")

    if args.prompt:
        messages = ([{"role": "system", "content": args.system}] if args.system else [])
        messages.append({"role": "user", "content": args.prompt})
        print(answer(model, tokenizer, messages, device))
    else:
        chat(model, tokenizer, device, args.system)


if __name__ == "__main__":
    main()
