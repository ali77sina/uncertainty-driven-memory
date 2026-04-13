"""
GPT-2 Latent Density Uncertainty for Math Reasoning
=====================================================

Uses GPT-2's hidden states to build a density model of "correct reasoning."
Compares three branching strategies on GSM8K-style arithmetic:

  1. Greedy decoding
  2. Token-entropy branching (standard)
  3. Latent-density branching (ours) — branches when hidden state is OOD

Run on GPU:  python gpt2_experiment.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
GMM_COMPONENTS = 16
N_CORRECT_EXAMPLES = 200
N_TEST = 50
MAX_GEN_TOKENS = 200
LATENT_LAYER = -1  # last hidden layer

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Simple arithmetic problems ───────────────────────────────────────────────

def generate_math_problems(n, rng=None):
    """Generate simple multi-step arithmetic word problems with solutions."""
    if rng is None:
        rng = np.random.RandomState(SEED)
    problems = []
    for _ in range(n):
        a = rng.randint(10, 100)
        b = rng.randint(10, 100)
        c = rng.randint(5, 50)
        op1 = rng.choice(["bought", "found", "received"])
        op2 = rng.choice(["gave away", "lost", "used"])
        item = rng.choice(["apples", "books", "coins", "marbles", "cards"])

        answer = a + b - c
        prompt = (
            f"Q: Tom had {a} {item}. He {op1} {b} more. "
            f"Then he {op2} {c}. How many {item} does Tom have now?\n"
            f"A: Let's solve step by step.\n"
        )
        solution = (
            f"Tom started with {a} {item}.\n"
            f"He {op1} {b} more: {a} + {b} = {a + b}.\n"
            f"He {op2} {c}: {a + b} - {c} = {answer}.\n"
            f"The answer is {answer}."
        )
        problems.append({
            "prompt": prompt,
            "solution": solution,
            "full": prompt + solution,
            "answer": answer,
        })
    return problems


# ── Latent state extraction ──────────────────────────────────────────────────

def get_hidden_states(model, tokenizer, text, layer=LATENT_LAYER):
    """Get hidden states from a specific layer for each token position."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer]  # (1, seq_len, hidden_dim)
    return hidden.squeeze(0).cpu().numpy()


def get_step_states(model, tokenizer, text, layer=LATENT_LAYER):
    """Get hidden state at newline positions (reasoning step boundaries)."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    token_ids = inputs["input_ids"].squeeze().tolist()
    newline_id = tokenizer.encode("\n")[0]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer].squeeze(0).cpu().numpy()

    step_positions = [i for i, tid in enumerate(token_ids) if tid == newline_id]
    if not step_positions:
        step_positions = [len(token_ids) - 1]

    return hidden[step_positions]


# ── Build density model from correct examples ───────────────────────────────

def build_density_model(model, tokenizer, correct_examples):
    """Fit GMM on hidden states from correct reasoning traces."""
    print("Building latent density model...")
    all_states = []
    for ex in correct_examples:
        states = get_step_states(model, tokenizer, ex["full"])
        all_states.append(states)
    all_states = np.concatenate(all_states, axis=0)
    print(f"  Collected {all_states.shape[0]} step states, dim={all_states.shape[1]}")

    gmm = GaussianMixture(
        n_components=min(GMM_COMPONENTS, len(all_states) // 2),
        covariance_type="diag",
        random_state=SEED,
    )
    gmm.fit(all_states)
    print(f"  GMM fit with {gmm.n_components} components")

    # calibration: score correct examples to set threshold
    scores = gmm.score_samples(all_states)
    threshold = np.percentile(scores, 10)
    print(f"  Score stats: mean={scores.mean():.1f}, p10={threshold:.1f}, min={scores.min():.1f}")

    return gmm, threshold


# ── Generation with different branching strategies ───────────────────────────

def generate_greedy(model, tokenizer, prompt, max_tokens=MAX_GEN_TOKENS):
    """Standard greedy generation."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    n_fwd = 0

    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        n_fwd += 1

        decoded = tokenizer.decode(next_token.squeeze())
        if "answer is" in tokenizer.decode(input_ids.squeeze()[-20:]):
            # generate a few more tokens to get the number
            for _ in range(10):
                with torch.no_grad():
                    outputs = model(input_ids)
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                n_fwd += 1
                if tokenizer.decode(next_token.squeeze()) in [".", "\n"]:
                    break
            break

    text = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
    return text, n_fwd


def generate_entropy_branching(model, tokenizer, prompt, max_tokens=MAX_GEN_TOKENS,
                                n_branches=3, entropy_threshold=2.0):
    """Branch when token entropy is high."""
    beams = [(tokenizer.encode(prompt, return_tensors="pt").to(DEVICE), 0.0)]
    n_fwd = 0

    for step in range(max_tokens):
        new_beams = []
        for input_ids, score in beams:
            with torch.no_grad():
                outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).item()
            log_probs = F.log_softmax(logits, dim=-1)
            n_fwd += 1

            if entropy > entropy_threshold:
                topk = torch.topk(log_probs.squeeze(), n_branches)
                for tok, lp in zip(topk.indices, topk.values):
                    new_ids = torch.cat([input_ids, tok.unsqueeze(0).unsqueeze(0)], dim=-1)
                    new_beams.append((new_ids, score + lp.item()))
            else:
                next_tok = logits.argmax(dim=-1, keepdim=True)
                lp = log_probs.squeeze()[next_tok.squeeze()].item()
                new_ids = torch.cat([input_ids, next_tok], dim=-1)
                new_beams.append((new_ids, score + lp))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:n_branches]

        # check if best beam has finished
        best_text = tokenizer.decode(beams[0][0].squeeze()[-30:])
        if "answer is" in best_text and ("." in best_text.split("answer is")[-1] or
                                          "\n" in best_text.split("answer is")[-1]):
            break

    best = max(beams, key=lambda x: x[1])
    text = tokenizer.decode(best[0].squeeze(), skip_special_tokens=True)
    return text, n_fwd


def generate_density_branching(model, tokenizer, prompt, gmm, density_threshold,
                                max_tokens=MAX_GEN_TOKENS, n_branches=3):
    """Branch when latent state density is low (OOD reasoning state)."""
    beams = [(tokenizer.encode(prompt, return_tensors="pt").to(DEVICE), 0.0)]
    n_fwd = 0
    newline_id = tokenizer.encode("\n")[0]

    for step in range(max_tokens):
        new_beams = []
        for input_ids, score in beams:
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            n_fwd += 1

            # check latent density at reasoning step boundaries
            last_token = input_ids[:, -1].item()
            should_branch = False
            if last_token == newline_id:
                hidden = outputs.hidden_states[LATENT_LAYER][:, -1, :].cpu().numpy()
                density_score = gmm.score_samples(hidden)[0]
                should_branch = density_score < density_threshold

            if should_branch:
                topk = torch.topk(log_probs.squeeze(), n_branches)
                for tok, lp in zip(topk.indices, topk.values):
                    new_ids = torch.cat([input_ids, tok.unsqueeze(0).unsqueeze(0)], dim=-1)
                    new_beams.append((new_ids, score + lp.item()))
            else:
                next_tok = logits.argmax(dim=-1, keepdim=True)
                lp = log_probs.squeeze()[next_tok.squeeze()].item()
                new_ids = torch.cat([input_ids, next_tok], dim=-1)
                new_beams.append((new_ids, score + lp))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:n_branches]

        best_text = tokenizer.decode(beams[0][0].squeeze()[-30:])
        if "answer is" in best_text and ("." in best_text.split("answer is")[-1] or
                                          "\n" in best_text.split("answer is")[-1]):
            break

    best = max(beams, key=lambda x: x[1])
    text = tokenizer.decode(best[0].squeeze(), skip_special_tokens=True)
    return text, n_fwd


# ── Answer extraction ────────────────────────────────────────────────────────

def extract_answer(text):
    """Pull the final number from 'The answer is X.'"""
    import re
    matches = re.findall(r"answer is[:\s]*(-?\d+)", text.lower())
    if matches:
        return int(matches[-1])
    numbers = re.findall(r"(-?\d+)", text.split("\n")[-1])
    if numbers:
        return int(numbers[-1])
    return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
    model.eval()

    # generate problems
    correct_examples = generate_math_problems(N_CORRECT_EXAMPLES, rng=np.random.RandomState(SEED))
    test_problems = generate_math_problems(N_TEST, rng=np.random.RandomState(SEED + 1000))

    # build latent density model from correct traces
    gmm, density_threshold = build_density_model(model, tokenizer, correct_examples)

    # calibrate entropy threshold
    print("\nCalibrating entropy threshold...")
    entropies = []
    for ex in correct_examples[:50]:
        inputs = tokenizer(ex["full"], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        ent = -(probs * (probs + 1e-10).log()).sum(dim=-1).squeeze()
        entropies.extend(ent.cpu().numpy().tolist())
    entropies = np.array(entropies)
    entropy_threshold = np.percentile(entropies, 80)
    print(f"  Entropy p80={entropy_threshold:.2f}")

    # run test
    strategies = {
        "Greedy": lambda p: generate_greedy(model, tokenizer, p),
        "Entropy Branch": lambda p: generate_entropy_branching(
            model, tokenizer, p, entropy_threshold=entropy_threshold
        ),
        "Density Branch": lambda p: generate_density_branching(
            model, tokenizer, p, gmm, density_threshold
        ),
    }

    results = {name: {"correct": 0, "total": 0, "fwd": 0} for name in strategies}

    for i, prob in enumerate(test_problems):
        print(f"\nProblem {i+1}/{N_TEST}: answer={prob['answer']}")
        for name, gen_fn in strategies.items():
            t0 = time.perf_counter()
            text, n_fwd = gen_fn(prob["prompt"])
            dt = time.perf_counter() - t0
            pred = extract_answer(text)
            correct = pred == prob["answer"]
            results[name]["correct"] += int(correct)
            results[name]["total"] += 1
            results[name]["fwd"] += n_fwd
            mark = "OK" if correct else "WRONG"
            print(f"  {name:18s}  pred={pred}  {mark}  fwd={n_fwd}  time={dt:.1f}s")

    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"{'Method':20s} {'Accuracy':>10s} {'Avg FWD':>10s}")
    print("-" * 60)
    for name, r in results.items():
        acc = r["correct"] / r["total"]
        avg_fwd = r["fwd"] / r["total"]
        print(f"{name:20s} {acc:10.3f} {avg_fwd:10.1f}")
    print("=" * 60)

    # save results
    with open("results.json", "w") as f:
        json.dump({
            name: {
                "accuracy": r["correct"] / r["total"],
                "avg_forward_passes": r["fwd"] / r["total"],
                "total_correct": r["correct"],
                "total": r["total"],
            }
            for name, r in results.items()
        }, f, indent=2)
    print("\nSaved results.json")


if __name__ == "__main__":
    main()
