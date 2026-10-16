"""Temperature-controlled autoregressive generation and generated-text metrics."""
from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F

from .data import PUNCT, detokenize, preprocess_seed


@torch.no_grad()
def generate(model, vocab, seed: str, n_words: int = 50, temperature: float = 1.0,
             seq_len: int = 20, generator: torch.Generator | None = None,
             device: str = "cpu") -> dict:
    """Generate `n_words` tokens one at a time, feeding each prediction back as input.

    temperature == 0 -> greedy (argmax) decoding; otherwise sample from
    softmax(logits / temperature). <pad> and <unk> are never sampled.
    """
    model.eval()
    seed_tokens = preprocess_seed(seed)
    ids = vocab.encode(seed_tokens)
    banned = torch.tensor([vocab.pad_id, vocab.unk_id])
    new_ids = []
    for _ in range(n_words):
        ctx = torch.tensor([ids[-seq_len:]], dtype=torch.long, device=device)
        logits = model(ctx)[0].cpu()
        logits[banned] = -float("inf")
        if temperature == 0:
            nxt = int(logits.argmax())
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=generator))
        ids.append(nxt)
        new_ids.append(nxt)
    gen_tokens = vocab.decode(new_ids)
    return {"seed": seed, "seed_tokens": seed_tokens, "generated_tokens": gen_tokens,
            "text": detokenize(seed_tokens + gen_tokens),
            "continuation": detokenize(gen_tokens)}


# --------------------------------------------------------------------------- #
# Generated-text metrics
# --------------------------------------------------------------------------- #
def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(tokens, n):
    g = ngrams(tokens, n)
    return len(set(g)) / len(g) if g else 0.0


def repetition_rate(tokens, n=3):
    """Fraction of n-grams that already occurred earlier in the same text."""
    g = ngrams(tokens, n)
    if not g:
        return 0.0
    seen, rep = set(), 0
    for x in g:
        rep += x in seen
        seen.add(x)
    return rep / len(g)


class CorpusReference:
    """n-gram sets from the training corpus used for fluency / copy-rate measures."""

    def __init__(self, train_tokens):
        self.bigrams = set(ngrams(train_tokens, 2))
        self.fourgrams = set(ngrams(train_tokens, 4))

    def bigram_fluency(self, tokens):
        """Share of generated bigrams that are attested in the training text (local coherence)."""
        g = ngrams(tokens, 2)
        return sum(x in self.bigrams for x in g) / len(g) if g else 0.0

    def copy_rate(self, tokens):
        """Share of generated 4-grams copied verbatim from the training text (memorisation)."""
        g = ngrams(tokens, 4)
        return sum(x in self.fourgrams for x in g) / len(g) if g else 0.0


def text_metrics(tokens, seed_tokens, ref: CorpusReference) -> dict:
    """Metrics computed on the generated continuation (bigram fluency includes the seed join)."""
    words = [t for t in tokens if t not in PUNCT]
    sents = sum(t in {".", "!", "?"} for t in tokens)
    joined = seed_tokens[-1:] + tokens
    return {
        "distinct_1": distinct_n(tokens, 1),
        "distinct_2": distinct_n(tokens, 2),
        "rep_3": repetition_rate(tokens, 3),
        "bigram_fluency": ref.bigram_fluency(joined),
        "copy_rate_4": ref.copy_rate(tokens),
        "max_token_share": Counter(words).most_common(1)[0][1] / len(words) if words else 0.0,
        "avg_sentence_len": len(tokens) / max(sents, 1),
    }
