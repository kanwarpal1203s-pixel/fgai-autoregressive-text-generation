"""Data loading, cleaning, tokenisation, vocabulary and sequence construction.

Dataset: "Pride and Prejudice" by Jane Austen, Project Gutenberg eBook #1342
Source : https://www.gutenberg.org/ebooks/1342
"""
from __future__ import annotations

import re
import urllib.request
from collections import Counter
from pathlib import Path

import torch

DATASET_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
DATASET_PAGE = "https://www.gutenberg.org/ebooks/1342"

START_MARKER = "It is a truth universally acknowledged"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"

# words + apostrophe contractions (don't, elizabeth's) and selected punctuation
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*|[.,!?;:]")
PUNCT = set(".,!?;:")
SENT_END = set(".!?")


# --------------------------------------------------------------------------- #
# Loading / cleaning
# --------------------------------------------------------------------------- #
def download_dataset(path: str | Path) -> Path:
    """Download the raw Gutenberg text if it is not present locally."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATASET_URL, path)
    return path


def load_raw(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def strip_gutenberg(raw: str) -> str:
    """Keep only the novel itself (drop licence, front matter, illustrations list)."""
    start = raw.find(START_MARKER)
    end = raw.find(END_MARKER)
    if start < 0 or end < 0:
        raise ValueError("Could not locate start/end markers in the raw text")
    body = raw[start:end]
    colophon = body.find("CHISWICK PRESS")  # printer's imprint after the last sentence
    return body[:colophon] if colophon > 0 else body


def remove_illustrations(text: str) -> str:
    """Remove '[Illustration ...]' blocks, which may span lines and nest brackets."""
    out, depth, i, tag = [], 0, 0, "[Illustration"
    while i < len(text):
        if depth == 0 and text.startswith(tag, i):
            depth, i = 1, i + len(tag)
            continue
        if depth > 0:
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
            i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def clean_text(text: str) -> str:
    """Normalise the novel text into a single lower-cased whitespace-separated string."""
    text = remove_illustrations(text)
    text = re.sub(r"(?m)^\s*CHAPTER\s+[IVXLC]+\.?\s*$", " ", text)  # chapter headings
    text = text.replace("_", "")  # _italics_ markers
    text = text.translate(str.maketrans({
        "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
        "\u2014": " ", "\u2013": " ", "-": " ",
    }))
    text = text.lower()
    # abbreviations: keep "mr." etc. as one word so '.' remains a sentence boundary
    text = re.sub(r"\b(mr|mrs|ms|dr|st)\.", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def preprocess_seed(seed: str) -> list[str]:
    """Apply the same cleaning + tokenisation used for the corpus to a seed string."""
    return tokenize(clean_text(seed))


def detokenize(tokens: list[str]) -> str:
    """Turn tokens back into readable text (spacing + capitalisation)."""
    words, cap_next = [], True
    for t in tokens:
        if t in PUNCT:
            if words:
                words[-1] += t
            else:
                words.append(t)
            cap_next = t in SENT_END
            continue
        w = t
        if w == "i" or w.startswith("i'"):
            w = "I" + w[1:]
        if w in {"mr", "mrs", "dr"}:
            w = w.capitalize() + "."
        if cap_next:
            w = w[0].upper() + w[1:]
            cap_next = False
        words.append(w)
    return " ".join(words)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class Vocab:
    PAD, UNK = "<pad>", "<unk>"

    def __init__(self, tokens: list[str], min_freq: int = 2):
        counter = Counter(tokens)
        self.counter = counter
        self.min_freq = min_freq
        self.itos = [self.PAD, self.UNK] + [w for w, c in counter.most_common() if c >= min_freq]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id, self.unk_id = self.stoi[self.PAD], self.stoi[self.UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def decode(self, ids) -> list[str]:
        return [self.itos[int(i)] for i in ids]


# --------------------------------------------------------------------------- #
# Input-target sequence construction
# --------------------------------------------------------------------------- #
def make_sequences(ids: list[int], seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sliding window (stride 1): X[i] = ids[i : i+seq_len], y[i] = ids[i+seq_len].

    This is the autoregressive next-word formulation p(w_t | w_{t-n}, ..., w_{t-1}).
    """
    t = torch.tensor(ids, dtype=torch.long)
    X = t.unfold(0, seq_len, 1)[:-1].contiguous()
    y = t[seq_len:].contiguous()
    assert len(X) == len(y)
    return X, y


def split_tokens(tokens: list[str], train_frac=0.8, val_frac=0.1):
    """Contiguous split so that no overlapping window leaks across splits."""
    n = len(tokens)
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    return tokens[:a], tokens[a:b], tokens[b:]


def prepare_data(raw_path: str | Path, seq_len: int = 20, min_freq: int = 2):
    """Full preprocessing pipeline. Returns a dict with everything needed downstream."""
    raw = load_raw(download_dataset(raw_path))
    body = strip_gutenberg(raw)
    text = clean_text(body)
    tokens = tokenize(text)
    tr_tok, va_tok, te_tok = split_tokens(tokens)
    vocab = Vocab(tr_tok, min_freq=min_freq)  # vocabulary from training split only
    splits = {}
    for name, toks in (("train", tr_tok), ("val", va_tok), ("test", te_tok)):
        splits[name] = make_sequences(vocab.encode(toks), seq_len)
    return {
        "raw": raw, "body": body, "text": text, "tokens": tokens,
        "train_tokens": tr_tok, "val_tokens": va_tok, "test_tokens": te_tok,
        "vocab": vocab, "splits": splits, "seq_len": seq_len,
    }


def dataset_statistics(data: dict) -> dict:
    body, tokens, vocab = data["body"], data["tokens"], data["vocab"]
    words = [t for t in tokens if t not in PUNCT]
    sentences = [s for s in re.split(r"[.!?]+", data["text"]) if s.strip()]
    sent_lens = [len(tokenize(s)) for s in sentences]
    unk_rate = {
        k: sum(1 for t in data[f"{k}_tokens"] if t not in vocab.stoi) / len(data[f"{k}_tokens"])
        for k in ("train", "val", "test")
    }
    return {
        "raw_characters": len(data["raw"]),
        "novel_characters": len(body),
        # +1: "Chapter I" heading sits just before the start marker
        "chapters": len(re.findall(r"(?i)\bchapter\s+[ivxlc]+\b", body)) + 1,
        "paragraphs": len([p for p in re.split(r"\n\s*\n", body) if p.strip()]),
        "sentences": len(sentences),
        "avg_sentence_length_tokens": sum(sent_lens) / len(sent_lens),
        "total_tokens_incl_punct": len(tokens),
        "word_tokens": len(words),
        "unique_words": len(set(words)),
        "hapax_legomena": sum(1 for c in Counter(words).values() if c == 1),
        "vocab_size_min_freq": len(vocab),
        "min_freq": vocab.min_freq,
        "train_tokens": len(data["train_tokens"]),
        "val_tokens": len(data["val_tokens"]),
        "test_tokens": len(data["test_tokens"]),
        "train_sequences": len(data["splits"]["train"][1]),
        "val_sequences": len(data["splits"]["val"][1]),
        "test_sequences": len(data["splits"]["test"][1]),
        "unk_rate": unk_rate,
        "sentence_lengths": sent_lens,
    }
