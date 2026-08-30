"""Topic 0 — Word2Vec (skip-gram with negative sampling) in PyTorch.

Written out rather than called from a library. Two reasons, and the second is the
one that matters:

  1. gensim has no wheel for Python 3.14 yet and will not build here.
  2. This is the model whose learned geometry the report has to explain. Writing
     the objective means the nearest-neighbour table in the results chapter is
     something you can account for line by line.

**Skip-gram**: given a centre word, predict the words around it. The opposite of
CBOW, and the better choice on a small corpus because each (centre, context) pair
is its own training example, so a few hundred thousand tokens still yield
millions of updates.

**Negative sampling**: the true objective is a softmax over the whole vocabulary,
which is far too slow. Instead, for each real (centre, context) pair, draw `k`
fake context words and train a binary classifier: real pairs score high, fake
pairs score low. The fakes are drawn from the unigram distribution raised to the
power 0.75, which pulls probability mass away from very frequent words without
removing them.

The result is two embedding tables. The centre-word table is what we keep; the
context table is scaffolding and gets discarded, which is what the original paper
does too.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import Vocab


class SkipGramNegativeSampling(nn.Module):
    """Two embedding tables and a binary objective over real vs sampled pairs."""

    def __init__(self, vocab_size: int, dim: int = 200):
        super().__init__()
        self.centre = nn.Embedding(vocab_size, dim)
        self.context = nn.Embedding(vocab_size, dim)
        # Small init on the centre table, zeros on the context table — the
        # asymmetry is from the original implementation and keeps early scores
        # near zero, so the sigmoid starts in its responsive range instead of
        # saturated.
        nn.init.uniform_(self.centre.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.context.weight)

    def forward(
        self, centre: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        """centre (B,), positive (B,), negative (B, k) -> scalar loss."""
        c = self.centre(centre)                       # (B, D)
        p = self.context(positive)                    # (B, D)
        n = self.context(negative)                    # (B, k, D)

        pos_score = (c * p).sum(dim=1)                # (B,)
        neg_score = torch.bmm(n, c.unsqueeze(2)).squeeze(2)  # (B, k)

        # -log σ(c·p) - Σ log σ(-c·n), averaged over the batch.
        pos_loss = F.logsigmoid(pos_score)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)
        return -(pos_loss + neg_loss).mean()


class SkipGramData:
    """Turns tokenized documents into (centre, context) pairs.

    Two standard tricks, both of which matter more on a small corpus:

    - **Subsampling of frequent words.** A token is kept with probability
      1 - sqrt(t / f). Very common words are dropped often, which both speeds
      training and stops the frequent-word neighbourhoods from swamping
      everything.
    - **Dynamic window.** The window for each centre word is drawn uniformly from
      1..window, which weights nearby words more heavily without a separate
      weighting term.
    """

    def __init__(
        self,
        docs: list[list[str]],
        vocab: Vocab,
        window: int = 5,
        subsample_t: float = 1e-3,
        seed: int = 42,
    ):
        self.vocab = vocab
        self.window = window
        self.rng = random.Random(seed)

        counts = torch.zeros(len(vocab))
        encoded = []
        for doc in docs:
            ids = [vocab.stoi[t] for t in doc if t in vocab.stoi]
            if len(ids) >= 2:
                encoded.append(ids)
                for i in ids:
                    counts[i] += 1
        self.counts = counts
        total = counts.sum().clamp(min=1)
        freq = counts / total
        # Keep-probability per token id; specials are never emitted anyway.
        with torch.no_grad():
            keep = (subsample_t / freq.clamp(min=1e-12)).sqrt().clamp(max=1.0)
        self.keep = keep
        self.docs = encoded

        # Negative-sampling distribution: unigram ** 0.75.
        weights = counts.pow(0.75)
        weights[vocab.pad_id] = 0.0
        weights[vocab.unk_id] = 0.0
        self.neg_weights = weights

    def pairs(self) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        keep = self.keep
        for ids in self.docs:
            kept = [i for i in ids if self.rng.random() < keep[i].item()]
            for pos, centre in enumerate(kept):
                w = self.rng.randint(1, self.window)
                lo, hi = max(0, pos - w), min(len(kept), pos + w + 1)
                for j in range(lo, hi):
                    if j != pos:
                        out.append((centre, kept[j]))
        self.rng.shuffle(out)
        return out


def train(
    docs: list[list[str]],
    vocab: Vocab,
    dim: int = 200,
    window: int = 5,
    epochs: int = 5,
    batch_size: int = 1024,
    negatives: int = 10,
    lr: float = 2.5e-3,
    seed: int = 42,
    device: str = "cpu",
    log_every: int = 200,
) -> tuple[SkipGramNegativeSampling, dict]:
    torch.manual_seed(seed)
    data = SkipGramData(docs, vocab, window=window, seed=seed)
    model = SkipGramNegativeSampling(len(vocab), dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    neg_weights = data.neg_weights.to(device)

    history: dict = {"epochs": [], "pairs_per_epoch": []}
    for epoch in range(1, epochs + 1):
        # Re-sampled every epoch: subsampling and the dynamic window are random,
        # so each pass sees a different view of the same corpus.
        pairs = data.pairs()
        history["pairs_per_epoch"].append(len(pairs))
        if not pairs:
            raise RuntimeError("no training pairs — corpus too small or all subsampled")

        centres = torch.tensor([c for c, _ in pairs], dtype=torch.long)
        contexts = torch.tensor([x for _, x in pairs], dtype=torch.long)
        total_loss, n_batches = 0.0, 0

        for start in range(0, len(pairs), batch_size):
            c = centres[start : start + batch_size].to(device)
            p = contexts[start : start + batch_size].to(device)
            neg = torch.multinomial(
                neg_weights, c.size(0) * negatives, replacement=True
            ).view(c.size(0), negatives)

            loss = model(c, p, neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            if log_every and n_batches % log_every == 0:
                print(
                    f"  epoch {epoch} batch {n_batches:5d}/"
                    f"{math.ceil(len(pairs) / batch_size)}  "
                    f"loss {total_loss / n_batches:.4f}"
                )

        mean_loss = total_loss / max(n_batches, 1)
        history["epochs"].append({"epoch": epoch, "loss": mean_loss, "pairs": len(pairs)})
        print(f"epoch {epoch}: {len(pairs):,} pairs, mean loss {mean_loss:.4f}")

    return model, history


@torch.no_grad()
def most_similar(
    model: SkipGramNegativeSampling, vocab: Vocab, term: str, topn: int = 8
) -> list[tuple[str, float]]:
    if term not in vocab.stoi:
        return []
    weights = F.normalize(model.centre.weight, dim=1)
    query = weights[vocab.stoi[term]]
    scores = weights @ query
    scores[vocab.stoi[term]] = -1e9
    for special in (vocab.pad_id, vocab.unk_id):
        scores[special] = -1e9
    top = torch.topk(scores, topn)
    return [(vocab.itos[i], float(s)) for s, i in zip(top.values, top.indices)]
