from typing import Dict, List, Optional, Tuple
import math

import torch

# Global constraint object (set once via set_constraints_vocab)
_CONSTRAINTS = None  # type: ignore[attr-defined]


def set_constraints_vocab(token_to_id: Dict[str, int]) -> None:
    """
    Initialize global constraints with the given vocabulary.
    Call this once before decoding (e.g., in evaluate.py).
    """
    global _CONSTRAINTS
    from syntax import LatexConstraints  # your syntax.py
    _CONSTRAINTS = LatexConstraints(token_to_id)


@torch.no_grad()
def beam_search(
    model,
    q_llm: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    beam: int = 5,
    max_len: int = 200,
    alpha: float = 0.6,
    temperature: float = 1.0,
) -> torch.LongTensor:
    """
    Constrained beam search.
    - model.decoder(tgt_ids, q_llm) -> logits of shape (B, T, V)
    - q_llm is the visual prompt (B, Q, H) already in the model/LLM dtype/device
    - returns: best sequence (1, L) LongTensor including BOS ... EOS (if reached)

    Hard constraints:
      - We call constraints.mask_invalid(logits, state) BEFORE softmax each step.
      - State is tracked per-beam and advanced with constraints.step(state, token).
    Length normalization:
      - score_norm = score_sum / (len(content) ** alpha), content excludes BOS.
    """
    device = q_llm.device
    vocab_size = _infer_vocab_size(model)

    # Initialize one-beam with BOS
    init_seq = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    # (seq, score_sum, state)
    beams: List[Tuple[torch.LongTensor, float, object]] = []
    completed: List[Tuple[torch.LongTensor, float]
                    ] = []      # (seq, score_sum)

    st0 = _constraints_start(bos_id)  # initial syntax state after BOS
    beams.append((init_seq, 0.0, st0))

    for step in range(1, max_len + 1):
        cand: List[Tuple[torch.LongTensor, float, object]] = []

        for seq, score_sum, state in beams:
            # If already ended with EOS, keep it in completed
            if seq[0, -1].item() == eos_id:
                completed.append((seq, score_sum))
                continue

            # Run decoder for current beam (inefficient but simple & safe)
            # logits: (1, t, V)
            logits = model.decoder(seq, q_llm)
            next_logits = logits[:, -1, :].squeeze(0)  # (V,)

            # Temperature
            if temperature != 1.0:
                next_logits = next_logits / float(temperature)

            # Apply hard constraints before softmax
            next_logits = _mask_constrained(next_logits, state, pad_id)

            # Safety fallback: if everything is -inf (dead-end), allow EOS
            if not torch.isfinite(next_logits).any():
                next_logits = torch.full_like(next_logits, float("-inf"))
                next_logits[eos_id] = 0.0

            log_probs = torch.log_softmax(next_logits, dim=-1)  # (V,)

            # Take top-k for this beam
            topk_logp, topk_idx = torch.topk(
                log_probs, k=min(beam, vocab_size))

            for lp, tok in zip(topk_logp.tolist(), topk_idx.tolist()):
                new_seq = torch.cat(
                    [seq, torch.tensor([[tok]], device=device, dtype=torch.long)], dim=1)
                new_score_sum = score_sum + lp
                new_state = _constraints_step(state, tok)
                cand.append((new_seq, new_score_sum, new_state))

        # If no candidates (all beams completed), stop
        if not cand:
            break

        # Rank candidates by length-normalized score
        # length here excludes BOS (i.e., content length)
        def _norm_key(item):
            s, sc, _ = item
            content_len = max(1, s.size(1) - 1)
            return sc / (content_len ** alpha)

        cand.sort(key=_norm_key, reverse=True)

        # Keep top global beams
        beams = cand[:beam]

        # Early stop if we have enough completed with EOS and they are better than open beams
        # (Optional: left simple—search continues to max_len or until no open beams remain)

    # If we ended with no completed sequences, choose the best current beam
    if not completed:
        best_seq, best_score, _ = max(
            beams,
            key=lambda b: b[1] / (max(1, b[0].size(1) - 1) ** alpha)
        )
        return best_seq

    # Otherwise, pick the best completed (length-normalized)
    best_seq, _ = max(
        completed,
        key=lambda item: item[1] / (max(1, item[0].size(1) - 1) ** alpha)
    )
    return best_seq


# -------------------- helpers --------------------

def _infer_vocab_size(model) -> int:
    # expects model.decoder.out_head is Linear(H, V) or equivalent
    head = getattr(model.decoder, "out_head", None)
    if head is None or not hasattr(head, "out_features"):
        raise RuntimeError(
            "Cannot infer vocab size from model.decoder.out_head")
    return int(head.out_features)


def _constraints_start(bos_id: int):
    """Start constraint state; advance with BOS so 'began' is true."""
    if _CONSTRAINTS is None:
        return None
    st = _CONSTRAINTS.start()
    # Advance state by BOS to mark that we've begun (avoids forbidding early tokens).
    try:
        st = _CONSTRAINTS.step(st, bos_id)
    except Exception:
        pass
    return st


def _constraints_step(state, token_id: int):
    if _CONSTRAINTS is None:
        return None
    return _CONSTRAINTS.step(state, token_id)


def _mask_constrained(next_logits: torch.Tensor, state, pad_id: int) -> torch.Tensor:
    """
    Apply constraint masking. Also blocks PAD.
    If no constraints are set, at least block PAD explicitly.
    """
    logits = next_logits.clone()
    # Always block PAD
    if 0 <= pad_id < logits.numel():
        logits[pad_id] = float("-inf")

    if _CONSTRAINTS is None:
        return logits

    try:
        logits = _CONSTRAINTS.mask_invalid(logits, state)
    except Exception:
        # If constraints fail for any reason, fall back to PAD-only mask
        pass
    return logits
