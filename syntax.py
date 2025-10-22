from dataclasses import dataclass
from typing import Dict, Optional, List, Set

import torch


@dataclass(frozen=True)
class LatexState:
    """Minimal state to enforce well-formed LaTeX during decoding."""
    brace_depth: int = 0           # { ... }
    bracket_depth: int = 0         # [ ... ]
    # '^' or '_' just emitted → next must be '{'
    need_group_after: Optional[str] = None
    frac_need: int = 0             # 0 none, 1 need {num}, 2 need {den}
    # after \sqrt: next may be '[' or must be '{' if '[' not taken
    sqrt_optional: bool = False
    began: bool = False            # we’ve produced at least one non-BOS token


class LatexConstraints:
    """
    Hard constraints for LaTeX tokens, applied BEFORE softmax.
    You must call mask_invalid(logits, state) at every step in beam search.

    Expected tokens in vocab:
      specials: <pad>, <bos>/<sos>, <eos>, <unk>
      delimiters: '{', '}', '[', ']', '^', '_'
      macros: '\\frac', '\\sqrt'
    """

    def __init__(self, token_to_id: Dict[str, int]) -> None:
        self.v2i = token_to_id

        # Resolve specials (be permissive with names)
        self.PAD = self._find_first(["<pad>", "<PAD>"])
        self.BOS = self._find_first(["<bos>", "<sos>", "<BOS>", "<SOS>"])
        self.EOS = self._find_first(["<eos>", "<EOS>"])
        self.UNK = self._find_first(["<unk>", "<UNK>"])

        # Structure tokens
        self.tLBRACE = self.v2i.get("{")
        self.tRBRACE = self.v2i.get("}")
        self.tLBRACK = self.v2i.get("[")
        self.tRBRACK = self.v2i.get("]")
        self.tHAT = self.v2i.get("^")
        self.tUND = self.v2i.get("_")

        # Common structural macros seen in your data
        # e.g., "\frac { 1 } { 2 }"  :contentReference[oaicite:2]{index=2}
        self.tFRAC = self.v2i.get("\\frac")
        # e.g., "\sqrt { 3 }"         :contentReference[oaicite:3]{index=3}
        self.tSQRT = self.v2i.get("\\sqrt")

        # Cache useful sets
        self.struct_ids: Set[int] = {
            i for i in [
                self.tLBRACE, self.tRBRACE, self.tLBRACK, self.tRBRACK,
                self.tHAT, self.tUND, self.tFRAC, self.tSQRT
            ] if i is not None
        }

    # ---- public API ---------------------------------------------------------
    def start(self) -> LatexState:
        return LatexState()

    def step(self, st: LatexState, token_id: int) -> LatexState:
        """Advance the syntax state after emitting token_id."""
        # Close braces/brackets if applicable
        if token_id == self.tRBRACE and st.brace_depth > 0:
            return LatexState(
                brace_depth=st.brace_depth - 1,
                bracket_depth=st.bracket_depth,
                need_group_after=None,                  # a group has been provided
                # satisfy one of the {num}/{den}
                frac_need=max(0, st.frac_need - 1),
                sqrt_optional=False,
                began=True
            )
        if token_id == self.tRBRACK and st.bracket_depth > 0:
            return LatexState(
                brace_depth=st.brace_depth,
                bracket_depth=st.bracket_depth - 1,
                need_group_after=st.need_group_after,
                frac_need=st.frac_need,
                sqrt_optional=False,
                began=True
            )

        # Open braces/brackets
        if token_id == self.tLBRACE:
            return LatexState(
                brace_depth=st.brace_depth + 1,
                bracket_depth=st.bracket_depth,
                need_group_after=None if st.need_group_after else st.need_group_after,
                frac_need=st.frac_need,
                sqrt_optional=False,
                began=True
            )
        if token_id == self.tLBRACK:
            # If we were right after \sqrt, taking '[' consumes the optional part
            return LatexState(
                brace_depth=st.brace_depth,
                bracket_depth=st.bracket_depth + 1,
                need_group_after=st.need_group_after,
                frac_need=st.frac_need,
                # optional now taken; later must still see a '{'
                sqrt_optional=False,
                began=True
            )

        # Operators that require a brace group next
        if token_id == self.tHAT:
            return LatexState(
                brace_depth=st.brace_depth,
                bracket_depth=st.bracket_depth,
                need_group_after="^",
                frac_need=st.frac_need,
                sqrt_optional=False,
                began=True
            )
        if token_id == self.tUND:
            return LatexState(
                brace_depth=st.brace_depth,
                bracket_depth=st.bracket_depth,
                need_group_after="_",
                frac_need=st.frac_need,
                sqrt_optional=False,
                began=True
            )

        # Macros that enforce structure
        if token_id == self.tFRAC:
            # require two subsequent '{...}' groups: {num}{den}
            return LatexState(
                brace_depth=st.brace_depth,
                bracket_depth=st.bracket_depth,
                need_group_after=None,
                frac_need=2,
                sqrt_optional=False,
                began=True
            )
        if token_id == self.tSQRT:
            # allow optional [..] once, then require '{...}'
            return LatexState(
                brace_depth=st.brace_depth,
                bracket_depth=st.bracket_depth,
                need_group_after=None,
                frac_need=0,
                sqrt_optional=True,
                began=True
            )

        # Default payload token
        return LatexState(
            brace_depth=st.brace_depth,
            bracket_depth=st.bracket_depth,
            # cleared only once a '{' is opened
            need_group_after=st.need_group_after,
            frac_need=st.frac_need,
            sqrt_optional=False if st.sqrt_optional else False,
            began=True
        )

    def mask_invalid(self, logits: torch.Tensor, st: LatexState) -> torch.Tensor:
        """
        Hard-mask invalid next tokens by setting logits to -inf (before softmax).
        logits: (V,) on the correct device.
        """
        V = logits.size(0)
        mask = torch.zeros(V, dtype=torch.bool,
                           device=logits.device)  # True => forbid

        # Never emit PAD
        if self.PAD is not None:
            mask[self.PAD] = True

        # Can't end if any group is still open
        if self.EOS is not None and (st.brace_depth > 0 or st.bracket_depth > 0 or st.frac_need > 0):
            mask[self.EOS] = True

        # Can't close when nothing open
        if st.brace_depth == 0 and self.tRBRACE is not None:
            mask[self.tRBRACE] = True
        if st.bracket_depth == 0 and self.tRBRACK is not None:
            mask[self.tRBRACK] = True

        # Immediately after '^' or '_' → must start a '{'
        if st.need_group_after is not None:
            self._mask_all_but(mask, [self.tLBRACE])
            return logits.masked_fill(mask, float("-inf"))

        # After \frac → next must be '{' until two brace groups are seen
        if st.frac_need > 0:
            self._mask_all_but(mask, [self.tLBRACE])
            return logits.masked_fill(mask, float("-inf"))

        # After \sqrt:
        #  - if optional not yet taken: next may be '[' or '{'
        #  - if '[' taken, regular bracket rules already apply; later we still need a '{'
        if st.sqrt_optional:
            allow = [x for x in [self.tLBRACK, self.tLBRACE] if x is not None]
            if allow:
                self._mask_all_but(mask, allow)
                return logits.masked_fill(mask, float("-inf"))

        # Do not allow stacked '^' or '_' (must have a group between)
        if self.tHAT is not None and st.need_group_after is not None:
            mask[self.tHAT] = True
        if self.tUND is not None and st.need_group_after is not None:
            mask[self.tUND] = True

        # At the very beginning (no content yet), discourage immediate closers/ops
        if not st.began:
            for forbid in [self.tRBRACE, self.tRBRACK, self.tHAT, self.tUND]:
                if forbid is not None:
                    mask[forbid] = True
            # Allow EOS at start only if you really want empty output (usually not)
            if self.EOS is not None:
                mask[self.EOS] = True

        return logits.masked_fill(mask, float("-inf"))

    # ---- helpers ------------------------------------------------------------
    def _find_first(self, keys: List[str]) -> Optional[int]:
        for k in keys:
            if k in self.v2i:
                return self.v2i[k]
        return None

    @staticmethod
    def _mask_all_but(mask: torch.Tensor, allow: List[Optional[int]]) -> None:
        keep = [i for i in allow if i is not None]
        mask[:] = True
        if keep:
            mask[keep] = False
