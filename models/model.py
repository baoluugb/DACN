import torch
import torch.nn as nn
from .qformer import QFormer
from .decoder import LLMDecoderLoRA
from .decoder import LLMDecoder
from .encoder import Encoder


class HMER_BLIP2(nn.Module):
    """Encoder -> Q-Former -> LLM decoder"""
    def __init__(self,
        latex_vocab_size: int,
        d_model: int = 256,
        qformer_dim: int = 256,
        qformer_heads: int = 8,
        qformer_layers: int = 4,
        num_queries: int = 16,
        llm_name: str = "gpt2",
        use_8bit: bool = False,
    ):
        super().__init__()
        self.image_encoder = Encoder(d_model)
        self.qformer = QFormer(
            d_model=d_model,
            query_dim=qformer_dim,
            num_heads=qformer_heads, 
            num_layers=qformer_layers,
            num_queries=num_queries
        )
        

        # Import the regular decoder without LoRA
        
        self.decoder = LLMDecoder(
            qformer_dim = qformer_dim,
            latex_vocab_size=latex_vocab_size,
            llm_name=llm_name,
            num_prompt_tokens=num_queries,
            use_8bit=use_8bit,
        )

    def forward(self, images: torch.Tensor, tgt_ids: torch.Tensor):
        visual_seq, _ = self.image_encoder(images)  # (B,S,Dv)
        q_feats = self.qformer(visual_seq)          # (B,Q,Dq)
        logits = self.decoder(tgt_ids, q_feats)       # (B,T,V)
        return logits




class HMER_BLIP2_LoRA(nn.Module):
    """Encoder -> Q-Former -> LoRA-LLM decoder"""
    def __init__(self,
        latex_vocab_size: int,
        d_model: int = 256,
        qformer_dim: int = 256,
        qformer_heads: int = 8,
        qformer_layers: int = 4,
        num_queries: int = 16,
        llm_name: str = "gpt2",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules=None,
        use_8bit: bool = False,
    ):
        super().__init__()
        self.image_encoder = Encoder(d_model)

        self.qformer = QFormer(
            d_model=d_model,
            query_dim=qformer_dim,
            num_heads=qformer_heads, 
            num_layers=qformer_layers,
            num_queries=num_queries
        )

        self.decoder = LLMDecoderLoRA(
            qformer_dim=qformer_dim,
            latex_vocab_size=latex_vocab_size,
            llm_name=llm_name,
            num_prompt_tokens=num_queries,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            use_8bit=use_8bit,
        )

    def forward(self, images: torch.Tensor, tgt_ids: torch.Tensor):
        visual_seq, _ = self.image_encoder(images)  # (B,S,Dv)
        q_feats = self.qformer(visual_seq)          # (B,Q,Dq)
        logits = self.decoder(tgt_ids, q_feats)       # (B,T,V)
        return logits
