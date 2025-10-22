from typing import Optional, List
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training



class LLMDecoder(nn.Module):
    def __init__(self,
        qformer_dim: int,
        latex_vocab_size: int,
        llm_name: str = "gpt2",
        num_prompt_tokens: int = 16,
        use_8bit: bool = False,
    ):
        super().__init__()

        dtype = torch.float32
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=use_8bit,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=dtype
        )

        for n, m in self.llm.named_modules():
            if isinstance(m, torch.nn.LayerNorm):
                m.to(dtype=dtype)


        hidden_size = getattr(self.llm.config, "hidden_size", None) or getattr(self.llm.config, "n_embd")
        self.hidden_size = hidden_size
        self.num_prompt_tokens = num_prompt_tokens

        self.latex_embed = nn.Embedding(latex_vocab_size, hidden_size)
        self.q_to_hidden = nn.Linear(qformer_dim, hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.out_head = nn.Linear(hidden_size, latex_vocab_size).to(dtype=dtype)

        for p in self.llm.parameters():
            p.requires_grad = False


    def forward(self, tgt_ids: torch.Tensor, qformer_feats: torch.Tensor) -> torch.Tensor:
        B, T = tgt_ids.shape
        Q = qformer_feats.size(1)
        device = tgt_ids.device
        llm_dtype = next(self.llm.parameters()).dtype
        scale = torch.tensor(self.hidden_size ** -0.5, device=device, dtype=llm_dtype)

        self.layer_norm = self.layer_norm.to(dtype=llm_dtype)
        prompt = self.q_to_hidden(qformer_feats).to(dtype=llm_dtype) # (B, Q, H)
        
        tok = self.latex_embed(tgt_ids).to(dtype=llm_dtype) # (B, T, H)
        x = torch.cat([prompt, tok], dim=1) # (B, Q+T, H)
        x = self.layer_norm(x) * scale
        attn_mask = torch.ones((B, Q + T), dtype=torch.float, device=device)

        outputs = self.llm(
            inputs_embeds=x,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )

        hidden = outputs.hidden_states[-1] # (B, Q+T, H)
        hidden_tgt = hidden[:, Q:, :]
        logits = self.out_head(hidden_tgt) # (B, T, V)
        return logits
    

class LLMDecoderLoRA(nn.Module):
    def __init__(self,
        qformer_dim: int,
        latex_vocab_size: int,
        llm_name: str = "gpt2",
        num_prompt_tokens: int = 16,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        use_8bit: bool = False,
    ):
        super().__init__()

        dtype = torch.float32
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=use_8bit,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            quantization_config=bnb_config,
            device_map="auto",
            use_cache=False,
            dtype=dtype
        )
        
        for n, m in self.llm.named_modules():
            if isinstance(m, torch.nn.LayerNorm):
                m.to(dtype=dtype)

                
        if use_8bit and prepare_model_for_kbit_training is not None:
            self.llm = prepare_model_for_kbit_training(self.llm)

        hidden_size = getattr(self.llm.config, 'hidden_size', None) or getattr(self.llm.config, 'n_embd')
        self.hidden_size = hidden_size
        self.num_prompt_tokens = num_prompt_tokens

        self.latex_embed = nn.Embedding(latex_vocab_size, hidden_size)
        self.q_to_hidden = nn.Linear(qformer_dim, hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.out_head = nn.Linear(hidden_size, latex_vocab_size)
        self.out_head = self.out_head.to(dtype=self.llm.dtype)
        
        if lora_target_modules is None:
            lora_target_modules = ["c_attn", "c_proj"]
        if LoraConfig is None:
            raise ImportError("peft not available; install it to use LoRA.")
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules,
        )
        self.llm = get_peft_model(self.llm, lora_cfg)

    def forward(self, tgt_ids: torch.Tensor, qformer_feats: torch.Tensor) -> torch.Tensor:
        B, T = tgt_ids.shape
        Q = qformer_feats.size(1)
        device = tgt_ids.device
        llm_dtype = next(self.llm.parameters()).dtype
        scale = torch.tensor(self.hidden_size ** -0.5, device=device, dtype=llm_dtype)
        self.layer_norm = self.layer_norm.to(dtype=llm_dtype)
        prompt = self.q_to_hidden(qformer_feats).to(dtype=llm_dtype) # (B, Q, H)
        
        tok = self.latex_embed(tgt_ids).to(dtype=llm_dtype) # (B, T, H)
        x = torch.cat([prompt, tok], dim=1) # (B, Q+T, H)
        x = self.layer_norm(x) * scale
        attn_mask = torch.ones((B, Q + T), dtype=torch.float, device=device)

        outputs = self.llm(
            inputs_embeds=x,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )

        hidden = outputs.hidden_states[-1] # (B, Q+T, H)
        hidden_tgt = hidden[:, Q:, :]
        logits = self.out_head(hidden_tgt) # (B, T, V)
        return logits


