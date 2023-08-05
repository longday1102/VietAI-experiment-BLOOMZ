import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, AutoPeftModelForCausalLM, get_peft_model
import torch

class Config:
    
    def tokenizer(self, model_checkpoint):
        tok = AutoTokenizer.from_pretrained(model_checkpoint)
        return tok
    
    def load_pretrained_model(self, model_checkpoint, device_map):
        model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map = device_map, torch_dtype = torch.float16)
        return model
    
    def add_lora(self, model, r: int, lora_alpha: int, lora_dropout: float):
        lora_config = LoraConfig(r = r,
                                 lora_alpha = lora_alpha,
                                 lora_dropout = lora_dropout,
                                 bias = "none",
                                 task_type = "CAUSAL_LM")
        lora_model = get_peft_model(model, lora_config)
        return lora_model

    def reload_pretrained_model(self, model_weight_path, device_map = None):
        lora_model = AutoPeftModelForCausalLM.from_pretrained(model_weight_path, device_map = device_map)
        return lora_model
