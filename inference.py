from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, LoraConfig, PeftModel, get_peft_model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Inference:
    def __init__(self,
                 prompt_process,
                 model_checkpoint: str,
                 model_for_tokenizer: str,
                 model_from_hub: str = None,
                 model_from_checkpoint = None,
                 from_hub: bool = False,
                 from_checkpoint: bool = False,
                 device = device):
        
        self.prompt_process = prompt_process
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_for_tokenizer)
        if from_checkpoint is True and from_hub is False:
            self.model = AutoModelForCausalLM.from_pretrained(model_checkpoint, load_in_8bit = True, torch_dtype = torch.float16, device_map = "auto")
            lora_config = LoraConfig(r = 8,
                                    lora_alpha = 16,
                                    lora_dropout = 0.05,
                                    bias = "none",
                                    task_type = "CAUSAL_LM")
            self.model = get_peft_model(self.model, lora_config)
            self.model.load_state_dict(model_from_checkpoint["model_state_dict"])
        elif from_checkpoint is False and from_hub is True:
            config = PeftConfig.from_pretrained(model_from_hub)
            self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, load_in_8bit = True, torch_dtype = torch.float16, device_map = "auto")
            self.model = PeftModel.from_pretrained(self.model, model_from_hub)
        self.model.to(device)

    def __call__(self,
                 instruction: str,
                 input: str = None,
                 target: str = None):
        prompt = self.prompt_process.generate_prompt(instruction = instruction, input = input)
        prompt += " " + self.tokenizer.eos_token
        inputs = self.tokenizer(prompt, return_tensors = "pt")
        inputs = {k:v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(**inputs,
                                      max_new_tokens = 512,
                                      no_repeat_ngram_size = 3,
                                      num_beams = 3,
                                      top_k = 40,
                                      top_p = 128,
                                      bos_token_id = self.tokenizer.bos_token_id,
                                      eos_token_id = self.tokenizer.eos_token_id,
                                      pad_token_id = self.tokenizer.pad_token_id,
                                      early_stopping = True)
        text = self.tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
        response = self.prompt_process.get_response(text)
        if target is not None:
            return {"target": target,
                  "response": response}
        else:
            return {"response": response}
        
