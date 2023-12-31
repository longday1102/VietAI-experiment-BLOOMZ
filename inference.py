from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel, get_peft_model
import torch
from prompt import Prompter

prompter = Prompter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Inference:
    def __init__(self,
                 model_for_tokenizer: str,
                 model_weight_path: str,
                 device = device,
                 prompt_process = prompter):
        
        self.prompt_process = prompt_process
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_for_tokenizer)
        config = PeftConfig.from_pretrained(model_weight_path)
        self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                                          device_map = "auto",
                                                          torch_dtype = torch.float16)
        self.model = PeftModel.from_pretrained(self.model, model_weight_path)

    def __call__(self,
                 instruction: str,
                 input: str = None,
                 target: str = None):
        prompt = self.prompt_process.generate_prompt(instruction = instruction, input = input)
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
            return {"label": target,
                  "response": response}
        else:
            return {"response": response}
        
