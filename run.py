from config import Config
from prompt import Prompter
from process_analysis import DataProcess
from model_inputs import MODEL_INPUTS
from train import Trainer

import os 

import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.cuda.amp import GradScaler, autocast

if __name__ == "__main__":

    # Mixed precision
    mixed_precision = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    ctx = autocast(dtype = mixed_precision)
    scaler = GradScaler()

    # ddp config
    backend = "nccl"
    init_process_group(backend = backend)
    local_rank = int(os.environ["LOCAL_RANK"])

    # Tokenizer and Model
    config = Config()
    tokenizer = config.tokenizer(model_checkpoint = "bigscience/bloomz")
    model = config.load_pretrained_model(model_checkpoint = "bigscience/bloomz-1b7", device_map = {"": torch.device(f"cuda:{local_rank}")})
    lora_model = config.add_lora(model = model, r = 8, lora_alpha = 16, lora_dropout = 0.05)

    # Dataset
    data_prcess = DataProcess(data_path = "MBZUAI/Bactrian-X", tokenizer = tokenizer)
    dataset = data_prcess.load_data()
    prompter = Prompter()

    splited_dataset = dataset.train_test_split(test_size = 0.1, seed = 42)

    # Model inputs
    model_inputs = MODEL_INPUTS(prompter =  prompter,
                                tokenizer = tokenizer,
                                max_length = 512)
    
    train_data = splited_dataset["train"].shuffle().map(model_inputs.generate_and_tokenize_prompt)
    valid_data = splited_dataset["test"].map(model_inputs.generate_and_tokenize_prompt)

    train_data = train_data.remove_columns(["instruction", "input", "id", "output"])
    valid_data = valid_data.remove_columns(["instruction", "input", "id", "output"])

    train_data.set_format("torch")
    valid_data.set_format("torch")

    train_dataloader, valid_dataloader = model_inputs.prepare_dataloader(train_data,
                                                                         valid_data,
                                                                         batch_size = 2)
    
    # Train
    trainer = Trainer(lr = 3e-4,
                      epochs = 3,
                      model = lora_model,
                      gradient_accumulation_steps = 4,
                      gpu_id = local_rank,
                      mixed_precision = mixed_precision,
                      scaler = scaler,
                      ctx = ctx)
    checkpoint = None
    trainer.train(train_dataloader = train_dataloader,
                  valid_dataloader = valid_dataloader,
                  display_steps = 1,
                  save_steps = 1,
                  save_name = "bloomz-1b7-checkpoint.pt",
                  save_checkpoint = True,
                  checkpoint = checkpoint)
    
    destroy_process_group()
