import torch
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from transformers import get_scheduler

class Trainer:
    def __init__(self,
                 lr: float,
                 epochs: int,
                 model,
                 gradient_accumulation_steps: int,
                 gpu_id,
                 mixed_precision,
                 scaler,
                 ctx):
        self.lr = lr
        self.epochs = epochs
        self.gpu_id = gpu_id
        self.model = model.to(f"cuda:{self.gpu_id}")
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.scaler = scaler
        self.ctx = ctx
        self.optimizer = None

    def eval_(self, model, dataset):
        model.eval()
        total_loss = 0
        for batch in dataset:
            batch = {k:v.to(self.gpu_id) for k, v in batch.items()}
            with torch.no_grad():
                with self.ctx:
                    outputs = model(**batch)

            loss = outputs.loss
            total_loss += loss.item()
        
        return {"loss": total_loss/len(dataset)}
    
    def train(self,
              train_dataloader,
              valid_dataloader,
              display_steps: int,
              save_steps: int,
              save_name: str = None,
              save_checkpoint: bool = False,
              checkpoint = None):
        
        num_update_steps_per_epoch = len(train_dataloader)

        self.model = DDP(self.model, device_ids = [self.gpu_id], output_device = self.gpu_id)
        self.optimizer = AdamW(self.model.parameters(), lr = self.lr, weight_decay = 0.06)
        
        if checkpoint is not None:
            current_steps = checkpoint["current_steps"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            num_steps = num_update_steps_per_epoch * self.epochs - current_steps
            progress_bar = tqdm(range(num_steps))
            lr_scheduler = get_scheduler("cosine",
                                         optimizer = self.optimizer,
                                         num_warmup_steps = 0,
                                         num_training_steps = num_steps)
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            self.model.load_state_dict(checkpoint["model_state_dict"])
            total_loss = checkpoint["total_loss"]

        else:
            current_steps = 0
            num_steps = num_update_steps_per_epoch * self.epochs
            # progress_bar = tqdm(range(num_steps))
            lr_scheduler = get_scheduler("cosine",
                                         optimizer = self.optimizer,
                                         num_warmup_steps = 100,
                                         num_training_steps = num_steps)
            total_loss = 0

        idx = 0
        for epoch in range(self.epochs):
            
            train_dataloader.sampler.set_epoch(epoch)    
            self.model.train()
            for batch in train_dataloader:
                idx += 1
                if idx > current_steps:
                    batch = {k:v.to(self.gpu_id) for k, v in batch.items()}
                    self.optimizer.zero_grad()
                    with self.ctx:
                        outputs = self.model(**batch)

                    loss = outputs.loss
                    total_loss += loss.item()

                    loss /= self.gradient_accumulation_steps
                    if self.mixed_precision:
                        self.scaler.scale(loss).backward()

                        if idx % self.gradient_accumulation_steps == 0:
                            self.scaler.step(self.optimizer)
                            lr_scheduler.step()
                            self.scaler.update()

                    else:
                        loss.backward()
                        if idx % self.gradient_accumulation_steps == 0:
                            self.optimizer.step()
                            lr_scheduler.step()
                    
                    progress_bar.update(1)
                    current_steps += 1

                    if current_steps % display_steps == 0:
                        print(f'Epoch: {epoch + 1} -- step: {current_steps} -- train_loss: {total_loss/current_steps}')
                            
                    if current_steps % len(train_dataloader) == 0:
                        eval_ = self.eval_(model = self.model, dataset = valid_dataloader)
                        print(f'Epoch: {epoch + 1} -- step: {current_steps} -- train_loss: {total_loss/current_steps} -- val_loss: {eval_["loss"]}')
                        print("----------------------------- End of epoch {} -----------------------------".format(epoch + 1))
                        
                    if save_checkpoint is True:
                        if current_steps % save_steps == 0:
                            print("Saving..........")
                            torch.save({"model_state_dict": self.model.module.state_dict(),
                                        "optimizer_state_dict": self.optimizer.state_dict(),
                                        "scaler_state_dict": self.scaler.state_dict(),
                                        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                                        "current_steps": current_steps,
                                        "total_loss": total_loss},
                                       save_name)
                            print("****** Save successfully ******")
                            
