import numpy as np
import torch
import torch.nn.functional as F

from decision_transformer.training.trainer import Trainer
from transformers import GPT2Tokenizer

class SequenceTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SequenceTrainer, self).__init__(*args, **kwargs)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Adjust the model name if needed
    
    def train_step(self):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            attention_mask,
        ) = self.get_batch(self.batch_size)
        print("train_step:timesteps", timesteps)
        
        action_target = torch.clone(actions)
        
        _, action_preds, _, _ = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )
        

        self.step += 1
        print("train_step:self.step", self.step)
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0
        ]
        print("train_step:action_preds", action_preds)
        print("train_step:action_target", action_target)
        print("train_step:action_preds.shape", action_preds.shape)
        print("train_step:action_target.shape", action_target.shape)
       
        loss = self.loss_fn(
            None,
            action_preds,
            None,
            None,
            action_target,
            None,
        )
        print("loss:", loss)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics["training/action_error"] = (
                torch.mean((action_preds - action_target) ** 2).detach().cpu().item()
            )

        return loss.detach().cpu().item()#, lm_loss.detach().cpu().item()