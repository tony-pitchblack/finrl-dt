import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers.pytorch_utils import Conv1D

import sys
import os
# use ../../decision_transformer as decision_transformer when run as main
if __name__=="__main__":
    sys.path.insert(0, os.path.abspath('../..'))
    sys.path.insert(0, os.path.abspath('..'))

from decision_transformer.models.model import TrajectoryModel
from transformers.models.gpt2 import GPT2Tokenizer
from decision_transformer.models.trajectory_gpt2 import GPT2Model, GPT2LMHeadModel
from decision_transformer.models.trajectory_gpt2_LoRA import GPT2Model_LoRA, GPT2LMHeadModel_LoRA
from decision_transformer.models.trajectory_gpt2_LoRA import GPT2Config_LoRA

from decision_transformer.models.utils import ResidualBlock, MLPBlock

class DecisionTransformer(TrajectoryModel):
    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    @property
    def transformer(self):
        return self.transformer_model.transformer
      
    def __init__(
        self,
        args,
        state_dim,
        act_dim,
        hidden_size,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        print('args["pretrained_lm"]: ', args["pretrained_lm"])

        self.hidden_size = hidden_size
        
        if args["pretrained_lm"] is not None:
            print("Loading from pretrained "+args["pretrained_lm"]+" model")
            if args['lora']:
                config = GPT2Config_LoRA.from_pretrained(args["pretrained_lm"])
                self.transformer_model = GPT2LMHeadModel_LoRA.from_pretrained(
                    args["pretrained_lm"],
                    config=config
                )
            hidden_size = config.n_embd
            self.hidden_size = config.n_embd

        if args['random_weights_pretrained_lm']:  # Randomly initialize the weights of the pretrained model for ablation study
            print("Randomly initializing the weights of the pretrained model...")
            
            # Check weights before initialization
            before_weights = {}
            for name, param in self.transformer_model.named_parameters():
                if 'weight' in name and 'lora' not in name.lower():
                    before_weights[name] = param.data.clone()
            
            # Directly reinitialize all non-LoRA weights
            for name, param in self.transformer_model.named_parameters():
                if 'lora' not in name.lower():
                    if 'weight' in name:
                        nn.init.normal_(param.data, mean=0.0, std=0.02)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
            
            # Check weights after initialization
            after_weights = {}
            changed_weights = []
            for name, param in self.transformer_model.named_parameters():
                if 'weight' in name and 'lora' not in name.lower():
                    after_weights[name] = param.data.clone()
                    if not torch.allclose(before_weights[name], after_weights[name]):
                        changed_weights.append(name)
            
            print(f"Number of changed weight tensors: {len(changed_weights)}")
            print("Changed weight tensors:")
            for name in changed_weights:
                print(f"  - {name}")

        if max_ep_len > config.n_positions and args["extend_positions"]:
            current_max_pos, embed_size = self.transformer.wpe.weight.shape
            new_encoder_pos_embed = self.transformer.wpe.weight.new_empty(
                max_ep_len, embed_size
            )
            # copy position embeddings over and over to initialize the new position embeddings
            orig_k = 2
            k = orig_k
            step = current_max_pos - k
            new_encoder_pos_embed[:k] = self.transformer.wpe.weight[:k]
            while k < max_ep_len - 1:
                new_encoder_pos_embed[k : (k + step)] = self.transformer.wpe.weight[
                    orig_k : min(max_ep_len - k + orig_k, current_max_pos)
                ]
                k += step
            self.transformer.wpe.weight.data = new_encoder_pos_embed

        if args["extend_positions"]:
            self.embed_timestep = self.transformer.wpe
        else:
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
    
        if args["mlp_embedding"]:
            self.embed_return = ResidualBlock(1, hidden_size)
            self.embed_state = ResidualBlock(self.state_dim, hidden_size)
            self.embed_action = ResidualBlock(self.act_dim, hidden_size)
        else:
            self.embed_return = torch.nn.Linear(1, hidden_size)
            self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
            self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        
        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        if args["mlp_embedding"]: # we run this
          if args["share_input_output_proj"]: raise ValueError("With MLP in embeddings, you cannot share the projections")
          self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
          self.predict_action = MLPBlock(self.hidden_size, self.act_dim, self.hidden_size)
          self.predict_return = torch.nn.Linear(hidden_size, 1)
        else:
          if args["share_input_output_proj"]:
            self.predict_state = lambda x: F.linear(x, self.embed_state.weight.t())
            self.predict_return = lambda x: F.linear(x, self.embed_return.weight.t())
            self.predict_action = lambda x: F.tanh(
                F.linear(x, self.embed_action.weight.t())
            )
          else:
            self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
            self.predict_action = nn.Sequential(
                *(
                    [nn.Linear(hidden_size, self.act_dim)]
                    + ([nn.Tanh()] if action_tanh else [])
                )
            )
            self.predict_return = torch.nn.Linear(hidden_size, 1)
        
        self.past_key_values = None
        print(self)

    def generate_explanation(self, hidden_states, attention_mask):
        # Use the last hidden state to generate an explanation
        last_hidden_state = hidden_states[:, -1, :]
        
        # You might need to add a linear layer to project to the vocab size
        # if not already present in the model
        if not hasattr(self, 'lm_head'):
            self.lm_head = nn.Linear(self.hidden_size, self.transformer_model.config.vocab_size, bias=False)
        
        logits = self.lm_head(last_hidden_state)
        
        # Convert logits to input_ids for generation
        input_ids = logits.argmax(dim=-1).unsqueeze(1)
        
        # Generate text using input_ids instead of inputs_embeds
        generated_ids = self.transformer_model.generate(
            input_ids=input_ids,
            max_length=50,  # Adjust as needed
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.transformer_model.config.pad_token_id,  # Ensure pad_token_id is set
            attention_mask=attention_mask, # Ensure attention_mask is set
        )
        
        # Decode the generated ids to text
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token # Set pad_token to eos_token

        explanation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return explanation

    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        attention_mask=None,
        past_key_values=None,
    ):

        batch_size, seq_length = states.shape[0], states.shape[1]
        
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        # embed each modality with a different head
        
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
       
        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        all_embs = self.embed_ln(stacked_inputs)

        stacked_inputs = all_embs + time_embeddings.repeat_interleave(3, dim=1)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            past_key_values=None,  # self.past_key_values,
            use_cache=True,
        )
        x = transformer_outputs["last_hidden_state"]
        self.past_key_values = transformer_outputs["past_key_values"]

        # explanation = self.generate_explanation(x)
        # print("Explanation: ", explanation)

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3) # I think we need to snap this to generate text to let the model explain why we acted the way he did

        action_preds = self.predict_action(x[:, 1])  # predict next action given state

        return None, action_preds, None, None 

    def get_action(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        past_key_values=None,
        **kwargs
    ):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            returns_to_go = returns_to_go[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]

            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            attention_mask = attention_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (
                            returns_to_go.shape[0],
                            self.max_length - returns_to_go.shape[1],
                            1,
                        ),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds, __ = self.forward(
            states,
            actions,
            None,
            returns_to_go,
            timesteps,
            attention_mask=attention_mask,
            **kwargs,
        )

        return action_preds[0, -1]
