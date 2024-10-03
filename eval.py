import argparse
import torch
import numpy as np
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import pandas as pd
from finrl.config import INDICATORS
from transformers import GPT2Config

# Load environment data
trade = pd.read_csv('trade_data.csv')
trade = trade.set_index(trade.columns[0])
trade.index.names = ['']

stock_dimension = len(trade.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

def evaluate(variant):
    # Set up environment
    env = StockTradingEnv(df=trade, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_ep_len = 755
    scale = 1e6

    # Load pretrained model config
    config = GPT2Config.from_pretrained(variant["pretrained_lm"])
    config.resid_pdrop = variant["dropout"]

    # Initialize model with pretrained weights
    model = DecisionTransformer(
        args=variant,
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=variant["K"],
        max_ep_len=max_ep_len,
        hidden_size=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_inner=4 * config.n_embd,
        activation_function=config.activation_function,
        n_positions=config.n_positions,
        resid_pdrop=config.resid_pdrop,
        attn_pdrop=config.attn_pdrop,
        mlp_embedding=variant["mlp_embedding"]
    )
    
    # Load pretrained weights
    model.transformer_model.load_state_dict(torch.load(variant["pretrained_lm"]))

    # Load LoRA parameters
    lora_state_dict = torch.load(variant["lora_path"])
    model.load_state_dict(lora_state_dict, strict=False)

    model.to(variant["device"])
    model.eval()

    # Evaluate for different target rewards
    target_rewards = [1_500_000]
    for target_rew in target_rewards:
        returns = []
        for _ in range(variant["num_eval_episodes"]):
            with torch.no_grad():
                ret, _ = evaluate_episode_rtg(
                    env,
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=max_ep_len,
                    scale=scale,
                    target_return=target_rew / scale,
                    mode="normal",
                    state_mean=None,
                    state_std=None,
                    device=variant["device"],
                )
            returns.append(ret)
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        print(f"Target reward: {target_rew}")
        print(f"Mean return: {mean_return:.2f}")
        print(f"Std return: {std_return:.2f}")
        print("-------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapt_mode", action="store_true", default=False)
    parser.add_argument("--adapt_embed", action="store_true", default=False)
    parser.add_argument("--lora", action="store_true", default=False) 
    parser.add_argument("--lora_path", type=str, required=True, help="Path to the saved LoRA parameters")
    parser.add_argument("--pretrained_lm", type=str, required=True, help="Name or path of the pretrained language model")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--env", type=str, default="stock_trading")
    parser.add_argument("--dataset", type=str, default="your_dataset_name")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mlp_embedding", action="store_true")
    parser.add_argument("--num_eval_episodes", type=int, default=10)
    parser.add_argument("--model_type", type=str, default="dt")  # dt for decision transformer, bc for behavior cloning
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--description", type=str, default="")

    parser.add_argument("--pct_traj", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_type", type=str, default="dt")  # dt for decision transformer, bc for behavior cloning

    # architecture, don't need to care about in our method
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--extend_positions", action="store_true", default=False)
    parser.add_argument("--share_input_output_proj", action="store_true", default=False)

    # adaptations
    parser.add_argument("--adapt_mode", action="store_true", default=False)
    parser.add_argument("--lora", action="store_true", default=False)
    parser.add_argument("--only_adapt_last_two_blocks", action="store_true", default=False)
    parser.add_argument("--adapt_last_two_blocks", action="store_true", default=False)
    parser.add_argument("--adapt_ln", action="store_true", default=False)
    parser.add_argument("--adapt_attn", action="store_true", default=False)
    parser.add_argument("--adapt_ff", action="store_true", default=False)
    parser.add_argument("--adapt_embed", action="store_true", default=False)
    parser.add_argument("--adapt_wte", action="store_true", default=False)
    parser.add_argument("--adapt_wpe", action="store_true", default=False)    

    args = parser.parse_args()
    evaluate(variant=vars(args))