# Benchmarks

## SB-3 | PPO | 252 timestamps (daily)

### Config #1
```
  "n_envs": 1 # number of vectorized envs

  "n_steps": 2048, # rllib 'batch_size'
  "n_epochs":10,
  "batch_size": 128, # rllib 'minibatch_size'
```

| Total timesteps | CPU   | GPU   | Execution Time         | Notes                        |
|------------------|-------|-------|------------------------|------------------------------|
| 50,000           | 2x    | -     | 3min 10s ± 5.17s       | Mean ± std. dev. of 5 runs, 1 loop each |
| 50,000           | 2x    | 1xT4  | 3min 4s ± 0ns          | Mean ± std. dev. of 1 run, 1 loop each |
|200_000 | 2x    | 1xT4 | 12min 8s ± 0 ns | mean ± std. dev. of 1 run, 1 loop each |


##  RLLib | PPO | 252 timestamps (daily)

### Config #1
```
    .env_runners(
        num_env_runners=0, # no runners, barebones env
        num_envs_per_env_runner=1
    )
    .training(
        train_batch_size=2048,
        num_epochs=10,
        minibatch_size=128,
    )
```

| Total timesteps | CPU   | GPU   | Execution Time         | Notes                        |
|------------------|-------|-------|------------------------|------------------------------|
| 50,000           | 2x    | -     | 4min 31s ± 5.06s       | Mean ± std. dev. of 5 runs, 1 loop each |
| 50,000           | 2x    | 1xT4  | 3min 52s ± 0ns         | Mean ± std. dev. of 1 run, 1 loop each |

