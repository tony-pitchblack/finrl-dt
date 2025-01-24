# Benchmarks

## SB-3 | PPO | 252 timestamps (daily)

### Config #1 (base)
```
  "n_envs": 1 # number of vectorized envs

  "n_steps": 2048, # rllib 'batch_size'
  "n_epochs":10,
  "batch_size": 128, # rllib 'minibatch_size'
```

| Total timesteps | CPU   | GPU   | Execution Time         | Notes                        | Train Sharpe |
|------------------|-------|-------|------------------------|------------------------------|------------|
|10,240 | 2x    | 1xT4 | 34.5 s ± 0 ns | mean ± std. dev. of 1 run, 1 loop each | 1 |
| 50,000           | 2x    | -     | 3min 10s ± 5.17s       | Mean ± std. dev. of 5 runs, 1 loop each | 2.8 |
| 50,000           | 2x    | 1xT4  | 3min 4s ± 0ns          | Mean ± std. dev. of 1 run, 1 loop each | 2.8 |
|200_000 | 2x    | 1xT4 | 12min 8s ± 0 ns | mean ± std. dev. of 1 run, 1 loop each | 4 |

##  RLLib | PPO | 252 timestamps (daily)

### Config #1 (base)
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

| Total timesteps | CPU   | GPU   | Execution Time         | Notes                        |Train Sharpe |
|------------------|-------|-------|------------------------|------------------------------|------------|
|10,240 | 2x    | 1xT4 | 45.4 s ± 0 ns | mean ± std. dev. of 1 run, 1 loop each | 2 |
| 50,000           | 2x    | -     | 4min 31s ± 5.06s       | Mean ± std. dev. of 5 runs, 1 loop each | 2.8 |
| 50,000           | 2x    | 1xT4  | 3min 52s ± 0ns         | Mean ± std. dev. of 1 run, 1 loop each | 2.8 |
|200,000 | 2x    | 1xT4 | 13min 47s ± 0 ns | mean ± std. dev. of 1 run, 1 loop each | 4 |

### Config #2
```
    .env_runners(
        num_env_runners=0, # no runners, barebones env
        num_envs_per_env_runner=5
    )
    .training(
        train_batch_size=2048,
        num_epochs=10,
        minibatch_size=128,
    )
```

| Total timesteps| Steps per runner | CPU   | GPU   | Execution Time         | Notes                        |Train Sharpe |
|------------------|------------------|-------|-------|------------------------|------------------------------|------------|
|10,240 | 2,048 | 2x    | 1xT4 |6.61 s ± 678 ms | mean ± std. dev. of 5 runs, 1 loop each |1.2 |
|10,240 | 5,120 | 2x    | 1xT4 |19.8 s ± 1.25 | mean ± std. dev. of 5 runs, 1 loop each | 1.5 |

