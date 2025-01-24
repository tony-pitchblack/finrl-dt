# Benchmarks

## SB-3 | PPO | 252 timestamps (daily)

```
  "n_envs": 1 # number of vectorized envs

  "n_steps": 2048, # rllib 'batch_size'
  "n_epochs":10,
  "batch_size": 128, # rllib 'minibatch_size'
```

| Configuration         | Execution Time         | Notes                        |
|-----------------------|------------------------|------------------------------|
| **[2xCPU]**           | 3min 10s ± 5.17s      | Mean ± std. dev. of 5 runs, 1 loop each |
| **[2xCPU, 1xGPU T4]** | 3min 4s ± 0ns         | Mean ± std. dev. of 1 run, 1 loop each |


##  RLLib | PPO | 252 timestamps (daily)
```
    .environment(
        num_envs=1
    )
    .training(
        train_batch_size=2048,
        num_epochs=10,
        minibatch_size=128,
    )
```

| Configuration         | Execution Time         | Notes                        |
|-----------------------|------------------------|------------------------------|
| **[2xCPU]**           | 4min 31s ± 5.06s      | Mean ± std. dev. of 5 runs, 1 loop each |
| **[2xCPU, 1xGPU T4]** | 3min 52s ± 0ns         | Mean ± std. dev. of 1 run, 1 loop each |
