# Benchmarks

## SB-3 | PPO | 252 timestamps (daily)

### Config #1
```
  "n_steps": 2048, # rllib 'batch_size'
  "n_epochs":10,
  "batch_size": 128, # rllib 'minibatch_size'
```

| Total steps |Envs| CPU   | GPU   | Execution Time         | Notes                        | Train Sharpe |Async|
|--------|---|-------|-------|------------------------|---------------------------------|------------|---|
| 50,000 | 1          | 2x    | -     | 3min 10s ± 5.17s       | Mean ± std. dev. of 5 runs, 1 loop each | 2.8 | No |
| 50,000 | 1          | 2x    | 1xT4  | 3min 4s ± 0ns          | Mean ± std. dev. of 1 run, 1 loop each | 2.8 | No |
| 50,000 | 1          | 2x    | - | 39.2 s ± 1.83 s         | Mean ± std. dev. of 2 run, 1 loop each | 1 | No |
| 50,000 | 1          | 2x    | - | ??   | Mean ± std. dev. of 2 run, 1 loop each | 1 | YES |
|200_000 | 1| 2x    | 1xT4 | 12min 8s ± 0 ns | mean ± std. dev. of 1 run, 1 loop each | 4 | No |
|10,240 | 1| 2x    | 1xT4 | 34.5 s ± 0 ns | mean ± std. dev. of 1 run, 1 loop each | 2.4 | No |
|10,240 | 2 | 2x    | 1xT4 |22.7 s ± 1.18 s |mean ± std. dev. of 5 runs, 1 loop each | 1.9 | No |
|10,240 | 5 | 2x    | 1xT4 |??? |??? | ???| No |

##  RLLib | PPO | 252 timestamps (daily)

### Config #1
```
    .training(
        train_batch_size=2048,
        num_epochs=10,
        minibatch_size=128,
    )
```

| Total steps |Envs | CPU   | GPU   | Execution Time         | Notes                        |Train Sharpe |Async|
|---------------|---|-------|-------|------------------------|------------------------------|------------|---|
| 50,000 | 1          | 2x    | -     | 4min 31s ± 5.06s       | Mean ± std. dev. of 5 runs, 1 loop each | 2.8 | No |
| 50,000 | 1           | 2x    | 1xT4  | 3min 52s ± 0ns         | Mean ± std. dev. of 1 run, 1 loop each | 2.8 | No |
|200,000| 1 | 2x    | 1xT4 | 13min 47s ± 0 ns | mean ± std. dev. of 1 run, 1 loop each | 4 | No |
|10,240 | 1 | 2x    | 1xT4 | 45.4 s ± 0 ns | mean ± std. dev. of 1 run, 1 loop each | 2.4 | No |
|10,240 | 2 | 2x    | 1xT4 |19.8 s ± 1.25s | mean ± std. dev. of 5 runs, 1 loop each | 1.9 | No |
|10,240 | 5 | 2x    | 1xT4 |6.61 s ± 678 ms | mean ± std. dev. of 5 runs, 1 loop each |1.2 | No |
|10,240 | 5 | 2x    | - |47.7 s ± 551 ms | mean ± std. dev. of 2 runs, 1 loop each | 0.8 | No |
|10,240 | 5 | 2x    | - |59.5 s ± 1.09 s | mean ± std. dev. of 2 runs, 1 loop each | 0.8 | Yes |
|10,240 | 2 | 2x    | - |53.7 s ± 1.21 s | mean ± std. dev. of 2 runs, 1 loop each | 0.8 | No |
|10,240 | 2 | 2x    | - | 1min ± 1.9 s | mean ± std. dev. of 2 runs, 1 loop each | ??? | Yes |



