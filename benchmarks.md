# Benchmarks

## SB-3 | PPO | 252 timestamps (daily)

### Config #1
```
  "n_steps": 2048, # rllib 'batch_size'
  "n_epochs":10,
  "batch_size": 128, # rllib 'minibatch_size'
```

| Total steps | Envs | Async | CPU   | GPU   | Execution Time         | Notes                        | Train Sharpe |
|-------------|------|-------|-------|-------|------------------------|-----------------------------|--------------|
| 50,000      | 1    | No    | 2x    | -     | 3min 10s ± 5.17s       | Mean ± std. dev. of 5 runs, 1 loop each | 2.8          |
| 50,000      | 1    | No    | 2x    | 1xT4  | 3min 4s ± 0ns          | Mean ± std. dev. of 1 run, 1 loop each  | 2.8          |
| 50,000      | 1    | No    | 2x    | -     | 39.2 s ± 1.83 s        | Mean ± std. dev. of 2 run, 1 loop each  | 1            |
| 50,000      | 1    | YES   | 2x    | -     | ??                     | Mean ± std. dev. of 2 run, 1 loop each  | 1            |
| 200,000     | 1    | No    | 2x    | 1xT4  | 12min 8s ± 0 ns        | Mean ± std. dev. of 1 run, 1 loop each  | 4            |
| 10,240      | 1    | No    | 2x    | 1xT4  | 34.5 s ± 0 ns          | Mean ± std. dev. of 1 run, 1 loop each  | 2.4          |
| 10,240      | 2    | No    | 2x    | 1xT4  | 22.7 s ± 1.18 s        | Mean ± std. dev. of 5 runs, 1 loop each | 1.9          |
| 10,240      | 5    | No    | 2x    | 1xT4  | ???                    | ???                          | ???          |


##  RLLib | PPO | 252 timestamps (daily)

### Config #1
```
    .training(
        train_batch_size=2048,
        num_epochs=10,
        minibatch_size=128,
    )
```

#### CPU vs GPU
| Total steps | Envs | Async | CPU   | GPU   | Execution Time         | Notes                        | Train Sharpe |
|-------------|------|-------|-------|-------|------------------------|------------------------------|--------------|
| 50,000      | 1    | No    | 2x    | -     | 4min 31s ± 5.06s   | Mean ± std. dev. of 5 runs, 1 loop each | 2.8          |
| 50,000      | 1    | No    | 2x    | 1xT4  | 3min 52s ± 0ns     | Mean ± std. dev. of 1 run, 1 loop each  | 2.8          |
| 200,000     | 1    | No    | 2x    | 1xT4  | 13min 47s ± 0 ns   | Mean ± std. dev. of 1 run, 1 loop each  | 4            |

#### Number of envs
| Total steps | Envs | Async | CPU   | GPU   | Execution Time         | Notes                        | Train Sharpe |
|-------------|------|-------|-------|-------|------------------------|------------------------------|--------------|
| 10,240      | 1    | No    | 2x    | 1xT4  | 1min 13s ± 730 ms   | mean ± std. dev. of 3 runs, 1 loop each  | 1.7        |
| 10,240      | 2    | No    | 2x    | 1xT4  | 59.7 s ± 3.93 s | mean ± std. dev. of 3 runs, 1 loop each  | 1.86        |
| 10,240      | 5    | No    | 2x    | 1xT4  | 51.8 s ± 572 ms   | mean ± std. dev. of 3 runs, 1 loop each  | 1         |

#### Sync vs Async
| Total steps | Envs | Async | CPU   | GPU   | Execution Time         | Notes                        | Train Sharpe |
|-------------|------|-------|-------|-------|------------------------|------------------------------|--------------|
| 10,240      | 2    | No    | 2x    | 1xT4  | 53.8 s ± 0 ns     | Mean ± std. dev. of 1 runs, 1 loop each | 1.6 |         |
| 10,240      | 2    | Yes    | 2x    | 1xT4  | 1min 32s        | Mean ± std. dev. of 1 runs, 1 loop each | 1.5 |
| 10,240    | 5    | No    | 2x    | 1xT4  | 48.5 s ± 0 ns        | Mean ± std. dev. of 1 runs, 1 loop each | 1.6 | 
| 10,240    | 5    |  Yes    | 2x    | 1xT4  | 1min 48s ± 0 ns   | Mean ± std. dev. of 1 runs, 1 loop each | 1.5 |         
| 10,240      | 5    | No    | 2x    | -     | 47.7 s ± 551 ms    | Mean ± std. dev. of 2 runs, 1 loop each | 0.8          |
| 10,240      | 5    | Yes   | 2x    | -     | 59.5 s ± 1.09 s  | Mean ± std. dev. of 2 runs, 1 loop each | 0.8          |
| 10,240      | 2    | No    | 2x    | -     | 53.7 s ± 1.21 s    | Mean ± std. dev. of 2 runs, 1 loop each | 0.8          |
| 10,240      | 2    | Yes   | 2x    | -     | 1min ± 1.9 s     | Mean ± std. dev. of 2 runs, 1 loop each | ???          |




