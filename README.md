# Deep Q-learning Stock Trading Bot (VN30)

A stock trading bot for the Vietnamese stock market, trained with deep reinforcement learning (deep Q-learning).

## Approach

DQN variants implemented in this project:

- Vanilla DQN
- DQN with a fixed target network
- Double DQN

## Results

Trained on daily data for the 30 stocks in the VN30 index from 1/1/2014 to 31/12/2022, and tested from 1/1/2023 to 4/8/2023. The test-period profit is in `Result.csv`.

## Caveats

- At each state the agent can buy 300, 600 or 900 shares, sell everything, or hold.
- The n-day window feature is the vector of consecutive differences in opening price, passed through a sigmoid to normalize values to [0, 1].

## Getting started

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train the agent:

```text
train.py <train-stock> <val-stock> [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug]
```

```bash
python train.py data/TCB.csv data/TCB20232024 --strategy dqn
```

Evaluate the trained agent and let it make trading decisions:

```text
eval.py <eval-stock> [--window-size=<window-size>] [--model-name=<model-name>] [--debug]
```

```bash
python eval.py data/TCB.csv data/TCB20232024.csv --model-name TCB_dqn_40
```
