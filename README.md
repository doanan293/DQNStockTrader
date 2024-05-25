<<<<<<< HEAD
# NCKH
Tradding Bot
=======
# Introduction

This project implements a Stock Trading Bot, trained using Deep Reinforcement Learning, specifically Deep Q-learning in Vietnamese Stock Market

## Approach

DQN algorithms implemented in this project:

- [x] Vanilla DQN
- [x] DQN with fixed target distribution
- [x] Double DQN


## Results

Trained on 30 stocks in VN30 index 1/1/2014 - 31/12/2022 stock data, tested on 1/1/2023 - 4/8/2023 with a profit in Result.csv

## Some Caveats

- At any given state, the agent can only decide to buy 300/600/900 shares or sell all or hold
- The n-day window feature representation is a vector of subsequent differences in Opening Price we're trading followed by a sigmoid operation, done in order to normalize the values to the range [0, 1].

## Getting Started

In order to use this project, you'll need to install the required python packages:

```bash
pip install -r requirements.txt
```

Now you can open up a terminal and start training the agent:

train.py <train-stock> <val-stock> [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug]
```bash
python train.py data/TCB.csv data/TCB20232024 --strategy dqn
```

Once you're done training, run the evaluation script and let the agent make trading decisions:

eval.py <eval-stock> [--window-size=<window-size>] [--model-name=<model-name>] [--debug]
```bash
python eval.py data/TCB.csv data/TCB20232024.csv --model-name TCB_dqn_40
```

>>>>>>> 18e7021 (finsh-project)
