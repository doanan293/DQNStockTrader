import os
import logging

import numpy as np

from tqdm import tqdm

from .utils import (
    format_currency,
    format_position
)
from .ops import (
    get_state
)

def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=10):
    total_profit = 0
    data_length = len(data) - 1
    capital = 100000000
    agent.inventory = []
    avg_loss = []

    state = get_state(data, 0, window_size + 1, agent.inventory)

    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1, agent.inventory)

        # select an action
        action = agent.act(np.asarray(state).reshape(-1, 11, 1))

        # BUY
        if action >= 1 and action <= 5:
            volume = 100 * action
            if capital >= volume * data[t]:
                agent.inventory.append((volume, data[t]))
                capital -= volume * data[t]

        # SELL
        elif action == 6 and len(agent.inventory) > 0:
            num_shares = sum(volume for volume, _ in agent.inventory)
            selling_price = data[t]
            bought_prices = sum(volume * price for volume, price in agent.inventory)
            reward = num_shares * selling_price - bought_prices
            total_profit += reward
            capital += bought_prices
            agent.inventory.clear()  # Clear inventory after selling

        # HOLD
        else:
            pass

        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    if episode % 10 == 0:
        agent.save(episode)

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, window_size, debug):
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.inventory = []

    state = get_state(data, 0, window_size + 1, agent.inventory)

    for t in range(data_length):
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1, agent.inventory)

        # select an action
        action = agent.act(np.asarray(state).reshape(-1 ,11 ,1), is_eval=True)

        # BUY
        if action == 1:
            agent.inventory.append(data[t])

            history.append((data[t], "BUY"))
            if debug:
                logging.debug("Buy at: {}".format(format_currency(data[t])))

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            num_shares = len(agent.inventory)
            selling_price = data[t]
            bought_prices = sum(agent.inventory)
            reward = num_shares * selling_price - bought_prices
            total_profit += reward
            agent.inventory.clear()  # Clear inventory after selling

            history.append((data[t], "SELL"))
            if debug:
                logging.debug("Sell at: {} | Position: {}".format(format_currency(data[t]),
                                                                  format_position(reward)))
        # HOLD
        else:
            history.append((data[t], "HOLD"))

        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            return total_profit, history
