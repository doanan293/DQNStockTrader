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
"""
def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=10):
    total_profit = 0
    data_length = len(data) - 1

    agent.inventory = []
    avg_loss = []

    state = get_state(data, 0, window_size + 1, agent.inventory)

    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1, agent.inventory)

        # select an action
        action = agent.act(np.asarray(state).reshape(-1, 11, 1))

        # BUY
        if action == 1:
            agent.inventory.append(data[t])

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            num_shares = len(agent.inventory)
            selling_price = data[t]
            bought_prices = sum(agent.inventory)
            reward = num_shares * selling_price - bought_prices
            total_profit += reward
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
"""
def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=10):
    total_profit = 0
    data_length = len(data) - 1
    capital = 100000000
    agent.inventory = []
    avg_loss = []
    num_shares = 0
    state = get_state(data, 0, window_size + 1, num_shares, capital)

    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):
        reward = 0
        num_shares = sum(volume for volume, _ in agent.inventory)
        next_state = get_state(data, t + 1, window_size + 1, num_shares, capital)
        #print(agent.inventory)
        #print(len(agent.inventory))
        # select an action
        action = agent.act(np.asarray(state).reshape(-1, 12, 1))

        #print(num_shares)
        # BUY
        if action >= 1 and action <= 10:
            volume = 100 * action
            if capital >= volume * data[t]:
                agent.inventory.append((volume, data[t]))
                capital -= volume * data[t]

        # SELL
        elif action == 11 and len(agent.inventory) > 0:
            selling_price = data[t]
            bought_prices = sum(volume * price for volume, price in agent.inventory)
            reward = num_shares * selling_price - bought_prices
            total_profit += reward
            capital += num_shares * selling_price
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
    capital = 100000000
    history = []
    agent.inventory = []
    num_shares = 0
    state = get_state(data, 0, window_size + 1, num_shares, capital)

    for t in range(data_length):
        reward = 0
        num_shares = sum(volume for volume, _ in agent.inventory)
        next_state = get_state(data, t + 1, window_size + 1, num_shares, capital)

        # select an action
        action = agent.act(np.asarray(state).reshape(-1 ,12 ,1), is_eval=True)

        # print(num_shares)
        # BUY
        if action >= 1 and action <= 10:
            volume = 100 * action
            if capital >= volume * data[t]:
                agent.inventory.append((volume, data[t]))
                capital -= volume * data[t]
                if debug:
                    logging.debug("Buy {} shares at: {}".format(volume, format_currency(data[t])))
        # SELL
        elif action == 11 and len(agent.inventory) > 0:
            selling_price = data[t]
            bought_prices = sum(volume * price for volume, price in agent.inventory)
            reward = num_shares * selling_price - bought_prices
            total_profit += reward
            capital += num_shares * selling_price
            agent.inventory.clear()  # Clear inventory after selling

            history.append((data[t], "SELL", num_shares))

            if debug:
                logging.debug(
                    "Sell {} shares at: {} | Position: {}".format(num_shares,
                                                                  format_currency(data[t]),
                                                                  format_position(reward)))
        # HOLD
        else:
            history.append((data[t], "HOLD"))

        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            return total_profit, history
