from typing import List, Dict
from dataclasses import dataclass
import numpy as np
from collections import deque


@dataclass
class Dataset:
    """(state, action, reward) dataset
    """
    observes:deque
    actions:deque
    rewards:deque

    @classmethod
    def create_empty(cls):
        return Dataset(
            observes=deque(),
            actions=deque(),
            rewards=deque(),
        )
    def append(self, observe, action, reward):
        self.observes.append(observe)
        self.actions.append(action)
        self.rewards.append(reward)

    def extend(self, other):
        self.observes.extend(other.observes)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)

    def overwrite_reward(self, reward):
        self.rewards = deque([reward] * len(self.rewards))

    def __len__(self):
        return self.observes.__len__()

@dataclass
class Datasets:
    """ model train dataset
    for rainforced learning, use only dahai dataset.
    """
    dahai_dataset: Dataset
    reach_dataset: Dataset
    chi_dataset: Dataset
    pon_dataset: Dataset
    kan_dataset: Dataset

    def extend(self, other):
        self.dahai_dataset.extend(other.dahai_dataset)
        self.reach_dataset.extend(other.reach_dataset)
        self.chi_dataset.extend(other.chi_dataset)
        self.pon_dataset.extend(other.pon_dataset)
        self.kan_dataset.extend(other.kan_dataset)

    @classmethod
    def create_empty(cls):
        return Datasets(
            dahai_dataset=Dataset.create_empty(),
            reach_dataset=Dataset.create_empty(),
            chi_dataset=Dataset.create_empty(),
            pon_dataset=Dataset.create_empty(),
            kan_dataset=Dataset.create_empty(),
        )
    
    def overwrite_reward(self, reward):
        self.dahai_dataset.overwrite_reward(reward)
        self.reach_dataset.overwrite_reward(reward)
        self.chi_dataset.overwrite_reward(reward)
        self.pon_dataset.overwrite_reward(reward)
        self.kan_dataset.overwrite_reward(reward)

