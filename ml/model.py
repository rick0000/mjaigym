import os
import gc
from abc import ABCMeta, abstractmethod
from typing import List
import random
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import mjaigym.loggers as lgs
from ml.net import Head34Net, Head2Net
from mjaigym.board import BoardState

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


EPS = 10**-9

class Model(metaclass=ABCMeta):
    def __init__(self, in_channels:int, mid_channels:int, blocks_num:int, learning_rate:float, batch_size:int):
        
        self.in_channels = in_channels
        model = self.build_model(in_channels, mid_channels, blocks_num)
        self.model = model.to(DEVICE)
        self.loss = []
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = self.get_criterion()
        self.softmax = nn.Softmax(dim=1)
        self.batch_size = batch_size

    def load(self, path):
        state = torch.load(path)
        self.set_state(state)

    def save(self, path):
        state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)

    def get_state(self):
        return {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}

    def set_state(self, state):
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])

    @abstractmethod
    def build_model(self, in_channels:int, mid_channels:int, blocks_num:int):
        raise NotImplementedError()
    
    @abstractmethod
    def get_criterion(self):
        raise NotImplementedError()
    
    @abstractmethod    
    def evaluate(self, states, actions):
        raise NotImplementedError()
    
    @abstractmethod
    def update(self, states, actions):
        raise NotImplementedError()


class Head34Model(Model):
    """打牌用モデル
    """
    def __init__(self, in_channels:int, mid_channels:int, blocks_num:int, learning_rate:float, batch_size:int):
        super().__init__(in_channels, mid_channels, blocks_num, learning_rate, batch_size)
        
    def build_model(self, in_channels:int, mid_channels:int, blocks_num:int):
        return Head34Net(in_channels, mid_channels, blocks_num)
    
    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def policy(self, states):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.Tensor(states).float().to(DEVICE)
            policy = self.model(inputs)
            prob = self.softmax(policy)
        return prob.cpu().detach().numpy()
    
    def estimate(self, states):
        raise NotImplementedError()

    def evaluate(self, experiences):
        batch_num = len(experiences) // self.batch_size
        
        total_loss = 0.0
        correct = 0
        total = 0
        for i in range(batch_num):
            target_experiences = experiences[i*self.batch_size:(i+1)*self.batch_size]
            states = [e[0] for e in target_experiences]
            actions = [e[1] for e in target_experiences]

            self.model.eval()
            with torch.no_grad():
                inputs = torch.Tensor(states).float().to(DEVICE)
                targets = torch.Tensor(actions).long().to(DEVICE)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets.data).cpu().sum().detach()
                total += len(states)
                total_loss += loss.cpu().detach()

            del states, actions, target_experiences, inputs, targets
        gc.collect()    
        acc = 100.0 * correct / (total + EPS)
        return float(total_loss / batch_num), float(acc)

    def update(self, experiences):

        batch_num = len(experiences) // self.batch_size
        
        total_loss = 0.0
        correct = 0
        total = 0
        for i in range(batch_num):
            target_experiences = experiences[i*self.batch_size:(i+1)*self.batch_size]
            states = [e[0] for e in target_experiences]
            actions = [e[1] for e in target_experiences]

            self.model.train()
            inputs = torch.Tensor(states).float().to(DEVICE)
            targets = torch.Tensor(actions).long().to(DEVICE)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(targets.data).cpu().sum().detach()
            total += len(states)
            total_loss += loss.cpu().detach()
            
            del states, actions, target_experiences, inputs, targets
        gc.collect()

        acc = 100.0 * correct / (total + EPS)
        return float(total_loss / batch_num), float(acc)


class Head2Model(Model):
    """立直、チー、ポン、カン用モデル
    """
    def __init__(self, in_channels:int, mid_channels:int, blocks_num:int, learning_rate:float, batch_size:int):
        super().__init__(in_channels, mid_channels, blocks_num, learning_rate, batch_size)
        
    def build_model(self, in_channels:int, mid_channels:int, blocks_num:int):
        return Head2Net(in_channels, mid_channels, blocks_num)

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def policy(self, states):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.Tensor(states).float().to(DEVICE)
            policy = self.model(inputs)
            prob = self.softmax(policy)
        return prob.cpu().detach().numpy()
    
    def estimate(self, states):
        raise NotImplementedError()

    def evaluate(self, experiences):
        states = [e[0] for e in experiences]
        actions = [e[1] for e in experiences]

        self.model.eval()
        with torch.no_grad():
            inputs = torch.Tensor(states).float().to(DEVICE)
            targets = torch.Tensor(actions).long().to(DEVICE)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            correct = predicted.eq(targets.data).cpu().sum().detach()
            total = len(states)
            acc = 100.0 * correct / (total + EPS)
        
        return loss, acc

    def update(self, experiences):
        raise NotImplementedError()


class CriticModel(Model):
    """状態価値予測用モデル
    """
    def __init__(self, in_channels:int, mid_channels:int, blocks_num:int, learning_rate:float, batch_size:int):
        super().__init__(in_channels, mid_channels, blocks_num, learning_rate, batch_size)

    @abstractmethod
    def build_model(self, in_channels:int, mid_channels:int, blocks_num:int):
        return Head2Net(in_channels, mid_channels, blocks_num)

    @abstractmethod
    def get_criterion(self):
        raise NotImplementedError()
    
    @abstractmethod
    def evaluate(self, states, actions):
        raise NotImplementedError()
    
    @abstractmethod
    def update(self, states, actions):
        raise NotImplementedError()