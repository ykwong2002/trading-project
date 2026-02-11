import gym
import numpy as np

from gym import spaces

class TradingEnv(gym.env):
    """
    Custom trading environment for EUR/USD. At each step:
    - Observe a window of data (OHLC + indicators)
    - Take an action: choose among NO TRADE or combo of (direction, SL, TP)
    - Computes reward based on PnL from that decision
    """
    def __init__()