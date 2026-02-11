import gym
import numpy as np

from gym import spaces

class TradingEnv(gym.Env):
    """
    Custom trading environment for EUR/USD. At each step:
    - Observe a window of data (OHLC + indicators)
    - Take an action: choose among NO TRADE or combo of (direction, stop loss SL, take profit TP)
    - Computes reward based on PnL from that decision
    """
    
    # Agent will read last 30 candles for decision making
    def __init__(self, df, window_size=30, sl_options=None, tp_options=None):
        super(TradingEnv, self).__init__()

        # Store dataframe containing prices and indicators
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)

        # Observation parameters
        self.window_size = window_size

        # Discretise SL and TP distances in pips or price terms
        self.sl_options = sl_options if sl_options else [60, 90, 120]
        self.tp_options = tp_options if tp_options else [60, 90, 120]

        # Create action space with NO TRADE option (potentially can create a neural network to learn action space instead of discrete actions)
        self.action_map = [(None, None, None)]
        for direction in [0, 1]:
            for sl in self.sl_options:
                for tp in self.tp_options:
                    self.action_map.append((direction, sl, tp))
        
        # Total number of discrete actions = 1 + (2 * len(sl_options) * len(tp_options))
        self.action_space = spaces.Discrete(len(self.action_map))

        # Number of features in the observation
        self.num_features = self.df.shape[1]
        # return window of these features as 2D array
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size, self.num_features), dtype=np.float32
        )

        # Internal state to track portfolio value, position, and PnL
        self.current_step = 0
        self.done = False
        self.equity = 10000.0 # Starting equity
        self.max_slippage = 0.000
        self.positions = []

        # For logging
        self.equity_curve = []
        self.last_trade_info = None # Track most recent trade details

        def _get_observation(self):
            """
            Return the last 'window_size' of observations as a 2D numpy array of shape (window_size, num_features)
            If at the start not enouhg data, pad with 0
            """
            start = max(self.current_step - self.window_size, 0)
            obs_df = self.df.iloc[start:self.current_step]

            # If not enough data, pad with 0
            if len(obs_df) < self.window_size:
                padding_rows = self.window_size - len(obs_df)
                first_part = np.tile(obs_df.iloc[0].values, (padding_rows, 1))
                obs_array = np.concatenate([first_part, obs_df.values])
            else:
                obs_array = obs_df.values
            
            return obs_array.astype(np.float32)
