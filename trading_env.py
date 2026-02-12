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

        def _calculate_reward(self, direction, sl, tp):
            """
            - Immediately calculate PnL based on the next bar's movement until SL/TP is hit
            """
            entry_price = self.df.loc[self.current_step, "Close"]

            # If last step, no movement
            if self.current_step >= self.n_steps - 1:
                return 0.0
            
            next_high = self.df.loc[self.current_step + 1, "High"]
            next_low = self.df.loc[self.current_step + 1, "Low"]

            # Convert pips into price distance
            pip_value = 0.0001
            sl_price_distance = sl * pip_value
            tp_price_distance = tp * pip_value

            # direction == 1 (long)
            if direction == 1:
                stop_loss = entry_price - sl_price_distance
                take_profit = entry_price + tp_price_distance

                # Check if next_low < stop_loss --> SL triggered
                # Check if next_high > take_profit --> TP triggered
                if next_low <= stop_loss and next_high >= take_profit:
                    # Both conditions met --> both SL triggered to play safe
                    pnl = -sl_price_distance
                elif next_low <= stop_loss:
                    # SL triggered, exit long position
                    pnl = -sl_price_distance
                elif next_high >= take_profit:
                    # TP triggered, exit long position
                    pnl = tp_price_distance
                else:
                    # No SL/TP triggered, continue holding and use Close for partial reward
                    next_close = self.df.loc[self.current_step + 1, "Close"]
                    pnl = next_close - entry_price
            else:
                # direction == 0 (short)
                stop_loss = entry_price + sl_price_distance
                take_profit = entry_price - tp_price_distance

                # Check if next_high > stop_loss --> SL triggered
                # Check if next_low < take_profit --> TP triggered
                if next_high >= stop_loss and next_low <= take_profit:
                    if (stop_loss - entry_price) < (entry_price - take_profit):
                        pnl = -sl_price_distance
                    else:
                        pnl = tp_price_distance
                elif next_high >= stop_loss:
                    # SL triggered, exit short position
                    pnl = -sl_price_distance
                elif next_low <= take_profit:
                    # TP triggered, exit short position
                    pnl = tp_price_distance
                else:
                    # No SL/TP triggered, continue holding and use Close for partial reward
                    next_close = self.df.loc[self.current_step + 1, "Close"]
                    pnl = entry_price - next_close
            
            reward = pnl * 10000 # Multiply by 10000 as reward in pips
            return reward
        
        def step(self, action):
            """
            Integer in [0, ... (1 + 2*len(sl_options)*len(tp_options))]
            - 0: NO TRADE
            - Anything else: TRADE (direction, sl, tp)
            """
            # Decode the action
            direction, sl, tp = self.action_map[action]

            if direction is None:
                # NO TRADE
                reward = 0.0
                exit_price = None
                self.last_trade_info = {
                    "entry_price": None,
                    "exit_price": None,
                    "pnl": 0.0
                }
            else:
                # direction == 0 or 1 --> short or long
                entry_price = self.fd.loc[self.current_step, "Close"]
                reward = self._calculate_reward(direction, sl, tp)

                # Next bar's close if possible
                if self.current_step < self.n_steps - 1:
                    exit_price = self.df.loc[self.current_step + 1, "Close"]
                else:
                    exit_price = entry_price
                
                self.last_trade_info = {
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": reward / 10000 # convert back to pips
                }

                # Update equity
                self.equity += reward
            
            # Log equity (if no trade, equity remains the same)
            self.equity_curve.append(self.equity)
            
            # Update current step
            self.current_step += 1
            if self.current_step >= self.n_steps - 1:
                self.done = True
            else:
                self.done = False
            
            # Observe next state
            obs = self._get_observation()

            return obs, reward, self.done, {}
        
        def reset(self):
            self.current_step = self.window_size
            self.equity = 10000.0
            self.done = False
            self.equity_curve = []
            self.last_trade_info = None
            return self._get_observation()
        
        def render(self, mode='human'):
            print(f"Step: {self.current_step}, Equity: {self.equity:.2f}, Last Trade: {self.last_trade_info}")