import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from indicators import load_and_preprocess_data
from trading_env import TradingEnv

def main():
    # Load and preprocess data
    df = load_and_preprocess_data("data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv")

    # Create environment
    env = TradingEnv(df=df,
                    window_size=30,
                    sl_options=[60, 90, 120], # Example SL distance in pips
                    tp_options=[60, 90, 120]) # Example TP distance in pips
    
    # Create vectorized environment wrapped in DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])

    # Define RL model (PPO)
    # PPO good for financial data as it can handle continuous actions and rewards
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log="./tensorboard_log/"
    )

    # Train the model
    model.learn(total_timesteps=50000)
    model.save("model_eurusd")
    print("Model saved successfully!")

    # Evaluate model
    obs = vec_env.reset()
    done = False
    equity_curve = []

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = vec_env.step(action)

        # Collect equity from unwrapped environment
        # Since already have DummyVecEnv, can access env_method to get attribtue
        current_equity = vec_env.get_attr("equity")[0]
        equity_curve.append(current_equity)

        if done[0]:
            break
    
    # Plot final equity curve
    plt.figure(figsize=(10,6))
    plt.plot(equity_surve, label='Equity')
    plt.title('Equity Curve during Evaluation')
    plt.xlabel('Time Steps')
    plt.ylabel('Equity')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
