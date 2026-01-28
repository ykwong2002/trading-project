I want to create a news-driven keyword specialised market prediction system that uses sentiment features and contextual bandits with verified rewards to make and continuously improve the next-day predictions.

i.e. Given a keyword/ticker, should the price go up or down tomorrow? And how can we improve the accuracy over time for each specific keyword?

1. Data / Feature layer
Purpose: Turn raw news into a structured context vector
- Everyday, fetch past 30 days of news for a keyword (e.g Gold, NVIDIA) (maybe can cache to reduce API calls?)
    - But this negates the popularity sorting in my news fetching, which may be relevant for the real world as some news only surface upon major events
- Use FinBERT to score daily sentiment
- Aggregate scores into rolling features:
    - 30 day avg sentiment
    - short term vs long term trend
    - volatility in sentiment
Output: Fixed size vector representing the market speculations
--
2. Decision Layer (what the system decides)
Purpose: convert context into action (UP/DOWN)
Approach: Contextual Bandit
    - State: Rolling sentiment features
    - Actions: UP / DOWN
    - Policy: learned mapping from context --> action probabilities
- use a shared backbone which is the common representation learning
- but different keywords use different policies (e.g gold learns differently from nvidia)
--
3. Verified reward loop (how the system learns)
Purpose: improve decisions based on the live outcomes
- make a prediction today
- observe tomorrows actual price movement (yfinance API)
- assign verified reward
    - if correct: +1
    - else: 0 or -1
- update ONLY THE POLICY HEAD for that keyword
- uses RLVR sinec:
    - rewards come from objective market data instead of human feedback (has pros and cons)
    - feedback is delayed but unambiguous
--
Goal:
- NLP Feature engineering (FinBERT)
- time series context design
- RLVR


- Do not use RLVR to update FinBERT model
    - Market can go up when bad sentiment, or go down when good sentiment
    - Do not confuse model with market randomness

- instead, only use RLVR in the decision policy layer (contextual bandit since single step)

- Get data from news article given keyword
- Get daily sentiment
- Get daily open and close (to check if increase or decrease)
- Get 30 day sentiment and open and close
- Implement lightweight policy per keyword
    - Each head outputs binary action (UP / DOWN)
    - Each head independenly trainable while sharing same backbone features
- RLVR (Contextual Bandit Problem)
    - After market close, compute verified reward
    - +1 if prediction direction matches movement of the day
    - -1 otherwise
    - Update only keyword-specific policy head using reward
    - DO NOT FINETUNE FINBERT
- Set up PyTorch for policy head tuning
    - input: features from FinBERT backbone
    - output: probabilities of UP / DOWN
    - learns via reinforcement learning with verifiable rewards
- evaluation:
    - track accuracy, sumulative rewar and rolling Sharpe metrics for each keyword
- optional:
    - maybe can expand to LSTM?
