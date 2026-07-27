import os
import time
import requests
import datetime
import pytz
import pandas as pd
import numpy as np
import upstox_client
from upstox_client.rest import ApiException

# ============================================================================
# SYSTEM ENVIRONMENT VARIABLES (Fetched from Railway)
# ============================================================================
UPSTOX_ACCESS_TOKEN  = os.environ.get("UPSTOX_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID     = os.environ.get("TELEGRAM_CHAT_ID")
NIFTY_INSTRUMENT_KEY = os.environ.get("NIFTY_INSTRUMENT_KEY", "NSE_INDEX|Nifty 50")

# Validation check for required variables
if not UPSTOX_ACCESS_TOKEN or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("CRITICAL: Missing environment variables! Please set UPSTOX_ACCESS_TOKEN, TELEGRAM_BOT_TOKEN, and TELEGRAM_CHAT_ID.")

# Strategy Parameters
CANDLE_INTERVAL        = "5minute"
SUPER_TREND_PERIOD     = 10
SUPER_TREND_MULTIPLIER = 3.0
MIN_BREAKOUT_BODY_PTS  = 12.0
SL_BUFFER_PTS          = 5.0

last_alert_time = None
IST = pytz.timezone('Asia/Kolkata')


# ============================================================================
# MARKET TIME SCHEDULER & SLEEP LOGIC
# ============================================================================
def is_market_open():
    """Checks if the current time is Monday-Friday between 09:00 AM and 03:30 PM IST."""
    now_ist = datetime.datetime.now(IST)
    
    # Monday = 0, Friday = 4, Saturday = 5, Sunday = 6
    if now_ist.weekday() >= 5:
        return False
        
    market_start = now_ist.replace(hour=9, minute=0, second=0, microsecond=0)
    market_end   = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_start <= now_ist <= market_end


def calculate_sleep_seconds():
    """Calculates sleep duration until next market open (09:00 AM IST next trading day)."""
    now_ist = datetime.datetime.now(IST)
    target_time = now_ist.replace(hour=9, minute=0, second=0, microsecond=0)
    
    # If today is after 03:30 PM or on weekend, advance target date
    if now_ist >= now_ist.replace(hour=15, minute=30, second=0, microsecond=0):
        target_time += datetime.timedelta(days=1)
        
    while target_time.weekday() >= 5:  # Skip Saturday (5) & Sunday (6)
        target_time += datetime.timedelta(days=1)
        
    sleep_seconds = (target_time - now_ist).total_seconds()
    return max(sleep_seconds, 60)


# ============================================================================
# TELEGRAM ALERT FUNCTION
# ============================================================================
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"[{datetime.datetime.now(IST).strftime('%H:%M:%S')}] Telegram alert sent successfully.")
        else:
            print(f"[{datetime.datetime.now(IST).strftime('%H:%M:%S')}] Telegram Error: {response.text}")
    except Exception as e:
        print(f"[{datetime.datetime.now(IST).strftime('%H:%M:%S')}] Telegram Connection Error: {e}")


# ============================================================================
# TECHNICAL INDICATOR CALCULATIONS
# ============================================================================
def calculate_vwap(df):
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    df['cum_tp_vol'] = df.groupby('date').apply(
        lambda x: (x['typical_price'] * (x['volume'] + 1)).cumsum()
    ).reset_index(level=0, drop=True)
    
    df['cum_vol'] = df.groupby('date')['volume'].apply(lambda x: (x + 1).cumsum()).reset_index(level=0, drop=True)
    df['vwap'] = np.where(df['cum_vol'] > 0, df['cum_tp_vol'] / df['cum_vol'], df['typical_price'].rolling(20).mean())
    return df


def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df


def calculate_supertrend(df, period=10, multiplier=3.0):
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    st_signal = [True] * len(df)
    
    for i in range(1, len(df)):
        if close.iloc[i] > upperband.iloc[i-1]:
            st_signal[i] = True
        elif close.iloc[i] < lowerband.iloc[i-1]:
            st_signal[i] = False
        else:
            st_signal[i] = st_signal[i-1]
            if st_signal[i] and lowerband.iloc[i] < lowerband.iloc[i-1]:
                lowerband.iloc[i] = lowerband.iloc[i-1]
            if not st_signal[i] and upperband.iloc[i] > upperband.iloc[i-1]:
                upperband.iloc[i] = upperband.iloc[i-1]
                
    df['supertrend_signal'] = st_signal
    return df


# ============================================================================
# UPSTOX DATA FETCHING
# ============================================================================
def fetch_historical_candles():
    configuration = upstox_client.Configuration()
    configuration.access_token = UPSTOX_ACCESS_TOKEN
    api_instance = upstox_client.HistoryApi(upstox_client.ApiClient(configuration))
    
    try:
        api_version = "2.0"
        
        # 1. Use get_intra_day_candle_data for live current day 1-minute candles
        api_response = api_instance.get_intra_day_candle_data(
            NIFTY_INSTRUMENT_KEY,
            "1minute",
            api_version
        )
        
        candles = api_response.data.candles
        if not candles:
            return None

        # Upstox returns: [timestamp, open, high, low, close, volume, oi]
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
        
        # 2. Convert timestamps and sort ASCENDING (oldest to newest)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
        
        # 3. Resample 1-minute candles into standard 5-minute candles
        df.set_index('timestamp', inplace=True)
        df_5m = df.resample('5min', label='left', closed='left').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'oi': 'last'
        }).dropna().reset_index()
        
        return df_5m

    except ApiException as e:
        print(f"[{datetime.datetime.now(IST).strftime('%H:%M:%S')}] Upstox API Exception: {e}")
        return None
# ============================================================================
# STRATEGY EVALUATION & ALERT LOGIC
# ============================================================================
def process_strategy(df):
    global last_alert_time  # Declared at top of function
    
    if df is None or len(df) < 30:
        print(f"[{datetime.datetime.now(IST).strftime('%H:%M:%S')}] Waiting for enough candle history (need >= 30 candles)...")
        return

    # Calculate indicators
    df = calculate_vwap(df)
    df = calculate_macd(df)
    df = calculate_supertrend(df, SUPER_TREND_PERIOD, SUPER_TREND_MULTIPLIER)
    
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    
    candle_time = latest['timestamp']

    # Candle Structure Metrics
    body       = abs(latest['close'] - latest['open'])
    lower_wick = min(latest['open'], latest['close']) - latest['low']
    upper_wick = latest['high'] - max(latest['open'], latest['close'])

    is_green_candle = latest['close'] > latest['open']
    is_red_candle   = latest['close'] < latest['open']

    st_flipped_green = (latest['supertrend_signal'] == True) and (prev['supertrend_signal'] == False)
    st_flipped_red   = (latest['supertrend_signal'] == False) and (prev['supertrend_signal'] == True)

    # Indicator Conditions
    ce_cond_vwap       = latest['close'] > latest['vwap']
    ce_cond_supertrend = latest['supertrend_signal'] == True
    ce_cond_macd       = latest['macd_hist'] > prev['macd_hist']

    ce_trigger_pullback = lower_wick > (body * 1.0)
    ce_trigger_breakout = is_green_candle and (body >= MIN_BREAKOUT_BODY_PTS) and (st_flipped_green or latest['close'] > prev['high'])

    pe_cond_vwap       = latest['close'] < latest['vwap']
    pe_cond_supertrend = latest['supertrend_signal'] == False
    pe_cond_macd       = latest['macd_hist'] < prev['macd_hist']

    pe_trigger_pullback = upper_wick > (body * 1.0)
    pe_trigger_breakout = is_red_candle and (body >= MIN_BREAKOUT_BODY_PTS) and (st_flipped_red or latest['close'] < prev['low'])

    # PRINT DIAGNOSTIC LOGS TO CONSOLE
    print("--------------------------------------------------------------------------------")
    print(f"⏰ [SCAN LOG] Candle Time: {candle_time.strftime('%H:%M:%S')} | Spot Close: {latest['close']}")
    print(f"📊 Indicators  -> VWAP: {round(latest['vwap'], 2)} | SuperTrend: {'GREEN' if latest['supertrend_signal'] else 'RED'} | MACD Hist: {round(latest['macd_hist'], 2)} (Prev: {round(prev['macd_hist'], 2)})")
    print(f"🕯️ Candle Body -> Body: {round(body, 2)} pts | Lower Wick: {round(lower_wick, 2)} | Upper Wick: {round(upper_wick, 2)}")
    print(f"🚀 CE Status   -> VWAP: {ce_cond_vwap} | SuperTrend: {ce_cond_supertrend} | MACD Uptick: {ce_cond_macd} | Pullback: {ce_trigger_pullback} | Breakout: {ce_trigger_breakout}")
    print(f"🔻 PE Status   -> VWAP: {pe_cond_vwap} | SuperTrend: {pe_cond_supertrend} | MACD Downtick: {pe_cond_macd} | Pullback: {pe_trigger_pullback} | Breakdown: {pe_trigger_breakout}")
    print("--------------------------------------------------------------------------------")

    if last_alert_time == candle_time:
        print(f"[{datetime.datetime.now(IST).strftime('%H:%M:%S')}] Alert already sent for candle {candle_time.strftime('%H:%M:%S')}. Skipping.")
        return

    atm_strike = round(latest['close'] / 50) * 50

    # 1. CALL OPTION (CE) TRIGGER
    if ce_cond_vwap and ce_cond_supertrend and ce_cond_macd and (ce_trigger_pullback or ce_trigger_breakout):
        signal_type   = "Pullback Rejection" if ce_trigger_pullback else "Breakout Momentum"
        sl_spot_price = round(latest['low'] - SL_BUFFER_PTS, 2)
        risk_points   = round(latest['close'] - sl_spot_price, 2)
        target_1      = round(latest['close'] + (risk_points * 1.5), 2)
        target_2      = round(latest['close'] + (risk_points * 2.0), 2)

        message = (
            f"🚀 *BUY CALL OPTION (CE) ALERT* 🚀\n\n"
            f"📊 *Broker:* Upstox\n"
            f"📊 *Index:* Nifty 50 (5-Min)\n"
            f"💲 *Nifty Spot Price:* `{latest['close']}`\n"
            f"🎯 *Suggested Strike:* `{atm_strike} CE` (ATM)\n\n"
            f"🛑 *Spot Stop Loss:* `{sl_spot_price}` (-{risk_points} pts)\n"
            f"🎯 *Target 1 (1:1.5):* `{target_1}`\n"
            f"🎯 *Target 2 (1:2.0):* `{target_2}`\n\n"
            f"📈 *Session VWAP:* `{round(latest['vwap'], 2)}`\n"
            f"⚡ *Signal Type:* `{signal_type}`\n"
            f"⏰ *Candle Time:* `{candle_time.strftime('%H:%M:%S')}`"
        )
        send_telegram_alert(message)
        last_alert_time = candle_time
        return

    # 2. PUT OPTION (PE) TRIGGER
    if pe_cond_vwap and pe_cond_supertrend and pe_cond_macd and (pe_trigger_pullback or pe_trigger_breakout):
        signal_type   = "Pullback Rejection" if pe_trigger_pullback else "Breakdown Momentum"
        sl_spot_price = round(latest['high'] + SL_BUFFER_PTS, 2)
        risk_points   = round(sl_spot_price - latest['close'], 2)
        target_1      = round(latest['close'] - (risk_points * 1.5), 2)
        target_2      = round(latest['close'] - (risk_points * 2.0), 2)

        message = (
            f"🔻 *BUY PUT OPTION (PE) ALERT* 🔻\n\n"
            f"📊 *Broker:* Upstox\n"
            f"📊 *Index:* Nifty 50 (5-Min)\n"
            f"💲 *Nifty Spot Price:* `{latest['close']}`\n"
            f"🎯 *Suggested Strike:* `{atm_strike} PE` (ATM)\n\n"
            f"🛑 *Spot Stop Loss:* `{sl_spot_price}` (+{risk_points} pts)\n"
            f"🎯 *Target 1 (1:1.5):* `{target_1}`\n"
            f"🎯 *Target 2 (1:2.0):* `{target_2}`\n\n"
            f"📉 *Session VWAP:* `{round(latest['vwap'], 2)}`\n"
            f"⚡ *Signal Type:* `{signal_type}`\n"
            f"⏰ *Candle Time:* `{candle_time.strftime('%H:%M:%S')}`"
        )
        send_telegram_alert(message)
        last_alert_time = candle_time
        return


# ============================================================================
# MAIN LOOP WITH SLEEP SCHEDULER
# ============================================================================
if __name__ == "__main__":
    print("==================================================")
    print("  Railway Upstox Nifty 50 Scanner Starting...     ")
    print("==================================================")
    
    while True:
        try:
            if is_market_open():
                print(f"[{datetime.datetime.now(IST).strftime('%H:%M:%S')}] Market Open. Scanning Upstox candles...")
                df_candles = fetch_historical_candles()
                if df_candles is not None:
                    process_strategy(df_candles)
                time.sleep(30)
            else:
                sleep_sec = calculate_sleep_seconds()
                hours_sleep = round(sleep_sec / 3600, 2)
                print(f"[{datetime.datetime.now(IST).strftime('%H:%M:%S')}] Market Closed. Going to sleep for {hours_sleep} hours until next market open...")
                time.sleep(sleep_sec)
                
        except Exception as err:
            print(f"[{datetime.datetime.now(IST).strftime('%H:%M:%S')}] Runtime Exception: {err}")
            time.sleep(30)