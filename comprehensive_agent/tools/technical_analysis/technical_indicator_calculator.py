"""
Technical Indicator Calculator Tool

Calculate major technical indicators (RSI, MACD, Bollinger Bands, Moving Averages, etc.)
NEW tool created for Technical Analysis Agent.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


async def calculate_technical_indicator(
    data: Any,
    indicator_type: str,
    **params
) -> Dict[str, Any]:
    """
    Calculate technical indicators

    Args:
        data: Price data (list of dicts with 'close' or 'price', or numpy array)
        indicator_type: Type of indicator:
            - "rsi": Relative Strength Index
            - "macd": Moving Average Convergence Divergence
            - "bollinger_bands": Bollinger Bands
            - "sma": Simple Moving Average
            - "ema": Exponential Moving Average
            - "stochastic": Stochastic Oscillator
            - "atr": Average True Range
        **params: Indicator-specific parameters (periods, etc.)

    Returns:
        Dict with indicator values:
            - status: "success" or "error"
            - indicator: Indicator type
            - values: Calculated indicator values
            - signals: Buy/sell signals if applicable
            - parameters: Parameters used
    """
    try:
        # Extract price data
        prices = _extract_prices(data)

        if len(prices) < 10:
            return {
                "status": "error",
                "error": "Insufficient data points (need at least 10)",
                "indicator": indicator_type
            }

        result = None

        if indicator_type == "rsi":
            period = params.get("period", 14)
            result = _calculate_rsi(prices, period)

        elif indicator_type == "macd":
            fast_period = params.get("fast_period", 12)
            slow_period = params.get("slow_period", 26)
            signal_period = params.get("signal_period", 9)
            result = _calculate_macd(prices, fast_period, slow_period, signal_period)

        elif indicator_type == "bollinger_bands":
            period = params.get("period", 20)
            std_dev = params.get("std_dev", 2)
            result = _calculate_bollinger_bands(prices, period, std_dev)

        elif indicator_type == "sma":
            period = params.get("period", 20)
            result = _calculate_sma(prices, period)

        elif indicator_type == "ema":
            period = params.get("period", 20)
            result = _calculate_ema(prices, period)

        elif indicator_type == "stochastic":
            k_period = params.get("k_period", 14)
            d_period = params.get("d_period", 3)
            # Need high/low data for stochastic
            highs = _extract_field(data, "high")
            lows = _extract_field(data, "low")
            if highs and lows:
                result = _calculate_stochastic(prices, highs, lows, k_period, d_period)
            else:
                return {
                    "status": "error",
                    "error": "Stochastic requires high/low data",
                    "indicator": indicator_type
                }

        elif indicator_type == "atr":
            period = params.get("period", 14)
            highs = _extract_field(data, "high")
            lows = _extract_field(data, "low")
            if highs and lows:
                result = _calculate_atr(prices, highs, lows, period)
            else:
                return {
                    "status": "error",
                    "error": "ATR requires high/low data",
                    "indicator": indicator_type
                }

        else:
            return {
                "status": "error",
                "error": f"Unknown indicator type: {indicator_type}"
            }

        if result:
            return {
                "status": "success",
                "indicator": indicator_type,
                **result,
                "parameters": params or {}
            }

        return {
            "status": "error",
            "error": f"Failed to calculate {indicator_type}",
            "indicator": indicator_type
        }

    except Exception as e:
        logger.error(f"Indicator calculation failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "indicator": indicator_type
        }


def _extract_prices(data: Any) -> np.ndarray:
    """Extract close/price array from data"""
    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, list):
        if not data:
            return np.array([])

        if isinstance(data[0], dict):
            # Extract 'close' or 'price' field
            for key in ["close", "price", "value"]:
                try:
                    prices = [float(item[key]) for item in data if key in item]
                    if prices:
                        return np.array(prices)
                except (KeyError, ValueError):
                    continue

        # Try as list of numbers
        try:
            return np.array([float(x) for x in data])
        except:
            pass

    return np.array([])


def _extract_field(data: Any, field: str) -> Optional[np.ndarray]:
    """Extract specific field from data"""
    if isinstance(data, list) and data and isinstance(data[0], dict):
        try:
            values = [float(item[field]) for item in data if field in item]
            return np.array(values) if values else None
        except:
            return None
    return None


def _calculate_rsi(prices: np.ndarray, period: int = 14) -> Dict[str, Any]:
    """Calculate Relative Strength Index"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gains = np.zeros(len(prices))
    avg_losses = np.zeros(len(prices))

    # Initial averages
    avg_gains[period] = np.mean(gains[:period])
    avg_losses[period] = np.mean(losses[:period])

    # Smooth subsequent values
    for i in range(period + 1, len(prices)):
        avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
        avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period

    rs = np.where(avg_losses != 0, avg_gains / avg_losses, 0)
    rsi = 100 - (100 / (1 + rs))

    # Generate signals
    current_rsi = float(rsi[-1])
    signals = []
    if current_rsi > 70:
        signals.append("Overbought (RSI > 70)")
    elif current_rsi < 30:
        signals.append("Oversold (RSI < 30)")

    return {
        "values": rsi.tolist(),
        "current": current_rsi,
        "signals": signals
    }


def _calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, Any]:
    """Calculate MACD"""
    ema_fast = _calculate_ema(prices, fast)["values"]
    ema_slow = _calculate_ema(prices, slow)["values"]

    macd_line = np.array(ema_fast) - np.array(ema_slow)
    signal_line = np.array(_calculate_ema(macd_line, signal)["values"])
    histogram = macd_line - signal_line

    # Generate signals
    signals = []
    if len(histogram) >= 2:
        if histogram[-2] < 0 and histogram[-1] > 0:
            signals.append("Bullish crossover")
        elif histogram[-2] > 0 and histogram[-1] < 0:
            signals.append("Bearish crossover")

    return {
        "values": {
            "macd": macd_line.tolist(),
            "signal": signal_line.tolist(),
            "histogram": histogram.tolist()
        },
        "current": {
            "macd": float(macd_line[-1]),
            "signal": float(signal_line[-1]),
            "histogram": float(histogram[-1])
        },
        "signals": signals
    }


def _calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Dict[str, Any]:
    """Calculate Bollinger Bands"""
    sma = _calculate_sma(prices, period)["values"]
    std = np.array([np.std(prices[max(0, i-period):i+1]) for i in range(len(prices))])

    upper_band = np.array(sma) + (std_dev * std)
    lower_band = np.array(sma) - (std_dev * std)

    # Generate signals
    current_price = float(prices[-1])
    current_upper = float(upper_band[-1])
    current_lower = float(lower_band[-1])

    signals = []
    if current_price >= current_upper:
        signals.append("Price at upper band (potential reversal)")
    elif current_price <= current_lower:
        signals.append("Price at lower band (potential reversal)")

    return {
        "values": {
            "upper": upper_band.tolist(),
            "middle": sma,
            "lower": lower_band.tolist()
        },
        "current": {
            "upper": current_upper,
            "middle": float(sma[-1]),
            "lower": current_lower
        },
        "signals": signals
    }


def _calculate_sma(prices: np.ndarray, period: int = 20) -> Dict[str, Any]:
    """Calculate Simple Moving Average"""
    sma = np.convolve(prices, np.ones(period)/period, mode='same')

    return {
        "values": sma.tolist(),
        "current": float(sma[-1]),
        "signals": []
    }


def _calculate_ema(prices: np.ndarray, period: int = 20) -> Dict[str, Any]:
    """Calculate Exponential Moving Average"""
    ema = np.zeros(len(prices))
    ema[0] = prices[0]

    multiplier = 2 / (period + 1)

    for i in range(1, len(prices)):
        ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))

    return {
        "values": ema.tolist(),
        "current": float(ema[-1]),
        "signals": []
    }


def _calculate_stochastic(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                         k_period: int = 14, d_period: int = 3) -> Dict[str, Any]:
    """Calculate Stochastic Oscillator"""
    k_values = np.zeros(len(closes))

    for i in range(k_period - 1, len(closes)):
        period_high = np.max(highs[i-k_period+1:i+1])
        period_low = np.min(lows[i-k_period+1:i+1])

        if period_high != period_low:
            k_values[i] = 100 * (closes[i] - period_low) / (period_high - period_low)

    # %D is SMA of %K
    d_values = _calculate_sma(k_values, d_period)["values"]

    # Signals
    current_k = float(k_values[-1])
    signals = []
    if current_k > 80:
        signals.append("Overbought (%K > 80)")
    elif current_k < 20:
        signals.append("Oversold (%K < 20)")

    return {
        "values": {
            "k": k_values.tolist(),
            "d": d_values
        },
        "current": {
            "k": current_k,
            "d": float(d_values[-1])
        },
        "signals": signals
    }


def _calculate_atr(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, period: int = 14) -> Dict[str, Any]:
    """Calculate Average True Range"""
    true_ranges = np.zeros(len(closes))

    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i-1])
        lc = abs(lows[i] - closes[i-1])
        true_ranges[i] = max(hl, hc, lc)

    atr = _calculate_ema(true_ranges, period)["values"]

    return {
        "values": atr,
        "current": float(atr[-1]),
        "signals": []
    }
