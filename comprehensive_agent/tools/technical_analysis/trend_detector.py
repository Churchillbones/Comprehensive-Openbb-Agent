"""
Trend Detector Tool

Identify trends, trend strength, and trend reversals.
Extracted from: comprehensive_agent/core/ml_widget_bridge.py
NEW tool created for Technical Analysis Agent.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


async def detect_trend(
    data: Any,
    method: str = "linear",
    sensitivity: str = "medium"
) -> Dict[str, Any]:
    """
    Detect trend in price data

    Args:
        data: Price data (list of dicts or numpy array)
        method: Trend detection method:
            - "linear": Linear regression slope
            - "moving_average": Moving average crossover
            - "peaks_troughs": Peak and trough analysis
        sensitivity: Trend sensitivity ("low", "medium", "high")

    Returns:
        Dict with trend analysis:
            - status: "success" or "error"
            - trend_direction: "uptrend", "downtrend", or "sideways"
            - trend_strength: Strength score (0-100)
            - trend_angle: Trend line angle in degrees
            - reversal_signals: List of potential reversal points
            - insights: Human-readable insights
    """
    try:
        # Extract price data
        prices = _extract_prices(data)

        if len(prices) < 5:
            return {
                "status": "error",
                "error": "Insufficient data for trend detection (need at least 5 points)"
            }

        # Set sensitivity thresholds
        thresholds = {
            "low": {"slope": 0.5, "strength": 20},
            "medium": {"slope": 0.2, "strength": 10},
            "high": {"slope": 0.1, "strength": 5}
        }
        threshold = thresholds.get(sensitivity, thresholds["medium"])

        result = {}

        if method == "linear":
            result = _detect_trend_linear(prices, threshold)

        elif method == "moving_average":
            result = _detect_trend_ma_crossover(prices, threshold)

        elif method == "peaks_troughs":
            result = _detect_trend_peaks_troughs(prices, threshold)

        else:
            return {
                "status": "error",
                "error": f"Unknown trend detection method: {method}"
            }

        # Add insights
        result["insights"] = _generate_trend_insights(result)

        logger.debug(f"Trend detected: {result.get('trend_direction')} "
                    f"(strength: {result.get('trend_strength')})")

        return {
            "status": "success",
            "method": method,
            "sensitivity": sensitivity,
            **result
        }

    except Exception as e:
        logger.error(f"Trend detection failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }


def _extract_prices(data: Any) -> np.ndarray:
    """Extract price array from various data formats"""
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


def _detect_trend_linear(prices: np.ndarray, threshold: Dict) -> Dict[str, Any]:
    """Detect trend using linear regression"""
    if len(prices) < 2:
        return {"trend_direction": "unknown", "trend_strength": 0}

    # Linear regression
    x = np.arange(len(prices))
    slope, intercept = np.polyfit(x, prices, 1)

    # Normalize slope by average price
    avg_price = np.mean(prices)
    normalized_slope = slope / avg_price * 100 if avg_price != 0 else 0

    # Calculate R-squared for trend strength
    y_pred = slope * x + intercept
    ss_res = np.sum((prices - y_pred) ** 2)
    ss_tot = np.sum((prices - np.mean(prices)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    trend_strength = abs(r_squared) * 100  # 0-100 scale

    # Determine trend direction
    if abs(normalized_slope) < threshold["slope"]:
        trend_direction = "sideways"
    elif normalized_slope > 0:
        trend_direction = "uptrend"
    else:
        trend_direction = "downtrend"

    # Calculate trend angle
    trend_angle = np.degrees(np.arctan(normalized_slope))

    # Detect potential reversals (price deviating from trend line)
    deviations = prices - y_pred
    std_dev = np.std(deviations)
    reversal_signals = []

    if len(prices) >= 3:
        # Check last few points for reversal
        recent_deviation = abs(deviations[-1])
        if recent_deviation > 2 * std_dev:
            reversal_signals.append({
                "type": "deviation",
                "strength": "strong",
                "message": "Price significantly deviating from trend"
            })

        # Check for slope change in recent data
        if len(prices) >= 20:
            recent_slope = np.polyfit(x[-10:], prices[-10:], 1)[0]
            recent_normalized = recent_slope / avg_price * 100

            if (normalized_slope > 0 and recent_normalized < -threshold["slope"]) or \
               (normalized_slope < 0 and recent_normalized > threshold["slope"]):
                reversal_signals.append({
                    "type": "slope_change",
                    "strength": "moderate",
                    "message": "Recent trend direction changing"
                })

    return {
        "trend_direction": trend_direction,
        "trend_strength": round(trend_strength, 2),
        "trend_angle": round(trend_angle, 2),
        "slope": round(normalized_slope, 4),
        "r_squared": round(r_squared, 4),
        "reversal_signals": reversal_signals
    }


def _detect_trend_ma_crossover(prices: np.ndarray, threshold: Dict) -> Dict[str, Any]:
    """Detect trend using moving average crossover"""
    short_period = 20
    long_period = 50

    if len(prices) < long_period:
        # Fall back to linear method
        return _detect_trend_linear(prices, threshold)

    # Calculate moving averages
    short_ma = np.convolve(prices, np.ones(short_period)/short_period, mode='valid')
    long_ma = np.convolve(prices, np.ones(long_period)/long_period, mode='valid')

    # Align arrays
    min_len = min(len(short_ma), len(long_ma))
    short_ma = short_ma[-min_len:]
    long_ma = long_ma[-min_len:]

    # Determine trend
    if short_ma[-1] > long_ma[-1]:
        trend_direction = "uptrend"
    elif short_ma[-1] < long_ma[-1]:
        trend_direction = "downtrend"
    else:
        trend_direction = "sideways"

    # Calculate strength based on separation
    separation = abs(short_ma[-1] - long_ma[-1]) / long_ma[-1] * 100
    trend_strength = min(100, separation * 10)  # Scale to 0-100

    # Detect crossovers (reversals)
    reversal_signals = []
    if len(short_ma) >= 2:
        prev_diff = short_ma[-2] - long_ma[-2]
        curr_diff = short_ma[-1] - long_ma[-1]

        if prev_diff < 0 and curr_diff > 0:
            reversal_signals.append({
                "type": "golden_cross",
                "strength": "strong",
                "message": "Bullish crossover detected (Golden Cross)"
            })
        elif prev_diff > 0 and curr_diff < 0:
            reversal_signals.append({
                "type": "death_cross",
                "strength": "strong",
                "message": "Bearish crossover detected (Death Cross)"
            })

    return {
        "trend_direction": trend_direction,
        "trend_strength": round(trend_strength, 2),
        "short_ma": round(short_ma[-1], 2),
        "long_ma": round(long_ma[-1], 2),
        "separation_pct": round(separation, 2),
        "reversal_signals": reversal_signals
    }


def _detect_trend_peaks_troughs(prices: np.ndarray, threshold: Dict) -> Dict[str, Any]:
    """Detect trend using peak and trough analysis"""
    if len(prices) < 10:
        return _detect_trend_linear(prices, threshold)

    # Find peaks and troughs
    peaks = []
    troughs = []

    for i in range(1, len(prices) - 1):
        if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
            peaks.append((i, prices[i]))
        elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
            troughs.append((i, prices[i]))

    # Analyze peaks and troughs progression
    if len(peaks) >= 2 and len(troughs) >= 2:
        # Higher highs and higher lows = uptrend
        peaks_rising = all(peaks[i][1] < peaks[i+1][1] for i in range(len(peaks)-1))
        troughs_rising = all(troughs[i][1] < troughs[i+1][1] for i in range(len(troughs)-1))

        # Lower highs and lower lows = downtrend
        peaks_falling = all(peaks[i][1] > peaks[i+1][1] for i in range(len(peaks)-1))
        troughs_falling = all(troughs[i][1] > troughs[i+1][1] for i in range(len(troughs)-1))

        if peaks_rising and troughs_rising:
            trend_direction = "uptrend"
            trend_strength = 70
        elif peaks_falling and troughs_falling:
            trend_direction = "downtrend"
            trend_strength = 70
        else:
            trend_direction = "sideways"
            trend_strength = 30
    else:
        # Fall back to linear
        return _detect_trend_linear(prices, threshold)

    return {
        "trend_direction": trend_direction,
        "trend_strength": trend_strength,
        "peaks_count": len(peaks),
        "troughs_count": len(troughs),
        "reversal_signals": []
    }


def _generate_trend_insights(trend_data: Dict[str, Any]) -> List[str]:
    """Generate human-readable insights from trend data"""
    insights = []

    direction = trend_data.get("trend_direction", "unknown")
    strength = trend_data.get("trend_strength", 0)

    # Main trend insight
    if direction == "uptrend":
        if strength > 70:
            insights.append("📈 Strong uptrend detected")
        elif strength > 40:
            insights.append("📈 Moderate uptrend in progress")
        else:
            insights.append("📈 Weak uptrend forming")
    elif direction == "downtrend":
        if strength > 70:
            insights.append("📉 Strong downtrend detected")
        elif strength > 40:
            insights.append("📉 Moderate downtrend in progress")
        else:
            insights.append("📉 Weak downtrend forming")
    else:
        insights.append("➡️ Price moving sideways (no clear trend)")

    # Reversal signals
    reversal_signals = trend_data.get("reversal_signals", [])
    for signal in reversal_signals:
        insights.append(f"⚠️ {signal.get('message', 'Reversal signal detected')}")

    # Trend angle insight
    angle = trend_data.get("trend_angle")
    if angle is not None:
        if abs(angle) > 45:
            insights.append(f"Steep trend angle: {abs(angle):.1f}°")

    return insights
