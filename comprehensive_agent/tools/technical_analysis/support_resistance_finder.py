"""
Support/Resistance Finder Tool

Find key support and resistance levels, and identify breakouts.
NEW tool created for Technical Analysis Agent.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


async def find_support_resistance(
    data: Any,
    method: str = "local_extrema",
    threshold: float = 0.02
) -> Dict[str, Any]:
    """
    Find support and resistance levels

    Args:
        data: Price data (list of dicts with 'high'/'low'/'close' or numpy array)
        method: Detection method:
            - "local_extrema": Find local peaks and troughs
            - "clustering": Cluster price levels
            - "pivot_points": Calculate pivot points
        threshold: Price proximity threshold for level clustering (2% default)

    Returns:
        Dict with support/resistance levels:
            - status: "success" or "error"
            - support_levels: List of support price levels
            - resistance_levels: List of resistance price levels
            - current_price: Current price
            - breakout_signals: Potential breakout/breakdown signals
            - insights: Human-readable insights
    """
    try:
        # Extract price data
        closes = _extract_prices(data, "close")
        highs = _extract_prices(data, "high") if _has_field(data, "high") else closes
        lows = _extract_prices(data, "low") if _has_field(data, "low") else closes

        if len(closes) < 10:
            return {
                "status": "error",
                "error": "Insufficient data for support/resistance detection (need at least 10 points)"
            }

        result = {}

        if method == "local_extrema":
            result = _find_sr_local_extrema(closes, highs, lows, threshold)

        elif method == "clustering":
            result = _find_sr_clustering(closes, highs, lows, threshold)

        elif method == "pivot_points":
            result = _find_sr_pivot_points(closes, highs, lows)

        else:
            return {
                "status": "error",
                "error": f"Unknown method: {method}"
            }

        # Add current price
        result["current_price"] = float(closes[-1])

        # Detect breakouts
        result["breakout_signals"] = _detect_breakouts(
            closes,
            result["support_levels"],
            result["resistance_levels"]
        )

        # Generate insights
        result["insights"] = _generate_sr_insights(result)

        logger.debug(f"Found {len(result['support_levels'])} support and "
                    f"{len(result['resistance_levels'])} resistance levels")

        return {
            "status": "success",
            "method": method,
            **result
        }

    except Exception as e:
        logger.error(f"Support/resistance detection failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }


def _extract_prices(data: Any, field: str = "close") -> np.ndarray:
    """Extract price array from data"""
    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, list):
        if not data:
            return np.array([])

        if isinstance(data[0], dict):
            try:
                prices = [float(item[field]) for item in data if field in item]
                if prices:
                    return np.array(prices)
            except (KeyError, ValueError):
                pass

            # Fallback to other fields
            for fallback in ["close", "price", "value"]:
                try:
                    prices = [float(item[fallback]) for item in data if fallback in item]
                    if prices:
                        return np.array(prices)
                except:
                    continue

        # Try as list of numbers
        try:
            return np.array([float(x) for x in data])
        except:
            pass

    return np.array([])


def _has_field(data: Any, field: str) -> bool:
    """Check if data has a specific field"""
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return field in data[0]
    return False


def _find_sr_local_extrema(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, threshold: float) -> Dict:
    """Find support/resistance using local extrema"""
    # Find local peaks (resistance) and troughs (support)
    resistance_candidates = []
    support_candidates = []

    # Use a window to find local extrema
    window = min(10, len(closes) // 5)

    for i in range(window, len(closes) - window):
        # Check if this is a local maximum
        if highs[i] == max(highs[i-window:i+window+1]):
            resistance_candidates.append(highs[i])

        # Check if this is a local minimum
        if lows[i] == min(lows[i-window:i+window+1]):
            support_candidates.append(lows[i])

    # Cluster nearby levels
    resistance_levels = _cluster_levels(resistance_candidates, threshold)
    support_levels = _cluster_levels(support_candidates, threshold)

    return {
        "resistance_levels": sorted(resistance_levels, reverse=True),
        "support_levels": sorted(support_levels, reverse=True)
    }


def _find_sr_clustering(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray, threshold: float) -> Dict:
    """Find support/resistance using price clustering"""
    # Combine all prices for clustering
    all_prices = np.concatenate([highs, lows, closes])

    # Create histogram of prices
    hist, bin_edges = np.histogram(all_prices, bins=50)

    # Find bins with high concentration (potential levels)
    mean_count = np.mean(hist)
    std_count = np.std(hist)
    threshold_count = mean_count + std_count

    levels = []
    for i, count in enumerate(hist):
        if count > threshold_count:
            # Use bin center as level
            level = (bin_edges[i] + bin_edges[i+1]) / 2
            levels.append(level)

    # Cluster nearby levels
    clustered = _cluster_levels(levels, threshold)

    # Separate into support and resistance based on current price
    current = closes[-1]
    support_levels = [l for l in clustered if l < current]
    resistance_levels = [l for l in clustered if l > current]

    return {
        "resistance_levels": sorted(resistance_levels, reverse=True),
        "support_levels": sorted(support_levels, reverse=True)
    }


def _find_sr_pivot_points(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Dict:
    """Find support/resistance using pivot points"""
    # Use recent period for pivot calculation
    period = min(20, len(closes))

    recent_high = np.max(highs[-period:])
    recent_low = np.min(lows[-period:])
    recent_close = closes[-1]

    # Standard pivot point calculation
    pivot = (recent_high + recent_low + recent_close) / 3

    # Calculate support and resistance levels
    r1 = 2 * pivot - recent_low
    r2 = pivot + (recent_high - recent_low)
    r3 = recent_high + 2 * (pivot - recent_low)

    s1 = 2 * pivot - recent_high
    s2 = pivot - (recent_high - recent_low)
    s3 = recent_low - 2 * (recent_high - pivot)

    return {
        "resistance_levels": [float(r3), float(r2), float(r1)],
        "support_levels": [float(s1), float(s2), float(s3)],
        "pivot_point": float(pivot)
    }


def _cluster_levels(levels: List[float], threshold: float) -> List[float]:
    """Cluster nearby price levels"""
    if not levels:
        return []

    levels = sorted(levels)
    clusters = []
    current_cluster = [levels[0]]

    for level in levels[1:]:
        # Check if level is close to current cluster
        if abs(level - np.mean(current_cluster)) / np.mean(current_cluster) < threshold:
            current_cluster.append(level)
        else:
            # Start new cluster
            clusters.append(np.mean(current_cluster))
            current_cluster = [level]

    # Add last cluster
    if current_cluster:
        clusters.append(np.mean(current_cluster))

    return [round(float(c), 2) for c in clusters]


def _detect_breakouts(closes: np.ndarray, support: List[float], resistance: List[float]) -> List[Dict[str, Any]]:
    """Detect potential breakouts/breakdowns"""
    signals = []

    if len(closes) < 2:
        return signals

    current_price = closes[-1]
    previous_price = closes[-2]

    # Check resistance breakouts
    for r_level in resistance:
        if previous_price < r_level <= current_price:
            signals.append({
                "type": "breakout",
                "level": r_level,
                "direction": "upward",
                "strength": "strong" if current_price > r_level * 1.01 else "weak",
                "message": f"Price breaking above resistance at {r_level:.2f}"
            })

    # Check support breakdowns
    for s_level in support:
        if previous_price > s_level >= current_price:
            signals.append({
                "type": "breakdown",
                "level": s_level,
                "direction": "downward",
                "strength": "strong" if current_price < s_level * 0.99 else "weak",
                "message": f"Price breaking below support at {s_level:.2f}"
            })

    # Check proximity to levels (approaching)
    proximity_threshold = 0.01  # 1%

    for r_level in resistance:
        if abs(current_price - r_level) / r_level < proximity_threshold:
            signals.append({
                "type": "approaching",
                "level": r_level,
                "direction": "resistance",
                "strength": "moderate",
                "message": f"Price approaching resistance at {r_level:.2f}"
            })

    for s_level in support:
        if abs(current_price - s_level) / s_level < proximity_threshold:
            signals.append({
                "type": "approaching",
                "level": s_level,
                "direction": "support",
                "strength": "moderate",
                "message": f"Price approaching support at {s_level:.2f}"
            })

    return signals


def _generate_sr_insights(sr_data: Dict[str, Any]) -> List[str]:
    """Generate human-readable insights"""
    insights = []

    current_price = sr_data.get("current_price", 0)
    resistance = sr_data.get("resistance_levels", [])
    support = sr_data.get("support_levels", [])

    # Nearest levels
    if resistance:
        nearest_resistance = min(resistance, key=lambda x: abs(x - current_price))
        distance_pct = ((nearest_resistance - current_price) / current_price) * 100
        insights.append(f"📊 Nearest resistance at {nearest_resistance:.2f} ({distance_pct:+.1f}%)")

    if support:
        nearest_support = min(support, key=lambda x: abs(x - current_price))
        distance_pct = ((nearest_support - current_price) / current_price) * 100
        insights.append(f"📊 Nearest support at {nearest_support:.2f} ({distance_pct:+.1f}%)")

    # Breakout signals
    breakouts = sr_data.get("breakout_signals", [])
    for signal in breakouts:
        if signal["type"] in ["breakout", "breakdown"]:
            insights.append(f"⚡ {signal['message']}")

    # Level count
    if len(resistance) + len(support) > 5:
        insights.append(f"Found {len(resistance)} resistance and {len(support)} support levels")

    return insights
