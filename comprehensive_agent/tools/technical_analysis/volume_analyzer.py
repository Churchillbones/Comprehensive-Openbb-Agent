"""
Volume Analyzer Tool

Analyze volume patterns, detect volume spikes, and calculate VWAP.
NEW tool created for Technical Analysis Agent.
"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


async def analyze_volume(
    data: Any,
    include_vwap: bool = True,
    detect_spikes: bool = True
) -> Dict[str, Any]:
    """
    Analyze volume patterns

    Args:
        data: OHLCV data (must include volume field)
        include_vwap: Calculate Volume Weighted Average Price
        detect_spikes: Detect abnormal volume spikes

    Returns:
        Dict with volume analysis:
            - status: "success" or "error"
            - average_volume: Average volume over period
            - current_volume: Most recent volume
            - volume_trend: "increasing", "decreasing", or "stable"
            - vwap: Volume Weighted Average Price (if requested)
            - volume_spikes: List of volume spike events
            - volume_profile: Distribution of volume by price level
            - insights: Human-readable insights
    """
    try:
        # Extract volume and price data
        volumes = _extract_field(data, "volume")
        closes = _extract_field(data, "close")

        if volumes is None or len(volumes) == 0:
            return {
                "status": "error",
                "error": "No volume data found in input"
            }

        if len(volumes) < 5:
            return {
                "status": "error",
                "error": "Insufficient data for volume analysis (need at least 5 points)"
            }

        result = {}

        # Basic volume statistics
        avg_volume = float(np.mean(volumes))
        current_volume = float(volumes[-1])

        result["average_volume"] = round(avg_volume, 2)
        result["current_volume"] = round(current_volume, 2)
        result["volume_ratio"] = round(current_volume / avg_volume, 2) if avg_volume > 0 else 0

        # Volume trend
        result["volume_trend"] = _detect_volume_trend(volumes)

        # Calculate VWAP if requested and price data available
        if include_vwap and closes is not None:
            highs = _extract_field(data, "high")
            lows = _extract_field(data, "low")

            vwap_result = _calculate_vwap(closes, highs, lows, volumes)
            result["vwap"] = vwap_result

        # Detect volume spikes
        if detect_spikes:
            spikes = _detect_volume_spikes(volumes, closes)
            result["volume_spikes"] = spikes

        # Volume profile (distribution)
        if closes is not None:
            profile = _calculate_volume_profile(closes, volumes)
            result["volume_profile"] = profile

        # Generate insights
        result["insights"] = _generate_volume_insights(result)

        logger.debug(f"Volume analysis: avg={avg_volume:.0f}, current={current_volume:.0f}, "
                    f"ratio={result['volume_ratio']:.2f}")

        return {
            "status": "success",
            **result
        }

    except Exception as e:
        logger.error(f"Volume analysis failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }


def _extract_field(data: Any, field: str) -> Optional[np.ndarray]:
    """Extract specific field from data"""
    if isinstance(data, list) and data and isinstance(data[0], dict):
        try:
            values = [float(item[field]) for item in data if field in item]
            return np.array(values) if values else None
        except (KeyError, ValueError):
            return None
    return None


def _detect_volume_trend(volumes: np.ndarray) -> str:
    """Detect if volume is increasing, decreasing, or stable"""
    if len(volumes) < 10:
        return "insufficient_data"

    # Compare recent vs earlier volume
    recent_avg = np.mean(volumes[-10:])
    earlier_avg = np.mean(volumes[-20:-10]) if len(volumes) >= 20 else np.mean(volumes[:-10])

    if earlier_avg == 0:
        return "stable"

    change_pct = (recent_avg - earlier_avg) / earlier_avg

    if change_pct > 0.15:
        return "increasing"
    elif change_pct < -0.15:
        return "decreasing"
    else:
        return "stable"


def _calculate_vwap(closes: np.ndarray, highs: Optional[np.ndarray],
                   lows: Optional[np.ndarray], volumes: np.ndarray) -> Dict[str, Any]:
    """Calculate Volume Weighted Average Price"""
    if highs is not None and lows is not None:
        # Typical price = (high + low + close) / 3
        typical_price = (highs + lows + closes) / 3
    else:
        # Use close price as fallback
        typical_price = closes

    # VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
    cumulative_tp_volume = np.cumsum(typical_price * volumes)
    cumulative_volume = np.cumsum(volumes)

    vwap = cumulative_tp_volume / cumulative_volume

    current_price = closes[-1]
    current_vwap = vwap[-1]

    # Price position relative to VWAP
    if current_price > current_vwap:
        position = "above"
        signal = "bullish"
    elif current_price < current_vwap:
        position = "below"
        signal = "bearish"
    else:
        position = "at"
        signal = "neutral"

    deviation_pct = ((current_price - current_vwap) / current_vwap) * 100

    return {
        "values": vwap.tolist(),
        "current": round(float(current_vwap), 2),
        "price_position": position,
        "signal": signal,
        "deviation_pct": round(deviation_pct, 2)
    }


def _detect_volume_spikes(volumes: np.ndarray, closes: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    """Detect abnormal volume spikes"""
    spikes = []

    # Calculate rolling statistics
    window = min(20, len(volumes) // 2)
    mean_volume = np.mean(volumes[-window:])
    std_volume = np.std(volumes[-window:])

    # Define spike threshold (2 standard deviations above mean)
    spike_threshold = mean_volume + (2 * std_volume)

    # Check recent volumes for spikes
    check_period = min(10, len(volumes))
    for i in range(-check_period, 0):
        idx = len(volumes) + i
        if volumes[idx] > spike_threshold:
            spike_data = {
                "index": idx,
                "volume": round(float(volumes[idx]), 2),
                "volume_ratio": round(volumes[idx] / mean_volume, 2),
                "significance": "high" if volumes[idx] > spike_threshold * 1.5 else "moderate"
            }

            # Add price change if available
            if closes is not None and idx > 0:
                price_change_pct = ((closes[idx] - closes[idx-1]) / closes[idx-1]) * 100
                spike_data["price_change_pct"] = round(price_change_pct, 2)

                # Determine if it's bullish or bearish volume
                if price_change_pct > 0:
                    spike_data["type"] = "bullish"
                    spike_data["message"] = f"High volume buying detected ({spike_data['volume_ratio']:.1f}x average)"
                elif price_change_pct < 0:
                    spike_data["type"] = "bearish"
                    spike_data["message"] = f"High volume selling detected ({spike_data['volume_ratio']:.1f}x average)"
                else:
                    spike_data["type"] = "neutral"
                    spike_data["message"] = f"High volume with minimal price change ({spike_data['volume_ratio']:.1f}x average)"
            else:
                spike_data["type"] = "unknown"
                spike_data["message"] = f"Volume spike detected ({spike_data['volume_ratio']:.1f}x average)"

            spikes.append(spike_data)

    return spikes


def _calculate_volume_profile(closes: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
    """Calculate volume distribution by price level"""
    if len(closes) < 10:
        return {}

    # Create price bins
    num_bins = min(20, len(closes) // 2)
    price_range = np.max(closes) - np.min(closes)

    if price_range == 0:
        return {}

    bin_edges = np.linspace(np.min(closes), np.max(closes), num_bins + 1)

    # Aggregate volume by price level
    volume_by_level = np.zeros(num_bins)

    for i, price in enumerate(closes):
        bin_idx = np.searchsorted(bin_edges[:-1], price, side='right') - 1
        bin_idx = max(0, min(num_bins - 1, bin_idx))
        volume_by_level[bin_idx] += volumes[i]

    # Find high volume nodes (HVN) - price levels with highest volume
    hvn_threshold = np.mean(volume_by_level) + np.std(volume_by_level)
    hvn_indices = np.where(volume_by_level > hvn_threshold)[0]

    high_volume_nodes = []
    for idx in hvn_indices:
        price_level = (bin_edges[idx] + bin_edges[idx + 1]) / 2
        high_volume_nodes.append({
            "price_level": round(float(price_level), 2),
            "volume": round(float(volume_by_level[idx]), 2)
        })

    # Find Point of Control (POC) - price with highest volume
    poc_idx = np.argmax(volume_by_level)
    poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2

    return {
        "point_of_control": round(float(poc_price), 2),
        "high_volume_nodes": high_volume_nodes,
        "distribution": {
            "price_levels": [round(float((bin_edges[i] + bin_edges[i+1])/2), 2) for i in range(num_bins)],
            "volumes": [round(float(v), 2) for v in volume_by_level]
        }
    }


def _generate_volume_insights(volume_data: Dict[str, Any]) -> List[str]:
    """Generate human-readable insights from volume analysis"""
    insights = []

    # Current volume vs average
    volume_ratio = volume_data.get("volume_ratio", 0)
    if volume_ratio > 2:
        insights.append(f"📊 Extremely high volume ({volume_ratio:.1f}x average)")
    elif volume_ratio > 1.5:
        insights.append(f"📊 Above-average volume ({volume_ratio:.1f}x average)")
    elif volume_ratio < 0.5:
        insights.append(f"📊 Below-average volume ({volume_ratio:.1f}x average)")

    # Volume trend
    trend = volume_data.get("volume_trend")
    if trend == "increasing":
        insights.append("📈 Volume trend is increasing - growing interest")
    elif trend == "decreasing":
        insights.append("📉 Volume trend is decreasing - waning interest")

    # VWAP insights
    if "vwap" in volume_data:
        vwap_data = volume_data["vwap"]
        position = vwap_data.get("price_position")
        deviation = abs(vwap_data.get("deviation_pct", 0))

        if position == "above" and deviation > 2:
            insights.append(f"💰 Price significantly above VWAP ({deviation:.1f}%) - strong buying")
        elif position == "below" and deviation > 2:
            insights.append(f"💰 Price significantly below VWAP ({deviation:.1f}%) - strong selling")

    # Volume spikes
    spikes = volume_data.get("volume_spikes", [])
    recent_spikes = [s for s in spikes if s.get("significance") == "high"]
    if recent_spikes:
        insights.append(f"⚡ {len(recent_spikes)} high-significance volume spike(s) detected")

    # Volume profile
    if "volume_profile" in volume_data:
        profile = volume_data["volume_profile"]
        poc = profile.get("point_of_control")
        if poc:
            insights.append(f"📍 Point of Control (highest volume) at {poc:.2f}")

    return insights
