from datetime import datetime


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(
        value.replace("Z", "+00:00")
    )


def extract_post_features(
    current: dict,
    previous: dict | None,
    current_time: str,
    previous_time: str | None
) -> dict:

    features = {
        "hot_delta": None,
        "hot_velocity": None,
        "interval_minutes": None
    }

    # 没有历史观测
    if previous is None or previous_time is None:
        return features

    current_hot = current.get("hot_score")
    previous_hot = previous.get("hot_score")

    # 热度数据缺失
    if current_hot is None or previous_hot is None:
        return features

    # 计算时间间隔
    dt = (
        parse_time(current_time)
        - parse_time(previous_time)
    ).total_seconds() / 60

    if dt <= 0:
        return features

    # 计算热度变化
    delta = current_hot - previous_hot

    features["hot_delta"] = delta
    features["hot_velocity"] = delta / dt
    features["interval_minutes"] = dt

    return features
