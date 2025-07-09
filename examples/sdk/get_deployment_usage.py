import math

from centml.sdk.api import get_centml_client
from centml.sdk import Metric


HOUR_IN_SECONDS = 60 * 60
DAY_IN_SECONDS = 24 * HOUR_IN_SECONDS
MAX_DATA_POINTS = 10_000


def get_step_size(start_time_in_seconds: int, end_time_in_seconds: int) -> int:
    time_delta_in_seconds = end_time_in_seconds - start_time_in_seconds
    if time_delta_in_seconds <= 3 * HOUR_IN_SECONDS:
        return 15
    elif time_delta_in_seconds <= 6 * HOUR_IN_SECONDS:
        return 30
    elif time_delta_in_seconds <= 12 * HOUR_IN_SECONDS:
        return 60
    elif time_delta_in_seconds <= 24 * HOUR_IN_SECONDS:
        return 2 * 60
    elif time_delta_in_seconds <= 2 * DAY_IN_SECONDS:
        return 2 * 60
    elif time_delta_in_seconds <= 7 * DAY_IN_SECONDS:
        return 10 * 60
    elif time_delta_in_seconds <= 15 * DAY_IN_SECONDS:
        return 30 * 60
    elif time_delta_in_seconds <= 30 * DAY_IN_SECONDS:
        return 60 * 60
    return math.ceil(time_delta_in_seconds / MAX_DATA_POINTS)


def main():
    with get_centml_client() as cclient:
        start_time = 1752084581
        end_time = 1752085181
        deployment_usage_values = cclient.get_deployment_usage(
            id=3801,
            metric=Metric.GPU,
            start_time_in_seconds=start_time,
            end_time_in_seconds=end_time,
            step=get_step_size(start_time, end_time),
        )
        print("Deployment usage values:", deployment_usage_values)


if __name__ == "__main__":
    main()
