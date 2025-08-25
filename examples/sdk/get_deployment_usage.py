import math

from centml.sdk.api import get_centml_client
from centml.sdk import Metric


HOUR_IN_SECONDS = 60 * 60
DAY_IN_SECONDS = 24 * HOUR_IN_SECONDS
MAX_DATA_POINTS = 10_000


def get_step_size(start_time_in_seconds: int, end_time_in_seconds: int) -> int:
    time_delta_in_seconds = end_time_in_seconds - start_time_in_seconds
    # 0 seconds to 2 days: 60s
    if time_delta_in_seconds <= 2 * DAY_IN_SECONDS:
        return 60
    # 2 days to 7 days: 5m
    elif time_delta_in_seconds <= 7 * DAY_IN_SECONDS:
        return 5 * 60
	#  7 days to 14 days: 10m
    elif time_delta_in_seconds <= 14 * DAY_IN_SECONDS:
        return 10 * 60
	# 14 days to 30 days: 30m
    elif time_delta_in_seconds <= 30 * DAY_IN_SECONDS:
        return 30 * 60
	# 30 days to 60 days: 1 hour
    elif time_delta_in_seconds <= 60 * DAY_IN_SECONDS:
        return HOUR_IN_SECONDS
	# 60 days to 90 days: 2 hours
    elif time_delta_in_seconds <= 90 * DAY_IN_SECONDS:
        return 2 * HOUR_IN_SECONDS
	# 90+ days: 3 hours (This catches all ranges greater than 90 days)
    else:
        return 3 * HOUR_IN_SECONDS

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
