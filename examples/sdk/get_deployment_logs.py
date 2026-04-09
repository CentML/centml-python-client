from datetime import datetime, timezone, timedelta

from centml.sdk.api import get_centml_client

# --- Configuration ---
DEPLOYMENT_ID = 1234  # Replace with your deployment ID
REVISION_NUMBER = 10
HOURS_BACK = 1  # Fetch logs from the last N hours


def format_event(event: dict) -> str:
    timestamp_ms = (
        event.get("timestamp")
        or event.get("time")
        or event.get("ts")
        or ""
    )
    message = (
        event.get("message")
        or event.get("msg")
        or event.get("log")
        or str(event)
    )
    if timestamp_ms:
        ts = datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc).isoformat()
        return f"[{ts}] {message}"
    return message


def main():
    stream = True
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - int(timedelta(hours=HOURS_BACK).total_seconds() * 1000)

    print(f"Fetching logs for deployment {DEPLOYMENT_ID}")
    print(
        f"Time window: "
        f"{datetime.fromtimestamp(start_time / 1000, tz=timezone.utc).isoformat()} → "
        f"{datetime.fromtimestamp(end_time / 1000, tz=timezone.utc).isoformat()}"
    )
    print()

    with get_centml_client() as cclient:
        if stream:
            # Streaming: print events as each page arrives
            for event in cclient.get_deployment_logs(
                deployment_id=DEPLOYMENT_ID,
                revision_number=REVISION_NUMBER,
                start_time=start_time,
                end_time=end_time,
                start_from_head=False,
                stream=stream,
            ):
                print(format_event(event))
        else:
            # Batch: collect all events then process
            events = cclient.get_deployment_logs(
                deployment_id=DEPLOYMENT_ID,
                revision_number=REVISION_NUMBER,
                start_time=start_time,
                end_time=end_time,
                start_from_head=False,
            )

            if not events:
                print("No logs found in the given time window.")
                return

            print(f"Found {len(events)} log entries:\n")
            for event in events:
                print(format_event(event))


if __name__ == "__main__":
    main()
