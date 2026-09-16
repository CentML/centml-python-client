import time
from datetime import datetime, timezone

from centml.sdk.api import get_centml_client

# --- Configuration ---
DEPLOYMENT_ID = 1234  # Replace with your deployment ID
REVISION_NUMBER = 10
WINDOW_MINUTES = 10  # How far back the window read looks
TAIL_LINES = 20  # How many tailed lines to print before stopping the tail loop
POLL_SECONDS = 2.0


def format_event(event) -> str:
    ts = datetime.fromtimestamp(event.timestamp / 1000, tz=timezone.utc).isoformat()
    return f"[{ts}] {event.pod} {event.message}"


def main():
    with get_centml_client() as cclient:
        # Discover pod names; terminated pods still within log retention are included.
        pods = cclient.get_deployment_pods(DEPLOYMENT_ID, REVISION_NUMBER)
        pod = pods[0]

        # A window read: start_time/end_time are epoch ms, inclusive. With end_time
        # set the iterator terminates once the window is delivered. fetch_logs is
        # lazy — chunks are yielded as they are fetched, with bounded memory
        # however large the window.
        now_ms = int(time.time() * 1000)
        print(f"Last {WINDOW_MINUTES} minutes of pod {pod}:\n")
        count = 0
        for chunk in cclient.fetch_logs(
            DEPLOYMENT_ID, REVISION_NUMBER, pod, start_time=now_ms - WINDOW_MINUTES * 60_000, end_time=now_ms
        ):
            for event in chunk:
                print(format_event(event))
            count += len(chunk)
        print(f"\nThe window holds {count} lines.")

        # A tail: without end_time the same generator never terminates. start_time
        # defaults to the moment of the call, and once caught up the generator
        # yields an empty chunk each time nothing new is stored yet — the caller
        # decides when to sleep or break. No line is ever delivered twice.
        print(f"\nTailing pod {pod}; stopping after {TAIL_LINES} new lines...")
        printed = 0
        for chunk in cclient.fetch_logs(DEPLOYMENT_ID, REVISION_NUMBER, pod):
            if not chunk:
                time.sleep(POLL_SECONDS)
                continue
            for event in chunk:
                print(format_event(event))
            printed += len(chunk)
            if printed >= TAIL_LINES:
                break


if __name__ == "__main__":
    main()
