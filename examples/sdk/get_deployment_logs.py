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
        if not pods:
            print("No pods have logged for this revision yet.")
            return

        pod = pods[0]

        # A window read: start_time/end_time are epoch ms, inclusive. With end_time
        # set the iterator terminates once the window is delivered or the store has
        # no more lines to give. fetch_logs is lazy — each server page is yielded as
        # one chunk, and only a short dedup window is held however large the read.
        # chunk_size is also the number of lines requested per round trip, so a bulk
        # read wants a large value.
        now_ms = int(time.time() * 1000)
        print(f"Last {WINDOW_MINUTES} minutes of pod {pod}:\n")
        count = 0
        for chunk in cclient.fetch_logs(
            DEPLOYMENT_ID,
            REVISION_NUMBER,
            pod,
            start_time=now_ms - WINDOW_MINUTES * 60_000,
            end_time=now_ms,
            chunk_size=1000,
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
