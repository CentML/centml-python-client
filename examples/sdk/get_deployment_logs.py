import time
from datetime import datetime, timezone

from centml.sdk.api import get_centml_client

# --- Configuration ---
DEPLOYMENT_ID = 1234  # Replace with your deployment ID
REVISION_NUMBER = 10
RECENT_LINES = 20  # How many of the newest stored lines to peek at
TAIL_LINES = 20  # How many tailed lines to print before stopping the tail loop
OVERLAP_MS = 30_000  # Covers the server's ~15s late-arrival re-delivery span
POLL_SECONDS = 2.0


def format_event(event) -> str:
    ts = datetime.fromtimestamp(event.timestamp / 1000, tz=timezone.utc).isoformat()
    return f"[{ts}] {event.pod} {event.message}"


def main():
    with get_centml_client() as cclient:
        # The newest stored lines, without reading the whole history: newest_first
        # walks the window backward, one chunk at a time. Direction changes only
        # the order chunks arrive — lines inside each chunk are always ascending.
        print(f"Newest {RECENT_LINES} lines of deployment {DEPLOYMENT_ID} revision {REVISION_NUMBER}:\n")
        printed = 0
        for chunk in cclient.fetch_logs(DEPLOYMENT_ID, REVISION_NUMBER, newest_first=True, chunk_size=RECENT_LINES):
            for event in chunk:
                print(format_event(event))
            printed += len(chunk)
            if printed >= RECENT_LINES:
                break

        # A full chronological read, all pods merged. fetch_logs is lazy — chunks
        # are yielded as they are fetched, with bounded memory however large the
        # window — and always terminates once caught up. Bound the window with
        # start_time/end_time (epoch ms, inclusive) when the history is long.
        count = 0
        for chunk in cclient.fetch_logs(DEPLOYMENT_ID, REVISION_NUMBER):
            count += len(chunk)
        print(f"\nFull retained history holds {count} lines.")

        # Tailing is a caller loop: re-call a forward fetch_logs with the next
        # window starting OVERLAP_MS below the newest seen line, so lines the log
        # store delivers late (up to ~15s after their timestamp) are not skipped,
        # and deduplicate by event.id. Only ids inside the overlap window can come
        # back again, so trimming `seen` to it keeps the loop's memory bounded.
        print(f"\nTailing; stopping after {TAIL_LINES} new lines...")
        seen = {}  # event id -> timestamp, trimmed to the overlap window each round
        boundary = int(time.time() * 1000)  # newest timestamp seen so far
        printed = 0
        while printed < TAIL_LINES:
            for chunk in cclient.fetch_logs(
                DEPLOYMENT_ID, REVISION_NUMBER, start_time=max(boundary - OVERLAP_MS, 0), newest_first=False
            ):
                for event in chunk:
                    if event.id in seen:
                        continue
                    seen[event.id] = event.timestamp
                    boundary = max(boundary, event.timestamp)
                    print(format_event(event))
                    printed += 1
            cutoff = boundary - OVERLAP_MS
            seen = {event_id: ts for event_id, ts in seen.items() if ts >= cutoff}
            time.sleep(POLL_SECONDS)

        # A single pod (discover names with get_deployment_pods; terminated pods
        # still within log retention are included), with caller-sized chunks:
        #   pods = cclient.get_deployment_pods(DEPLOYMENT_ID, REVISION_NUMBER)
        #   for chunk in cclient.fetch_logs(
        #       DEPLOYMENT_ID, REVISION_NUMBER, pod=pods[0], start_time=t1_ms, end_time=t2_ms, chunk_size=500
        #   ):
        #       ...


if __name__ == "__main__":
    main()
