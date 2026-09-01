import time
from datetime import datetime, timezone

from centml.sdk.api import get_centml_client

# --- Configuration ---
DEPLOYMENT_ID = 1234  # Replace with your deployment ID
REVISION_NUMBER = 10
TAIL_SECONDS = 30  # How long to keep polling for new lines after reading history
TAIL_LINES = 20  # How much history to print before tailing


def format_event(event) -> str:
    ts = datetime.fromtimestamp(event.timestamp / 1000, tz=timezone.utc).isoformat()
    return f"[{ts}] {event.message}"


def main():
    with get_centml_client() as cclient:
        # Logs are read per pod: discover the pods that have logged for this revision
        # (terminated pods within log retention are included).
        pods = cclient.get_deployment_pods(DEPLOYMENT_ID, REVISION_NUMBER)
        if not pods:
            print("No pods have logged for this revision yet.")
            return

        pod = pods[0]
        print(f"Reading logs for deployment {DEPLOYMENT_ID} revision {REVISION_NUMBER}, pod {pod}\n")

        # The session tracks what it has fetched and anchors every request itself.
        session = cclient.deployment_log_session(DEPLOYMENT_ID, REVISION_NUMBER, pod)

        # Read the full history: newest page first, then page back to the beginning.
        while session.fetch_older():
            pass
        events = session.events
        print(f"Found {len(events)} log entries; showing the last {TAIL_LINES}:\n")
        for event in events[-TAIL_LINES:]:
            print(format_event(event))

        # Keep tailing: each call returns only the lines the session does not hold yet.
        print(f"\nPolling for new lines for {TAIL_SECONDS}s...")
        deadline = time.monotonic() + TAIL_SECONDS
        while time.monotonic() < deadline:
            for event in session.fetch_newer():
                print(format_event(event))
            time.sleep(2)

        # The same paging is available statelessly via get_deployment_logs, anchored
        # on events you already hold — useful when you manage storage yourself:
        #   page  = cclient.get_deployment_logs(DEPLOYMENT_ID, REVISION_NUMBER, pod=pod)  # tail
        #   older = cclient.get_deployment_logs(..., pod=pod, before=page)  # empty return = beginning
        #   newer = cclient.get_deployment_logs(..., pod=pod, after=page)   # empty return = nothing new
        # A specific time window (all pods merged, oldest first, pod on each event):
        #   window = cclient.get_deployment_logs_range(
        #       DEPLOYMENT_ID, REVISION_NUMBER, start_time=t1_ms, end_time=t2_ms
        #   )


if __name__ == "__main__":
    main()
