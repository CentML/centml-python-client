import itertools
from datetime import datetime, timezone

from centml.sdk.api import get_centml_client

# --- Configuration ---
DEPLOYMENT_ID = 1234  # Replace with your deployment ID
REVISION_NUMBER = 10
FOLLOW_LINES = 20  # How many tailed lines to print before stopping the follow


def format_event(event) -> str:
    ts = datetime.fromtimestamp(event.timestamp / 1000, tz=timezone.utc).isoformat()
    return f"[{ts}] {event.pod} {event.message}"


def main():
    with get_centml_client() as cclient:
        # Stream the revision's full history, all pods merged chronologically.
        # The iterator is lazy: pages are fetched as you consume it, and its held
        # state stays bounded no matter how many lines stream by.
        print(f"Logs for deployment {DEPLOYMENT_ID} revision {REVISION_NUMBER}:\n")
        count = 0
        for event in cclient.iter_deployment_logs(DEPLOYMENT_ID, REVISION_NUMBER):
            print(format_event(event))
            count += 1
        print(f"\nCaught up after {count} lines.")

        # follow=True keeps tailing instead of returning: it re-polls caught-up pods
        # every poll_interval seconds and picks up new pods of the revision as they
        # first log. Stop by breaking out (or just abandon the iterator).
        print(f"\nFollowing; stopping after {FOLLOW_LINES} new lines...")
        stream = cclient.iter_deployment_logs(DEPLOYMENT_ID, REVISION_NUMBER, follow=True)
        for event in itertools.islice(stream, FOLLOW_LINES):
            print(format_event(event))

        # A single pod (discover names with get_deployment_pods) or a bounded start:
        #   pods = cclient.get_deployment_pods(DEPLOYMENT_ID, REVISION_NUMBER)
        #   for event in cclient.iter_deployment_logs(
        #       DEPLOYMENT_ID, REVISION_NUMBER, pod=pods[0], start_time=t1_ms
        #   ):
        #       ...
        # A specific time window as a list (all pods merged, oldest first):
        #   window = cclient.get_deployment_logs_range(
        #       DEPLOYMENT_ID, REVISION_NUMBER, start_time=t1_ms, end_time=t2_ms
        #   )
        # Manual paging, anchored on events you already hold — useful when you
        # manage storage yourself (deployment_log_session wraps this statefully):
        #   page  = cclient.get_deployment_logs(DEPLOYMENT_ID, REVISION_NUMBER, pod=pods[0])  # tail
        #   older = cclient.get_deployment_logs(..., pod=pods[0], before=page)  # empty return = beginning
        #   newer = cclient.get_deployment_logs(..., pod=pods[0], after=page)   # empty return = nothing new


if __name__ == "__main__":
    main()
