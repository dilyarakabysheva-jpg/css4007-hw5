"""Parts 2-3 command line: chat with the helpdesk (model + tools + memory).

Interactive:
    python chat.py --user demo-user-a --thread t1 --profile normal
One or more messages, then exit:
    python chat.py --user demo-user-a --thread t1 --profile normal --message "I cannot connect to Wi-Fi."
Offline with the scripted FAKE model (no API key):
    python chat.py --user demo-user-a --thread t1 --fake-model
"""

import argparse

from helpdesk.data import load_service_status
from helpdesk.session import HelpdeskSession, format_trace, make_model


def main():
    profiles = list(load_service_status()["profiles"])
    parser = argparse.ArgumentParser(description="University IT helpdesk (practice, fictional data)")
    parser.add_argument("--user", required=True, help="invented user id, e.g. demo-user-a")
    parser.add_argument("--thread", required=True, help="conversation id, e.g. t1")
    parser.add_argument("--profile", default="normal", choices=profiles, help="service status snapshot")
    parser.add_argument("--message", action="append", help="message to send (can be repeated)")
    parser.add_argument("--fake-model", action="store_true", help="use the scripted fake model (not an LLM)")
    parser.add_argument("--quiet", action="store_true", help="print only the replies, not the trace")
    args = parser.parse_args()

    model, model_label = make_model(fake=args.fake_model)
    session = HelpdeskSession(model)
    print(f"Model: {model_label}")
    print(f"user={args.user} thread={args.thread} status_profile={args.profile}")

    def send(text):
        trace = session.send(args.user, args.thread, args.profile, text)
        print(("HELPDESK: " + trace["reply"]) if args.quiet else format_trace(trace))
        print()

    try:
        if args.message:
            for text in args.message:
                send(text)
        else:
            print("Type a message (or 'quit' to exit).")
            while True:
                text = input("YOU: ").strip()
                if text.lower() in ("quit", "exit"):
                    break
                if text:
                    send(text)
    finally:
        session.close()


if __name__ == "__main__":
    main()
