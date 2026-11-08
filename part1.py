"""Part 1 command line: run the rule-based graph (no LLM) and print the trace.

    python part1.py "I cannot connect to Wi-Fi."
    python part1.py "My Wi-Fi password does not work." --variant with_clarification
    python part1.py --draw            # print both graphs as Mermaid diagrams
"""

import argparse

from helpdesk.part1_graph import build_part1_graph, run_with_trace


def format_part1_trace(message, variant):
    result = run_with_trace(message, with_clarification=(variant == "with_clarification"))
    lines = [f"graph_variant={variant}", f"USER: {message}",
             "Route: START -> " + " -> ".join(result["route"]) + " -> END", "State after each node:"]
    for number, step in enumerate(result["steps"], 1):
        lines.append(f"  {number}. {step['node']}  update={step['update']}")
        lines.append(f"       state = {step['state_after']}")
    lines.append(f"HELPDESK: {result['final_state']['reply']}")
    return "\n".join(lines), result


def main():
    parser = argparse.ArgumentParser(description="Part 1 helpdesk graph without an LLM")
    parser.add_argument("message", nargs="?")
    parser.add_argument("--variant", choices=["original", "with_clarification"], default="original")
    parser.add_argument("--draw", action="store_true", help="print the graphs as Mermaid text")
    args = parser.parse_args()

    if args.draw:
        for variant in (False, True):
            print("with_clarification" if variant else "original")
            print(build_part1_graph(variant).get_graph().draw_mermaid())
        return
    if not args.message:
        parser.error("give a message or --draw")
    text, _ = format_part1_trace(args.message, args.variant)
    print(text)


if __name__ == "__main__":
    main()
