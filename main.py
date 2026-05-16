"""
main.py - CLI entry point for the RAG agent
Usage:
    python main.py # interactive mode
    python main.py "your question" # single query mode
"""

import sys
from agent import RAGAgent

def print_log(log):
    """Print the escalation trace table to terminal."""
    print("\n── Escalation trace ────────────────────────────────────────")
    header = f"{'Lvl':<4} {'Strategy':<36} {'Conf':>5} {'Tokens':>7} {'Time':>8}  {'✓'}"
    print(header)
    print("─" * len(header))
    for row in log.summary_table():
        print(
            f"{row['Level']:<4} {row['Strategy']:<36} {row['Confidence']:>5} "
            f"{row['Tokens']:>7} {row['Time (ms)']:>7}ms  {row['Resolved']}"
        )
    print(f"\nTotal tokens: {log.total_tokens}  |  Total time: {log.total_time_ms:.0f}ms")
    print("────────────────────────────────────────────────────────────\n")

def run_interactive(agent: RAGAgent):
    print("\n RAG Agent — type 'quit' to exit\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        print("\nSearching...\n")
        log = agent.ask(query)
        print_log(log)
        print(f"Answer:\n{log.final_answer}\n")

def main():
    agent = RAGAgent()
    n_chunks = agent.load_docs("docs")
    print(f"Indexed {n_chunks} chunks from docs/")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        log = agent.ask(query)
        print_log(log)
        print(f"Answer:\n{log.final_answer}")
    else:
        run_interactive(agent)

if __name__ == "__main__":
    main()