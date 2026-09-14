"""
query_cli.py
------------
A tiny interactive command-line interface for the IR system, useful
for taking screenshots of live queries for the assignment report.

Run with:  python query_cli.py
Type 'exit' to quit.
"""

import os

from ir_system import IRSystem

DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "documents")


def main():
    system = IRSystem(use_stemming=True)
    system.add_documents_from_folder(DOCS_FOLDER)
    system.build_index()

    stats = system.stats()
    print(f"Loaded {stats['num_documents']} documents, "
          f"{stats['vocabulary_size']} vocabulary terms.")
    print("Enter a Boolean query (AND / OR / NOT / parentheses), "
          "or 'exit' to quit.")
    print("Example: python AND (snake OR language)\n")

    while True:
        try:
            query = input("query> ").strip()
        except EOFError:
            break
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue
        try:
            results = system.search(query)
        except Exception as exc:
            print(f"  Error: {exc}")
            continue
        if results:
            print(f"  {len(results)} match(es): {results}")
        else:
            print("  No matches.")


if __name__ == "__main__":
    main()
