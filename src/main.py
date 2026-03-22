import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

from agents import node1_topic_agent, node2_rag_retriever, node3_writer_agent


def ask(prompt: str, default: str) -> str:
    """Show prompt, return user input or default if Enter pressed."""
    user_input = input(prompt).strip()
    return user_input if user_input else default


def human_checkpoint_1(state: dict) -> dict:
    """
    Human Checkpoint 1  review Node 1 suggestion.
    Human can approve the topic or reject and pick a different category.
    If rejected, Node 1 re-runs with the preferred category as a constraint.
    """
    while True:
        print("\nCheckpoint 1 - Topic Review")
        print(f"  Topic    : {state['suggested_topic']}")
        print(f"  Category : {state['suggested_category']}")
        print(f"  Keywords : {', '.join(state['suggested_keywords'])}")

        choice = ask("\n  Approve? (Enter = Yes  |  n = No): ", "j")

        if choice.lower() in ["j", "ja", "y", "yes", ""]:
            print("  Approved.")
            return {
                **state,
                "approved_topic":    state["suggested_topic"],
                "approved_category": state["suggested_category"],
                "approved_keywords": state["suggested_keywords"],
            }
        else:
            print("\n  Rejected. Please enter preferred category.\n")
            category_input = ask(
                "  Category (Strategie / Training / Technologie): ",
                state["suggested_category"],
            )
            state["user_preferred_category"] = category_input
            state["user_preferred_topic"]    = ""
            print("\n  Re-running Node 1 with new category...\n")
            state = {**state, **node1_topic_agent(state)}


def human_checkpoint_2(state: dict) -> dict:
    """
    Human Checkpoint 2  select content type and writing tone.
    Shows a preview of retrieved chunks before selection.
    """
    print("\nCheckpoint 2 - Content Type & Tone")

    # show preview of retrieved content
    chunks = state.get("retrieved_chunks", [])
    if chunks:
        preview = chunks[0].get("text", "")[:400]
        print(f"\n  Content preview:\n  {preview}...\n")

    print("  Content type:")
    print("    1. How-To       (step-by-step guide)")
    print("    2. Framework    (phases, pillars, models)")
    print("    3. Story        (personal or customer story)")
    print("    4. Facts & Data (numbers, studies, data)")
    print("    5. Case Study   (before/after, real example)")

    inhaltstyp_map = {
        "1": "How-To",
        "2": "Framework",
        "3": "Story",
        "4": "Facts & Data",
        "5": "Case Study",
    }

    inhaltstyp = inhaltstyp_map.get(ask("\n  Type [1-5]: ", "1"), "How-To")

    print("\n  Writing tone:")
    print("    1. Professionell")
    print("    2. Edukativ")
    print("    3. Praktisch")

    tone_map = {
        "1": "Professionell",
        "2": "Edukativ",
        "3": "Praktisch",
    }

    tone = tone_map.get(ask("\n  Tone [1-3]: ", "1"), "Professionell")

    print(f"\n  Type: {inhaltstyp} | Tone: {tone}")

    return {
        **state,
        "inhaltstyp": inhaltstyp,
        "tone":       tone,
    }


def save_output(state: dict) -> None:
    """Saves the final article as a .txt file in the output directory."""
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    title      = state.get("article_title", "artikel")
    safe_title = "".join(c for c in title if c.isalnum() or c in " -_")[:50].strip()
    output_file = output_dir / f"{safe_title}.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"TITEL: {state.get('article_title', '')}\n")
        f.write(f"KATEGORIE: {state.get('category_tag', '')}\n")
        f.write(f"KEYWORDS: {', '.join(state.get('final_keywords', []))}\n\n")
        f.write("GLIEDERUNG:\n")
        for item in state.get("article_outline", []):
            f.write(f"  {item}\n")
        f.write("\nARTIKEL:\n\n")
        f.write(state.get("article_draft", ""))

    print(f"\n  Article saved: {output_file}")


def main():
    state = {}

    # get topic direction from user
    print("\nSESTdigital SEO Content Agent")
    user_query = ask(
        "\n  What should the article be about?\n  (e.g. 'KI Training für Mitarbeiter' or 'KI Automatisierung')\n\n  Your topic: ",
        "KI und Digitalisierung im Unternehmen",
    )
    state["user_query"] = user_query

    # run pipeline
    state = {**state, **node1_topic_agent(state)}
    state = human_checkpoint_1(state)
    state = {**state, **node2_rag_retriever(state)}
    state = human_checkpoint_2(state)
    state = {**state, **node3_writer_agent(state)}

    # print final output
    print("\nFinal Output")
    print(f"  Title    : {state.get('article_title')}")
    print(f"  Category : {state.get('category_tag')}")
    print(f"  Keywords : {state.get('final_keywords')}")
    print("\n  Outline:")
    for item in state.get("article_outline", []):
        print(f"    - {item}")
    print("\n  Article Draft:\n")
    print(state.get("article_draft", ""))

    save_output(state)


if __name__ == "__main__":
    main()