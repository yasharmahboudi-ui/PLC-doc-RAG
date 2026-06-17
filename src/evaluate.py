import os
import sys
from query import query_rag

EVAL_QUESTIONS = [
    "What does error code 8090 mean?",
    "How do I configure a digital output module?",
    "What do the LEDs on the CPU indicate?",
    "How do you connect the power supply to the S7-1200 CPU?",
    "What is the maximum number of signal modules that can be added to an S7-1200 CPU?"
]

def run_evaluation():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_path = os.path.join(base_dir, "evaluation_results.md")
    
    print(f"Starting evaluation of {len(EVAL_QUESTIONS)} questions...")
    
    results = []
    for idx, q in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n[{idx}/{len(EVAL_QUESTIONS)}] Querying: '{q}'")
        res = query_rag(q)
        
        if res:
            answer = res["answer"]
            sources = []
            for doc in res["context"]:
                page = doc.metadata.get('page', 0)
                sources.append(f"Page {page + 1 if isinstance(page, int) else page}")
            
            unique_sources = sorted(list(set(sources)))
            source_str = ", ".join(unique_sources)
            results.append((q, answer, source_str))
        else:
            results.append((q, "Error running query.", "N/A"))

    # Write evaluation table to Markdown
    print(f"\nSaving evaluation results to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Results\n\n")
        f.write("This document logs the evaluation queries, grounded answers, and source citations retrieved from the Siemens S7-1200 manual.\n\n")
        f.write("| # | Question | Answer | Source Pages |\n")
        f.write("|---|----------|--------|--------------|\n")
        for idx, (q, ans, src) in enumerate(results, 1):
            # Clean up newlines for markdown table
            clean_ans = ans.replace("\n", " ").replace("|", "\\|")
            f.write(f"| {idx} | **{q}** | {clean_ans} | {src} |\n")
            
    print("\nEvaluation completed! Summary of results:")
    print("="*60)
    for idx, (q, ans, src) in enumerate(results, 1):
        print(f"\nQuestion {idx}: {q}")
        print(f"Sources: {src}")
        print(f"Answer:\n{ans}")
        print("-"*60)
    print("="*60)

if __name__ == "__main__":
    run_evaluation()
