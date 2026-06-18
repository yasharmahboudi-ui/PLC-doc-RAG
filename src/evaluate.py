import os
import sys
from query import query_rag
from evaluate_ragas import run_ragas_evaluation

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
    
    raw_results = []
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
            contexts = [doc.page_content for doc in res["context"]]
            
            raw_results.append({
                "question": q,
                "answer": answer,
                "source_str": source_str,
                "contexts": contexts
            })
        else:
            raw_results.append({
                "question": q,
                "answer": "Error running query.",
                "source_str": "N/A",
                "contexts": []
            })

    # Run RAGAS evaluation on the results
    questions = [r["question"] for r in raw_results]
    answers = [r["answer"] for r in raw_results]
    contexts = [r["contexts"] for r in raw_results]
    
    print("\nRunning RAGAS evaluation on the results...")
    ragas_scores = run_ragas_evaluation(questions, answers, contexts)

    # Write evaluation table to Markdown
    print(f"\nSaving evaluation results to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Results\n\n")
        f.write("This document logs the evaluation queries, grounded answers, source citations, and RAGAS alignment metrics retrieved from the Siemens S7-1200 manual.\n\n")
        f.write("| # | Question | Answer | Source Pages | Faithfulness | Answer Relevancy |\n")
        f.write("|---|----------|--------|--------------|--------------|------------------|\n")
        for idx, r in enumerate(raw_results):
            # Clean up newlines for markdown table
            clean_ans = r["answer"].replace("\n", " ").replace("|", "\\|")
            
            # Extract scores
            score_dict = ragas_scores[idx] if idx < len(ragas_scores) else {"faithfulness": "N/A", "answer_relevancy": "N/A"}
            f_score = f"{score_dict['faithfulness']:.2f}" if isinstance(score_dict['faithfulness'], float) else score_dict['faithfulness']
            ar_score = f"{score_dict['answer_relevancy']:.2f}" if isinstance(score_dict['answer_relevancy'], float) else score_dict['answer_relevancy']
            
            f.write(f"| {idx+1} | **{r['question']}** | {clean_ans} | {r['source_str']} | {f_score} | {ar_score} |\n")
            
    print("\nEvaluation completed! Summary of results:")
    print("="*60)
    for idx, r in enumerate(raw_results):
        score_dict = ragas_scores[idx] if idx < len(ragas_scores) else {"faithfulness": "N/A", "answer_relevancy": "N/A"}
        f_score = f"{score_dict['faithfulness']:.2f}" if isinstance(score_dict['faithfulness'], float) else score_dict['faithfulness']
        ar_score = f"{score_dict['answer_relevancy']:.2f}" if isinstance(score_dict['answer_relevancy'], float) else score_dict['answer_relevancy']
        
        print(f"\nQuestion {idx+1}: {r['question']}")
        print(f"Sources: {r['source_str']}")
        print(f"RAGAS Faithfulness: {f_score} | Answer Relevancy: {ar_score}")
        print(f"Answer:\n{r['answer']}")
        print("-"*60)
    print("="*60)

if __name__ == "__main__":
    run_evaluation()
