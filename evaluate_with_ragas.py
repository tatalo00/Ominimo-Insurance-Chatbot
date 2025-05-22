import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall
)
from dotenv import load_dotenv
from qa_chain import get_qa_chain, retriever

# Load environment variables (from .env file)
load_dotenv()

# Set your API key if needed
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key is None:
    raise ValueError("OPENAI_API_KEY not set in environment variables")

# Load sample eval dataset from JSON file
with open("./Evaluation/ragas_eval_dataset.json", "r", encoding="utf-8") as f:
    examples = json.load(f)

questions = [
    "Meddig tárolják a biztosítási szerződésekhez kapcsolódó adatokat?",
    "Mi az a kötelező gépjármű-felelősségbiztosítás?",
    "Milyen esetekre nem terjed ki a biztosítás?",
    "Milyen dokumentumokat kérhet be a biztosító egy kárigényhez?",
    "Hogyan zajlik az adatkezelés a biztosítónál?",
    "Mi történik, ha az ügyfél nem járul hozzá az adatkezeléshez?",
    "Mikor kezdődik a biztosítási fedezet?",
    "Milyen jogai vannak az ügyfélnek az adatkezeléssel kapcsolatban?",
    "Kik férhetnek hozzá a biztosítási titoknak minősülő adatokhoz?",
    "Milyen célból kér a biztosító egészségügyi adatokat?",
    "Mire vonatkozik a biztosítási titok fogalma?",
    "Mi történik az adatokkal a szerződés megszűnése után?",
    "Milyen esetekben kérhet a biztosító kiegészítő adatokat?",
    "Mi a teendő, ha az ügyfél vitatja a biztosító döntését?"
]

ground_truths = [
    ["8 év szerződés megszűnését követő naptári év"],
    ["jármű üzemeltetése kár fedezete"],
    ["háborús cselekmény karbantartás verseny"],
    ["jegyzőkönyv orvosi dokumentum bankszámlaszám"],
    ["GDPR célhoz kötött meghatározott ideig"],
    ["visszautasítás szolgáltatás ellehetetlenülése"],
    ["szerződésben meghatározott időpont"],
    ["hozzáférés törlés adathordozhatóság"],
    ["biztosító hatóság érintett engedélye"],
    ["kockázat-elbírálás kárigény elbírálása"],
    ["személyi vagyoni helyzet biztosítási szerződés"],
    ["8 év jogi eljárás"],
    ["egészségkárosodás vitás ügy"],
    ["panasz jogorvoslat"]
]

questions_en = [
    "How long are insurance contract-related data stored?",
    "What is mandatory vehicle liability insurance?",
    "What cases are not covered by the insurance?",
    "What documents can the insurer request for a claim?",
    "How does data processing work at the insurer?",
    "What happens if the customer does not consent to data processing?",
    "When does insurance coverage begin?",
    "What rights does the customer have regarding data processing?",
    "Who can access data considered as insurance secrets?",
    "For what purpose does the insurer request medical data?",
    "What does the term insurance secret refer to?",
    "What happens to the data after the contract ends?",
    "When can the insurer request additional data?",
    "What can the customer do if they dispute the insurer’s decision?"
]

ground_truths_en = [
    ["8 years from the first day of the year after contract termination"],
    ["vehicle operation damage coverage"],
    ["war actions maintenance race"],
    ["police report medical document bank account number"],
    ["GDPR purpose-bound defined period"],
    ["rejection service impossibility"],
    ["date specified in the contract"],
    ["access deletion data portability"],
    ["insurer authority subject authorization"],
    ["risk assessment claim evaluation"],
    ["personal financial situation insurance contract"],
    ["8 years legal proceedings"],
    ["health impairment disputed case"],
    ["complaint legal remedy"]
]


answers = []
contexts = []


# Inference
for query in questions_en:
    result = get_qa_chain().invoke({"query": query})
    answers.append(result["result"])
    docs = retriever.invoke(query)
    contexts.append([doc.page_content for doc in docs])

# Build dataset
examples = []
for q, a, c, gt in zip(questions_en, answers, contexts, ground_truths_en):
    examples.append({
        "question": q,
        "answer": a,
        "contexts": c,
        "ground_truth": " ".join(gt) if isinstance(gt, list) else gt
    })

# Convert dict to dataset
dataset = Dataset.from_list(examples)

# Run evaluation
results = evaluate(
    dataset,
    metrics=[
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall
    ]
)

df = results.to_pandas()

df.to_csv("ragas_scores_en.csv", index=False, encoding="utf-8")