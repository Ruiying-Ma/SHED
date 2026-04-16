import tiktoken
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
import eval.eval_finance

TOKENIZER = tiktoken.get_encoding("cl100k_base")
ROOT_FOLDER = "/home/ruiying/SHTRAG/data"

def queries_with_long_evidence(dataset):
    query_path = os.path.join(ROOT_FOLDER, dataset, "queries.json")
    with open(query_path, 'r') as file:
        queries = json.load(file)

    long_evidence_query_ids = []
    for query in queries:
        evidence_text = " ".join(query['context'])
        tokenized_evidence = TOKENIZER.encode(evidence_text)
        if len(tokenized_evidence) > 100:
            long_evidence_query_ids.append(query["id"])
    
    return long_evidence_query_ids

def queries_use_hierarchical_structures(dataset):
    query_path = os.path.join(ROOT_FOLDER, dataset, "queries_justification.json")
    with open(query_path, 'r') as file:
        queries = json.load(file)

    load_dotenv("/home/ruiying/SHTRAG/structured_rag/.env")

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_API_KEY"),
    )

    prompt_template = '''Given the following query, the correct answer, the evidence context used to answer the query, and an explanation of how to derive the correct answer from the evidence context, your task is to determine whether hierarchical structures in the evidence context are **important** for answering the query correctly. Hierarchical structures refer to the organization of information in a multi-level format, such as section headers or subsection headers that provide a clear hierarchy of ideas. It is not sufficient for hierarchical structures to be merely present in the evidence context; they must play a crucial role in deriving the correct answer, **without which answering the query would be significantly more difficult or impossible**.

**INSTRUCTIONS:**
1. Determine whether the evidence context contains hierarchical structures (e.g., section headers, subsection headers). If not, answer "No".
2. If hierarchical structures are present, assess their importance in deriving the correct answer based on the provided explanation and the evidence context. If the explanation indicates that understanding or utilizing these hierarchical structures is essential to arrive at the correct answer, then hierarchical structures are considered important, and you should answer "Yes". Otherwise, answer "No".

**QUERY**:
{query_text}

**CORRECT ANSWER**:
{true_answer}

**EVIDENCE CONTEXT**:
{evidence_text}

**EXPLANATION**:
{justification}

**ANSWER (Yes or No)**:
Do not provide any explanation, only answer with "Yes" or "No".'''

    for query in queries:
        evidence_text = "\n".join(query['context'])
        formatted_prompt = prompt_template.format(
            query_text=query['query'],
            true_answer=query['answer'][0],
            evidence_text=evidence_text,
            justification=query['justification']
        )
        
        try: 
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.0,
                max_tokens=10,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error processing query ID {query['id']}: {e}")
            answer = "LLMs fail to answer"
        
        logging_info = {
            "query_id": query["id"],
            "prompt": formatted_prompt,
            "response": answer
        }

        with open(f"hierarchical_structure_detection_with_justification_2_log_{dataset}.jsonl", 'a') as log_file:
            log_file.write(json.dumps(logging_info) + "\n")

        if "yes" in answer.lower():
            query['uses_hierarchical_structure'] = True
        else:
            query['uses_hierarchical_structure'] = False

    return [q['id'] for q in queries if q['uses_hierarchical_structure'] == True]

def queries_vanilla_fail():
    vanilla_ctx_config = ("vanilla", None, "sbert", None, None, None, 0.2)
    vanilla_acc_list, vanilla_answer_id_list = eval.eval_finance.finance_eval_answer_llm_list(vanilla_ctx_config), list(range(150))
    failed_query_ids = [vanilla_answer_id_list[i] for i, acc in enumerate(vanilla_acc_list) if acc <= 0]
    return failed_query_ids

def queries_shed_win():
    vanilla_config = ("vanilla", None, "sbert", None, None, None, 0.2)
    raptor_config = ("raptor", None, "sbert", None, None, None, 0.2)
    graphrag_config = ("graphrag", None, None, None, None, None, None)
    hipporag_config = ("hipporag", None, "sbert", None, None, None, 0.2)
    shed_config = ("sht", "intrinsic", "sbert", True, True, True, 0.2)

    vanilla_acc_list, vanilla_answer_id_list = eval.eval_finance.finance_eval_answer_llm_list(vanilla_config), list(range(150))
    raptor_acc_list, raptor_answer_id_list = eval.eval_finance.finance_eval_answer_llm_list(raptor_config), list(range(150))
    # graphrag_acc_list, graphrag_answer_id_list = eval.eval_finance.finance_eval_answer_llm_list(graphrag_config), list(range(150))
    # hipporag_acc_list, hipporag_answer_id_list = eval.eval_finance.finance_eval_answer_llm_list(hipporag_config), list(range(150))
    shed_acc_list, shed_answer_id_list = eval.eval_finance.finance_eval_answer_llm_list(shed_config), list(range(150))
    shed_win_query_ids = []
    for i in range(150):
        if shed_acc_list[i] >= 2 and vanilla_acc_list[i] <= 1:
            shed_win_query_ids.append(i)
    return shed_win_query_ids


if __name__ == "__main__":
    
    dataset = "finance"
    shed_win_queries = queries_shed_win()
    print(f"SHTRAG wins over all other methods on queries: {shed_win_queries}")

    # true_win_queries = [0, 2, 3, 7, 8, 9, 11, 12, 13, 15, 19, 20, 21, 22, 24, 25, 26, 27, 31, 34, 35, 37, 38, 39, 42, 43, 44, 45, 48, 49, 51, 52, 53, 54, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 68, 70, 71, 72, 74, 76, 77, 78, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 92, 94, 95, 96, 97, 101, 103, 104, 105, 108, 109, 110, 113, 115, 117, 118, 121, 123, 124, 125, 126, 128, 130, 133, 134, 135, 136, 137, 139, 140, 142, 143, 144, 146, 147, 148]

    # shed_win_queries = [0, 2, 3, 6, 7, 8, 9, 12, 13, 15, 19, 20, 21, 24, 26, 29, 30, 34, 35, 37, 38, 39, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 61, 63, 64, 65, 66, 67, 68, 70, 71, 72, 74, 75, 76, 77, 78, 80, 81, 82, 83, 86, 87, 88, 89, 90, 94, 95, 96, 97, 98, 100, 101, 103, 104, 105, 107, 109, 110, 113, 115, 117, 118, 120, 121, 123, 124, 125, 126, 127, 128, 130, 131, 132, 133, 134, 135, 137, 139, 140, 142, 143, 144, 146, 147, 148]

    # wide_bad_queries = [2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 16, 17, 18, 19, 23, 26, 27, 28, 29, 31, 36, 37, 40, 42, 45, 47, 48, 50, 53, 54, 55, 56, 60, 61, 62, 64, 69, 70, 71, 73, 75, 76, 77, 78, 79, 80, 81, 82, 85, 86, 87, 88, 91, 92, 93, 94, 95, 96, 98, 99, 100, 102, 104, 105, 107, 111, 112, 114, 116, 118, 121, 123, 125, 127, 129, 130, 131, 133, 135, 136, 140, 141, 143, 144, 145, 146, 147, 148]

    # vanilla_bad_queries = [2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 17, 18, 26, 28, 29, 31, 37, 38, 39, 40, 45, 47, 48, 50, 53, 54, 60, 61, 62, 64, 65, 66, 67, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 87, 88, 89, 93, 94, 95, 96, 98, 99, 100, 102, 103, 104, 105, 107, 111, 114, 116, 118, 121, 122, 123, 124, 127, 129, 130, 131, 133, 135, 137, 140, 141, 143, 144, 145, 146, 147, 148, 149] <= 1
    # vanilla_bad_queries = [2, 5, 8, 9, 10, 13, 14, 17, 18, 26, 29, 31, 38, 40, 45, 47, 50, 53, 54, 60, 61, 67, 70, 71, 72, 75, 76, 77, 82, 84, 85, 87, 88, 94, 95, 96, 99, 102, 103, 104, 105, 107, 111, 114, 116, 118, 121, 123, 124, 127, 133, 135, 140, 141, 144, 146, 147, 149] <= 0

    # print(list(set(vanilla_bad_queries).difference(set(wide_bad_queries))))

    # failed_queries = queries_vanilla_fail(dataset)
    # print(f"Vanilla failed queries: {failed_queries}")



    # hierarchical_structure_query_ids = queries_use_hierarchical_structures(dataset)
    # print(f"Queries that use hierarchical structures: {hierarchical_structure_query_ids}")

    # long_queries = queries_with_long_evidence("finance")
    # hier_queries = [0, 1, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 23, 28, 30, 31, 32, 35, 36, 38, 39, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 97, 99, 100, 101, 102, 103, 104, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126, 127, 131, 132, 134, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]

    # print(list(set(long_queries).intersection(set(hier_queries))))

    # hier_queries = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 28, 30, 31, 32, 35, 36, 42, 43, 45, 46, 47, 48, 49, 51, 52, 53, 54, 55, 56, 58, 59, 60, 62, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 97, 99, 100, 101, 102, 103, 104, 106, 108, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 127, 132, 136, 138, 139, 140, 142, 143, 144, 145, 146, 147, 148, 149]

    # print(len(set(long_queries).intersection(set(hier_queries))))


    # opt_queries_12 = [0, 1, 3, 5, 7, 12, 19, 20, 21, 24, 26, 34, 39, 42, 43, 48, 49, 51, 52, 56, 59, 61, 62, 64, 65, 68, 74, 78, 80, 81, 82, 83, 86, 89, 94, 95, 97, 101, 105, 106, 109, 110, 112, 113, 115, 118, 123, 125, 126, 128, 131, 133, 134, 135, 139, 142, 143, 147, 148]

    # opt_queries_1 = [0, 1, 2, 3, 5, 7, 8, 12, 13, 15, 19, 20, 21, 24, 28, 32, 34, 35, 37, 39, 42, 43, 44, 45, 48, 49, 51, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 70, 74, 76, 77, 78, 80, 81, 82, 83, 85, 86, 88, 90, 94, 95, 97, 101, 105, 106, 109, 110, 111, 112, 117, 123, 125, 126, 128, 129, 130, 131, 133, 134, 135, 139, 141, 142, 143, 145, 146, 147, 148]

    # long_hier_queries = list(set(long_queries).intersection(set(hier_queries)))

    

