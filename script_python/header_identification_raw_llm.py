import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import config
import logging
import logging_config

def _parse_md(raw_md_str_list: list):
    md_lines = []
    for raw_md_str in raw_md_str_list:
        if raw_md_str.strip().startswith("```markdown") and raw_md_str.strip().endswith("```"):
            md_str = raw_md_str.strip()[len("```markdown"): -len("```")]
        else:
            md_str = raw_md_str
        md_lines += md_str.splitlines()
    
    header_list = []
    cur_line = ""
    for line in md_lines:
        if line.startswith("#"):
            if cur_line.startswith("#"):
                header_list.append(cur_line)
            cur_line = line
        else:
            cur_line += line
    if cur_line.startswith("#"):
        header_list.append(cur_line)

    return header_list


def _load_sht_headers(dataset, filename, sht_type):
    sht_path = os.path.join(
        config.DATA_ROOT_FOLDER,
        dataset,
        sht_type if sht_type != "shed" else "",
        f"sbert.gpt-4o-mini.c100.s100",
        "sht" if sht_type not in ['wide', 'deep', 'llm_txt', 'llm_vision'] else "sht_skeleton",
        filename + ".json"
    )
    assert os.path.exists(sht_path), sht_path
    # logging.info(f"Loading SHT for {filename} ({sht_type}) from {sht_path}...")
    
    with open(sht_path, 'r') as file:
        raw_sht_nodes = json.load(file)['nodes']

    header_list = [
        node['heading']
        for node in raw_sht_nodes
        if node['is_dummy'] == False and node['type'] == 'head'
    ]

    return header_list

def _load_llm_vision_headers(dataset, filename):
    sht_type = "llm_vision"
    llm_result_path = os.path.join(
        config.DATA_ROOT_FOLDER,
        dataset,
        sht_type,
        sht_type,
        filename + ".jsonl"
    )
    with open(llm_result_path, 'r') as file:
        llm_results = [json.loads(line) for line in file.readlines()]
    raw_md_str_list = [r['headers'] for r in llm_results]
    headers = [_parse_md([md_str]) for md_str in raw_md_str_list]
    return headers


def _load_llm_txt_headers(dataset):
    llm_txt_folder = os.path.join(config.DATA_ROOT_FOLDER, dataset, "llm_txt") 
    with open(os.path.join(llm_txt_folder, "llm_txt", "sht_parsed.jsonl"), 'r') as file:
        raw_raw_md_list = [json.loads(line) for line in file]

    raw_md_list = []
    pdf_id_set = set()
    for raw_raw_md_info in raw_raw_md_list:
        pdf_id = raw_raw_md_info['id'] if isinstance(raw_raw_md_info['id'], int) else raw_raw_md_info['id'][0]
        assert [r['id'] for r in raw_md_list] == list(range(len(raw_md_list))) == list(sorted(pdf_id_set))
        if pdf_id not in pdf_id_set:
            pdf_id_set.add(pdf_id)
            assert pdf_id == (raw_md_list[-1]['id'] + 1 if len(raw_md_list) > 0 else 0), f"{pdf_id} vs {raw_md_list[-1]['id'] if len(raw_md_list) > 0 else 'N/A'}"
            raw_md_list.append({
                "id": pdf_id,
                "headers": [raw_raw_md_info['headers']],
            })
        else:
            last_md = raw_md_list[-1]
            assert pdf_id == last_md['id'], f"{pdf_id} vs {last_md['id']}"
            last_md['headers'].append(raw_raw_md_info['headers'])

    assert [r['id'] for r in raw_md_list] == list(range(len(raw_md_list))) == list(sorted(pdf_id_set))
    
    headers_list = [_parse_md(r['headers']) for r in raw_md_list] # = sorted list for pdfs

    return headers_list



if __name__ == "__main__":
    dataset_list = ['civic', 'contract', 'qasper', 'finance']
    sht_type_list = ['llm_txt', 'llm_vision']

    for sht_type in sht_type_list:
        logging.info(f"Evaluating SHT type: {sht_type}...")
        m_dataset_recall = {}
        m_dataset_precision = {}
        m_dataset_f1 = {}
        for dataset in dataset_list:
            if sht_type == "llm_vision":
                recall_sum = 0.0
                precision_sum = 0.0
                f1_sum = 0.0
                pdf_filename_list = sorted(os.listdir(os.path.join(config.DATA_ROOT_FOLDER, dataset, "pdf")))
                for pdf_filenmame in pdf_filename_list:
                    logging.info(f"Evaluating {dataset}/{pdf_filenmame}...")
                    filename = pdf_filenmame.replace(".pdf", "")
                    header_list = _load_llm_vision_headers(dataset, filename, sht_type)
                    extracted_header_list = _load_sht_headers(dataset, filename, sht_type)
                    true_header_list = _load_sht_headers(dataset, filename, "intrinsic")
                    
                    recall = len(extracted_header_list) / len(true_header_list) if len(true_header_list) > 0 else 0.0
                    precision = len(extracted_header_list) / len(header_list) if len(header_list) > 0 else 0.0
                    
                    if recall == 0 or precision == 0:
                        f1 = 0.0
                    else:
                        f1 = 2 * (precision * recall) / (precision + recall)

                    recall_sum += recall
                    precision_sum += precision
                    f1_sum += f1
            else:
                recall_sum = 0.0
                precision_sum = 0.0
                f1_sum = 0.0
                headers_list = _load_llm_txt_headers(dataset)
                pdf_filename_list = sorted(os.listdir(os.path.join(config.DATA_ROOT_FOLDER, dataset, "pdf")))
                assert len(headers_list) == len(pdf_filename_list)
                for id, pdf_filenmame in enumerate(pdf_filename_list):
                    logging.info(f"Evaluating {dataset}/{pdf_filenmame}...")
                    filename = pdf_filenmame.replace(".pdf", "")
                    header_list = headers_list[id]
                    extracted_header_list = _load_sht_headers(dataset, filename, sht_type)
                    true_header_list = _load_sht_headers(dataset, filename, "intrinsic")
                    
                    recall = len(extracted_header_list) / len(true_header_list) if len(true_header_list) > 0 else 0.0
                    assert recall <= 1.0, f"Recall > 1.0 for {dataset} {filename}"
                    precision = len(extracted_header_list) / len(header_list) if len(header_list) > 0 else 0.0
                    assert precision <= 1.0, f"Precision > 1.0 for {dataset} {filename}"
                    
                    if recall == 0 or precision == 0:
                        f1 = 0.0
                    else:
                        f1 = 2 * (precision * recall) / (precision + recall)

                    recall_sum += recall
                    precision_sum += precision
                    f1_sum += f1


            
            m_dataset_recall[dataset] = recall_sum / len(pdf_filename_list)
            m_dataset_precision[dataset] = precision_sum / len(pdf_filename_list)
            m_dataset_f1[dataset] = f1_sum / len(pdf_filename_list)

        # print recall
        tab_str = f"{sht_type} "
        for dataset in dataset_list:
            tab_str += "& " + f"{m_dataset_recall[dataset]*100:.2f}\%"

        # print precision
        for dataset in dataset_list:
            tab_str += "& " + f"{m_dataset_precision[dataset]*100:.2f}\%"
        # print f1
        for dataset in dataset_list:
            tab_str += "& " + f"{m_dataset_f1[dataset]*100:.2f}\%"

        print(tab_str + "\\\n")
            
