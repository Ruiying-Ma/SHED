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
    headers = _parse_md(raw_md_str_list)
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
    src_log_path = "/home/ruiying/SHTRAG/eval/header_identification_log.csv"
    with open(src_log_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    
    tab_str = lines[0] + ",predicted_raw\n"
    # llm-txt
    m_file_llmtxt = dict()
    for dataset in ['civic', 'contract', 'qasper', 'finance']:
        headers_list = _load_llm_txt_headers(dataset)
        pdf_names = sorted(os.listdir(os.path.join(config.DATA_ROOT_FOLDER, dataset, 'pdf')))
        assert len(headers_list) == len(pdf_names), f"Mismatch in number of PDFs and headers for dataset {dataset}"
        for idx, pdf_name in enumerate(pdf_names):
            filename = pdf_name.replace(".pdf", "")
            file = f"{dataset},{filename}"
            m_file_llmtxt[file] = len(headers_list[idx])
    
    # llm-vision
    m_file_llmvision = dict()
    for dataset in ['civic', 'contract', 'qasper', 'finance']:
        pdf_names = sorted(os.listdir(os.path.join(config.DATA_ROOT_FOLDER, dataset, 'pdf')))
        for pdf_name in pdf_names:
            filename = pdf_name.replace(".pdf", "")
            headers = _load_llm_vision_headers(dataset, filename)
            file = f"{dataset},{filename}"
            m_file_llmvision[file] = len(headers)

    for line in lines[1:]:
        parts = line.split(",")
        sht_type = parts[0]
        dataset = parts[1]
        filename = parts[2]
        file = f"{dataset},{filename}"
        predicted_cleaned = int(parts[5])
        if sht_type == "llm_txt":
            predicted_raw = m_file_llmtxt[file]
            assert predicted_raw >= predicted_cleaned, f"predicted_raw < predicted_cleaned for {file} in llm_txt: {predicted_raw} vs {predicted_cleaned}"
        elif sht_type == "llm_vision":
            predicted_raw = m_file_llmvision[file]
            assert predicted_raw >= predicted_cleaned, f"predicted_raw < predicted_cleaned for {file} in llm_vision: {predicted_raw} vs {predicted_cleaned}"
        else:
            predicted_raw = ""
        tab_str += line + f",{predicted_raw}\n"
    
    dst_log_path = "/home/ruiying/SHTRAG/eval/header_identification_raw_llm_log.csv"
    with open(dst_log_path, "w") as f:
        f.write(tab_str)


