import os

def parse_csv_line(line: str):
    parts = line.strip().split(",")
    sht_type = parts[0]
    dataset = parts[1]
    filename = parts[2]
    tp = int(parts[3])
    true_headers = int(parts[4])
    predicted_cleaned = int(parts[5])
    if parts[6] == '':
        predicted_raw = predicted_cleaned
    else:
        predicted_raw = int(parts[6])
    return sht_type, dataset, filename, tp, true_headers, predicted_cleaned, predicted_raw

def metrics(tp, true_headers, predicted_cleaned, predicted_raw):
    assert predicted_raw >= predicted_cleaned, "predicted_raw should be >= predicted_cleaned"
    if predicted_raw == 0 and true_headers == 0:
        return 1, 1, 1, 1
    elif predicted_raw == 0 or true_headers == 0:
        recall = 0
        precision = 0
        f1 = 0
        if predicted_raw == 0:
            fidelity = 1.0
        else:
            fidelity = predicted_cleaned / predicted_raw
        return recall, precision, f1, fidelity
    
    assert predicted_raw > 0 and true_headers > 0
    recall = tp / true_headers
    precision = tp / predicted_raw
    if recall + precision == 0:
        f1 = 0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    fidelity = predicted_cleaned / predicted_raw
    return recall, precision, f1, fidelity


def get_all_results():
    result_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval", "header_identification_raw_llm_log.csv")
    with open(result_path, "r") as f:
        lines = f.readlines()

    m_results = dict() # method -> dataset -> filename -> metrics
    for line in lines[1:]:
        sht_type, dataset, filename, tp, true_headers, predicted_cleaned, predicted_raw = parse_csv_line(line)
        if sht_type not in m_results:
            m_results[sht_type] = dict()
        if dataset not in m_results[sht_type]:
            m_results[sht_type][dataset] = dict()
        m = metrics(tp, true_headers, predicted_cleaned, predicted_raw)
        assert all(0 <= m <= 1 for m in m), f"Metrics out of range for {sht_type}, {dataset}, {filename}: {m}"
        m_results[sht_type][dataset][filename] = m
    
    return m_results

def tab_pr():
    m_results = get_all_results()
    
    tab_str = ""
    for sht_type in m_results:
        tab_str += f"{sht_type} & "
        for metric in [0, 1, 2, 3]: # recall, precision, f1
            for dataset in ['civic', 'contract', 'qasper', 'finance']:
                vals = [m_results[sht_type][dataset][filename][metric] for filename in m_results[sht_type][dataset]]
                avg_val = sum(vals) / len(vals)
                tab_str += f"{avg_val * 100:.2f}\\% & "
        tab_str = tab_str.rstrip(" & ") + " \\\\\n"

    print(tab_str)


    

if __name__ == "__main__":
    tab_pr()
