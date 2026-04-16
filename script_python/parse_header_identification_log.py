import os


def parse_line(line: str, line_id: int):
    """
    Parse a log line into: sht_type,filename,tp,true,predicted.
    Line formatting:
    [{sht_type}][{filename}] Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, #True Headers: {len(true_header_list)}, #Predicted Headers: {len(header_list)}
    An example line:
    [shed][01262022-1835] Recall: 0.9500, Precision: 0.9500, F1: 0.9500, #True Headers: 40, #Predicted Headers: 40
    """
    parts = line.split("] ")
    sht_type_filename = parts[0][1:]  # Remove the leading '['
    metrics_part = parts[1]

    sht_type, filename = sht_type_filename.split("][")
    assert sht_type in ['shed', 'grobid', 'llm_vision', 'llm_txt'], f"Unknown sht_type: {sht_type}"

    line_id = line_id % 592
    if line_id in range(0, 19):
        datatset = "civic"
    elif line_id in range(19, 19+73):
        datatset = "contract"
    elif line_id in range(19+73, 19+73+416):
        datatset = "qasper"
    else:
        assert line_id in range(19+73+416, 19+73+416+84), f"line_id out of range: {line_id}"
        datatset = "finance"

    pdf_path = os.path.join(
        "/home/ruiying/SHTRAG/data",
        datatset,
        "pdf",
        f"{filename}.pdf"
    )
    assert os.path.exists(pdf_path), f"PDF file does not exist: {pdf_path}"
    

    metrics = metrics_part.split(", ")
    
    recall = float(metrics[0].split(": ")[1])
    assert "Recall" in metrics[0], f"Unexpected metric format: {metrics[0]}"
    
    precision = float(metrics[1].split(": ")[1])
    assert "Precision" in metrics[1], f"Unexpected metric format: {metrics[1]}"
    
    f1 = float(metrics[2].split(": ")[1])
    assert "F1" in metrics[2], f"Unexpected metric format: {metrics[2]}"

    true_headers = int(metrics[3].split(": ")[1])
    assert "True Headers" in metrics[3], f"Unexpected metric format: {metrics[3]}"

    predicted_headers = int(metrics[4].split(": ")[1])
    assert "Predicted Headers" in metrics[4], f"Unexpected metric format: {metrics[4]}"

    tp = round(recall * true_headers)  # True Positives
    return sht_type, datatset, filename, tp, true_headers, predicted_headers

if __name__ == "__main__":
    log_path = "/home/ruiying/SHTRAG/eval/header_identification_log.txt"
    with open(log_path, "r") as f:
        lines = f.readlines()

    dst_path = "/home/ruiying/SHTRAG/eval/header_identification_log.csv"
    tab_str = "sht_type,dataset,filename,tp,true,predicted_cleaned\n"
    for line_id, line in enumerate(lines):
        if not line.startswith("["):
            continue
        sht_type, dataset, filename, tp, true_headers, predicted_headers = parse_line(line.strip(), line_id)
        tab_str += f"{sht_type},{dataset},{filename},{tp},{true_headers},{predicted_headers}\n"

    with open(dst_path, "w") as f:
        f.write(tab_str)