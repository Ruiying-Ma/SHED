import json
import os
import pandas as pd
import logging
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATASET_LIST, DATA_ROOT_FOLDER
import logging_config

if __name__ == "__main__":
    direction = "top_down"
    
    sht_type_list = [
        'deep',
        'wide',
        'grobid',
        'llm_txt_sht',
        'llm_vision_sht',
        '',
    ]

    for sht_type in sht_type_list:
        recall_list = []
        precision_list = []
        f1_list = []
        for dataset in DATASET_LIST:
            if dataset == "civic_rand_v1":
                start_id = 0
                end_id = 107
            elif dataset == 'finance_rand_v1':
                start_id = 0
                end_id = 100
            elif dataset == 'contract_rand_v0_1':
                start_id = 0
                end_id = 248
            elif dataset == 'qasper_rand_v1':
                start_id = 0
                end_id = 500
            else:
                raise ValueError(f"Invalid dataset: {dataset}")
        
            dataset_path = Path(__file__).resolve().parent / "results" / "sht_eval" / direction  / sht_type / dataset
            per_ds_recall_list = []
            per_ds_precision_list = []
            per_ds_f1_list = []
            for file_id in range(start_id, end_id):
                file_name = str(file_id)
                result_path = dataset_path / f"{file_name}.jsonl"
                if not result_path.exists():
                    # logging.warning(f"Result file not found for {dataset}/{file_name} ({sht_type}), path: {result_path}")
                    rec = 0
                    prec = 0
                    f1 = 0
                else:
                    per_doc_recall_list = []
                    per_doc_precision_list = []
                    per_doc_f1_list = []
                    with open(result_path, 'r') as f:
                        for line in f:
                            metric_info = json.loads(line)
                            rec = metric_info['max_recall']
                            prec = metric_info['max_precision']
                            f1 = metric_info['max_f1']
                            per_doc_recall_list.append(rec)
                            per_doc_precision_list.append(prec)
                            per_doc_f1_list.append(f1)
                    rec = sum(per_doc_recall_list) / len(per_doc_recall_list)
                    prec = sum(per_doc_precision_list) / len(per_doc_precision_list)
                    f1 = sum(per_doc_f1_list) / len(per_doc_f1_list)
                per_ds_recall_list.append(rec)
                per_ds_precision_list.append(prec)
                per_ds_f1_list.append(f1)
            avg_rec = sum(per_ds_recall_list) / len(per_ds_recall_list)
            avg_prec = sum(per_ds_precision_list) / len(per_ds_precision_list)
            avg_f1 = sum(per_ds_f1_list) / len(per_ds_f1_list)
            recall_list.append(avg_rec)
            precision_list.append(avg_prec)
            f1_list.append(avg_f1)
        
        print(f"{sht_type},{','.join([f'{rec:.2f}' for rec in recall_list])},{','.join([f'{prec:.2f}' for prec in precision_list])}, {','.join([f'{f1:.2f}' for f1 in f1_list])}")
            

                