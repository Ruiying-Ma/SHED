import os
import json

dataset = 'finance'

folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", dataset, "sbert.gpt-4o-mini.c100.s100", "sht")

total_cost = 0.0
for sht_name in os.listdir(folder):
    sht_path = os.path.join(folder, sht_name)
    with open(sht_path, 'r') as f:
        sht = json.load(f)
    
    
    total_cost += (sht['estimated_cost']['input_tokens'] * 0.15 + sht['estimated_cost']['output_tokens'] * 0.6) / 1000000
    
print(f"Total estimated cost for dataset '{dataset}': ${total_cost:.6f}")