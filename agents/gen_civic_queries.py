import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
import logging_config
import time

from config import DATA_ROOT_FOLDER



def gen_civic_queries():
    projects_path = os.path.join(DATA_ROOT_FOLDER, 'civic', 'projects.json')
    with open(projects_path, 'r') as f:
        m_file_proj = json.load(f)
    
    queries = []
    for filename, projects in m_file_proj.items():
        m_status_proj = dict()
        m_status_types = dict()
        for proj, info in projects.items():
            stat = info['status']
            ptype = info['type']
            if stat not in m_status_proj:
                m_status_proj[stat] = []
            m_status_proj[stat].append(proj.strip())
            if stat not in m_status_types:
                m_status_types[stat] = set()
            m_status_types[stat].add(ptype.strip())
        
        for stat, projs in m_status_proj.items():
            if len(projs) <= 1:
                continue
            if len(m_status_types[stat]) <= 1:
                continue
            
            for lid in range(1, 4):
                if lid >= len(projs):
                    break
                last_proj = projs[len(projs) - lid]
                
                qid = len(queries)
                query_str = f"Return a list of project names for all projects whose status matches the status of project '{last_proj}'. You must include projects of all types. Make sure the queried project itself is included in the list."
                queries.append({
                    'id': qid,
                    'file_name': filename,
                    'query': query_str,
                    'prompt_template': query_str,
                    'answer': projs,
                })
    

    out_path = os.path.join(DATA_ROOT_FOLDER, 'civic_new', 'queries.json')
    with open(out_path, 'w') as f:
        json.dump(queries, f, indent=4)

if __name__ == "__main__":
    gen_civic_queries()