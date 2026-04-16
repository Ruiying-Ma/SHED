import os
import sys
sys.path.append("/home/ruiying/SHTRAG")
import json
import logging
import logging_config
import eval.utils
from collections import Counter
from typing import List, Dict
import config
from structured_rag import utils as struct_utils

def format_table_as_markdown_kv(cell_list):
    table_matrix = dict() # (row_id, col_id) -> cell_text
    for cell in cell_list:
        row_id_list = range(cell["start_row_offset_idx"], cell["end_row_offset_idx"])
        col_id_list = range(cell["start_col_offset_idx"], cell["end_col_offset_idx"])
        assert len(row_id_list) == cell['row_span']
        assert len(col_id_list) == cell['col_span']
        for rid in row_id_list:
            for cid in col_id_list:
                if (rid, cid) not in table_matrix:
                    table_matrix[(rid, cid)] = cell
                else:
                    table_matrix[(rid, cid)]['text'] += "\n" + cell['text'].strip()

    # check rows
    row_ids = set([pos[0] for pos in table_matrix.keys()])
    assert sorted(list(row_ids)) == list(range(len(row_ids)))
    # check cols
    col_ids = set([pos[1] for pos in table_matrix.keys()])
    assert sorted(list(col_ids)) == list(range(len(col_ids)))
    

    markdown_kv_str = ""
    for rid in range(len(row_ids)):
        for cid in range(len(col_ids)):
            assert (rid, cid) in table_matrix
            cell = table_matrix[(rid, cid)]
            if 'in_context' not in cell:
                continue
            if cell['column_header'] == True:
                continue
            if cell['text'].strip() == "":
                continue

            assert 'in_context' in cell and cell['in_context'] == True

            suffix = ""
            if cell['row_section'] == True:
                suffix = "# "
            if cell['row_header'] == True:
                assert suffix == ""
                suffix = "## "

            if suffix != "":
                # tuple id
                markdown_kv_str += f"{suffix}{cell['text']}\n"
            else:
                # tuple value
                candid_column_header_pos = []
                for rrid in range(0, rid):
                    if table_matrix[(rrid, cid)]['column_header'] == True and ('in_context' in table_matrix[(rrid, cid)]):
                        candid_column_header_pos.append((rrid, cid))
                
                if len(candid_column_header_pos) == 0:
                    markdown_kv_str += f"{cell['text']}\n"
                else:
                    if not sorted([p[0] for p in candid_column_header_pos]) == list(range(min(p[0] for p in candid_column_header_pos), max(p[0] for p in candid_column_header_pos)+1)):
                        # logging.warning("Non-continuous column headers found!")
                        pass
                    sorted_header_pos = sorted(candid_column_header_pos, key=lambda x: x[0])
                    txt = ""
                    for hid, header_pos in enumerate(sorted_header_pos):
                        txt += " " * hid + "- " + table_matrix[header_pos]['text'] + "\n"
                    
                    txt = txt.strip()
                    txt += ": " + cell['text'] + "\n"
                    markdown_kv_str += txt

    return markdown_kv_str.strip()


def reform_context(context: str, node_clustering: List[Dict], table_list: List[Dict], order):
    chunk_list = context.split("\n\n")

    # map chunk to clusters
    m_chunk_to_cluster = dict()
    for chunk_id, chunk in enumerate(chunk_list):
        candid_cluster_list = []
        for cluster_id, cluster in enumerate(node_clustering):
            if chunk.strip() in cluster['text']:
                candid_cluster_list.append(cluster_id)
        # assert len(candid_cluster_list) > 0, f"{chunk}"
        m_chunk_to_cluster[chunk_id] = candid_cluster_list

    
    prev_cluster_id = -1
    for chunk_id, chunk in enumerate(chunk_list):
        if len(m_chunk_to_cluster[chunk_id]) == 0:
            m_chunk_to_cluster[chunk_id] = None
            continue
        if chunk_id == 0:
            m_chunk_to_cluster[chunk_id] = min(m_chunk_to_cluster[chunk_id])
            prev_cluster_id = m_chunk_to_cluster[chunk_id]
        else:
            if order == True:
                candid_cluster_ids = [i for i in m_chunk_to_cluster[chunk_id] if i >= prev_cluster_id]
                if len(candid_cluster_ids) == 0:
                    m_chunk_to_cluster[chunk_id] = min(m_chunk_to_cluster[chunk_id])
                    # logging.warning(f"No ordered cluster found for chunk id {chunk_id}:\n{chunk_list[chunk_id-1]}\n{chunk}")
                else:
                    assert len(candid_cluster_ids) > 0, f"{m_chunk_to_cluster}\n{chunk_list[chunk_id-1]}\n{chunk}"
                    m_chunk_to_cluster[chunk_id] = min(candid_cluster_ids)
                    prev_cluster_id = m_chunk_to_cluster[chunk_id]
            else:
                m_chunk_to_cluster[chunk_id] = min(m_chunk_to_cluster[chunk_id])
                prev_cluster_id = m_chunk_to_cluster[chunk_id]
    
    # get table chunks
    m_chunk_table = dict()
    for chunk_id, chunk in enumerate(chunk_list):
        if m_chunk_to_cluster[chunk_id] is None:
            continue
        cluster_id = m_chunk_to_cluster[chunk_id]
        cluster = node_clustering[cluster_id]
        cluster_is_table = (cluster["type"] == "Table")
        if cluster_is_table == False:
            continue
        candid_table_list = [table for table in table_list if cluster['page_number'] in [p['page_no'] for p in table['prov']]]
        if len(candid_table_list) == 0:
            continue
        
        cleaned_chunk = "".join(chunk.split()).lower()
        table_pos_list = [[] for _ in candid_table_list]
        for table_id, table in enumerate(candid_table_list):
            cell_list = table['data']['table_cells']
            for cell_id, cell in enumerate(cell_list):
                cell_text = "".join(cell['text'].split()).lower()
                if cell_text in cleaned_chunk:
                    table_pos_list[table_id].append(cell_id)
        candid_table_id, max_pos_list = max([(table_id, pos_list) for table_id, pos_list in enumerate(table_pos_list)], key=lambda x: len(x[1]))
        if len(max_pos_list) == 0:
            continue
        candid_table = candid_table_list[candid_table_id]
        for pos in max_pos_list:
            candid_table['data']['table_cells'][pos]['in_context'] = True

        m_chunk_table[chunk_id] = candid_table

    used_table = set()
    new_context = ""
    for chunk_id, chunk in enumerate(chunk_list):
        if chunk_id in m_chunk_table:
            table = m_chunk_table[chunk_id]
            table_id = table['self_ref']
            if table_id in used_table:
                continue
            used_table.add(table_id)
            table_markdown_str = format_table_as_markdown_kv(table['data']['table_cells'])
            new_context += "```markdown\n" + table_markdown_str.strip() + "\n```\n\n"
        else:
            new_context += chunk + "\n\n"
    return new_context.strip()


if __name__ == "__main__":
    queries_path = os.path.join(config.DATA_ROOT_FOLDER, "finance", "queries.json")
    # for rag_config in config.CONTEXT_CONFIG_LIST:
    for rag_config in (
        [("sht", "intrinsic", "sbert", True, True, True, 0.2)] +
        # [("sht", "grobid", "sbert", True, True, True, 0.2)] + 
        [("sht", "wide", "sbert", True, True, True, 0.2)] + 
        [("sht", "deep", "sbert", True, True, True, 0.2)] + 
        [("sht", "llm_txt", "sbert", True, True, True, 0.2)] + 
        [("sht", "llm_vision", "sbert", True, True, True, 0.2)] +
        [("sht", None, "sbert", True, True, True, 0.2)]
    ):
        orig_context_path = config.get_config_jsonl_path("finance", rag_config)
        assert "l1.h1" in orig_context_path, orig_context_path
        orig_context_path = orig_context_path.replace("l1.h1", "exclude_h_for_token_count")
        assert os.path.exists(orig_context_path), orig_context_path
        new_context_path = orig_context_path.replace("context.jsonl", "reform_table_context.jsonl")
        existing_context_ids = set()
        if os.path.exists(new_context_path):
            with open(new_context_path, "r") as f:
                for line in f.readlines():
                    context_info = json.loads(line)
                    existing_context_ids.add(context_info['id'])
        
        print(f"\n\nProcessing RAG config: {rag_config}\nOriginal context path: {orig_context_path}\nNew context path: {new_context_path}")
        
        node_clustering_folder = "/home/ruiying/SHTRAG/data/finance/node_clustering"
        table_folder = "/home/ruiying/SHTRAG/data/finance/docling_tables"

        with open(queries_path, "r") as f:
            query_list = json.load(f)
        with open(orig_context_path, "r") as f:
            context_list = [json.loads(line) for line in f.readlines()]
        
        assert len(query_list) == len(context_list)
        for query_info, context_info in zip(query_list, context_list):
            
            
            assert query_info['id'] == context_info['id']
            if context_info['id'] in existing_context_ids:
                print(f"Skip existing context id: {context_info['id']}")
                continue
            file_name = query_info['file_name']
            
            node_clustering_path = os.path.join(node_clustering_folder, f"{file_name}.json")
            print(f"\tLoading node clustering from: {node_clustering_path}")
            assert os.path.exists(node_clustering_path), node_clustering_path
            table_path = os.path.join(table_folder, f"{file_name}.json")
            print(f"\tLoading tables from: {table_path}")
            assert os.path.exists(table_path), table_path
            with open(node_clustering_path, "r") as f:
                node_clustering = json.load(f)
            with open(table_path, "r") as f:
                table_list = json.load(f)

            
            
            orig_context = context_info['context']
            print(f"\tReforming context for id: {context_info['id']}, file: {file_name}")
            order = True
            if rag_config[0] != 'sht':
                order = False
            new_context = reform_context(orig_context, node_clustering, table_list, order)
            new_context_info = {
                "id": context_info['id'],
                "context": new_context
            }
            with open(new_context_path, "a") as f:
                f.write(json.dumps(new_context_info) + "\n")
            existing_context_ids.add(context_info['id'])
            print(f"Reformed context for id: {context_info['id']}")

            





