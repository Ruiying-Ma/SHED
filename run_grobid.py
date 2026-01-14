import re
from structured_rag import SHTBuilderConfig, SHTBuilder, StructuredRAG
import os
import json
from datetime import datetime
import logging
import logging_config
import traceback

def concat(list_of_str, concatenator) -> str:
    return concatenator.join([s.strip() for s in list_of_str if len(s.strip()) > 0])

def handle_author(author: dict):
    name_info = [author["first"]]
    name_info += author["middle"]
    name_info += [
        author["last"],
        author["suffix"],
    ]

    aff_info = []
    if len(author["affiliation"]) > 0:
        aff_info = [author["affiliation"]["laboratory"], author["affiliation"]["institution"]]
        aff_info += list(author["affiliation"]["location"].values())

    email_info = author["email"]

    return concat([concat(name_info, " "), concat(aff_info, ", "), email_info], "\n")

def handle_paragraph(para: dict, m_ref_entries: dict, para_idx: int) -> str:
    if len(para["eq_spans"]) > 0:
        assert para["text"] == "EQUATION"
        return concat([eq_span["raw_str"] for eq_span in para["eq_spans"]], "\n\n")
    
    ref_texts = []
    for ref_span in para["ref_spans"]:
        ref_id = ref_span["ref_id"]
        if ref_id != None and ref_id in m_ref_entries:
            ref_entry = m_ref_entries[ref_id]
            if "first_para_idx" not in ref_entry:
                ref_entry["first_para_idx"] = para_idx
                ref_texts.append(handle_ref_entry(
                    ref_entry=ref_entry,
                    id=para["text"][ref_span["start"]:ref_span["end"]]
                ))

    return concat([para["text"]] + ref_texts, "\n\n")

def handle_ref_entry(ref_entry: dict, id: str) -> str:
    if ref_entry["type_str"] == "table":
        table_pattern = r"(?i)(tab|table)([.:])?\s*(\d+)?"
        if re.match(table_pattern, ref_entry["text"].strip().lower()):
            caption = ref_entry["text"].strip()
        else:
            if id != None:
                caption = f"Table {id.strip()} " + ref_entry["text"].strip()
            else:
                caption = f"Table " + ref_entry["text"].strip()
        return concat([ref_entry["content"], caption], "\n\n")
    
    else:
        assert ref_entry["type_str"] == "figure"
        fig_pattern = r"(fig|figure)([.:])?\s*(\d+)?"
        if re.match(fig_pattern, ref_entry["text"].strip().lower()):
            caption = ref_entry["text"].strip()
        else:
            if id != None:
                caption = f"Figure {id.strip()} " + ref_entry["text"].strip()
            else:
                caption = f"Figure " + ref_entry["text"].strip()
        return caption.strip()
    
def create_obj(id, text, type, cluster_id):
    if text.strip() == "":
        return None
    
    return {
        "id": id,
        "text": text.strip(),
        "type": type,
        "cluster_id": cluster_id,
        "features": {}
    }
        
def _extract_sec(s):
    if s == None:
        return tuple([])
    return tuple([int(num) for num in re.findall(r'\d+', s.strip())])

def _extract_heading(s):
    if s == None:
        return ""
    return s.strip()

def _get_oid(m_sec_oid, sec, title_id):
    assert sec not in m_sec_oid
    max_len = 0
    max_len_sec = None
    for candid_sec in m_sec_oid:
        if len(candid_sec) < len(sec) and list(candid_sec) == list(sec[:len(candid_sec)]):
            if len(candid_sec) > max_len:
                max_len = len(candid_sec)
                max_len_sec = candid_sec

    if max_len_sec != None:
        assert max_len > 0
        return m_sec_oid[max_len_sec]

    assert max_len == 0
    return title_id


def parse_grobid(grobid):
    '''
    An extracted grobid json file contains:
    - title (str)
    - authors (list[dict]): a list of authors:
        - first (str): first name
        - middle (list[str]): a list of middle names
        - last (str): last name
        - suffix (str)
        - affiliation (dict)
            - laboratory (str)
            - institution (str)
            - location (dict): concatenate values (str) by commas
        - email (str)
    - pdf_parse (dict):
        - abstract (list[dict]): a list of paragraphs
        - body_text (list): a list of paragraphs
            - text (str)
            - ref_spans (list[dict]): add corresponding figures/tables at the end of first reference
            - eq_spans (list[dict]): the corresponding parent text is always EQUATION
                - raw_str (str)
            - section (str): section name
            - sec_num (str | None): separated by "."
        - back_matter (list): a list of paragraphs
        - bib_entries (dict): 
            - BIBREF*: {raw_text (str)}
        - ref_entries (dict):
            - FIGREF*: 
                - text (str): caption
            - TABREF*:
                - content (str): table contents
                - num (|null) : always null
                - html (|null): always null
                - text (str): caption
    '''

    '''
    fake the node clustering result:
    - id
    - text
    - type:
        - Section header: section headings
        - Title: title
        - List item: bib entry
        - Text: author, paragraph, ref entry
    - cluster_id == section_num for section header/list items/titles
        - Remember to add section Reference for bib entries

    The use SHTBuilder
    '''

    objects = []

    title_id = -1

    SECTION_HEADER = "Section header"
    TITLE = "Title"
    LIST_ITEM = "List item"
    TEXT = "Text"

    # title
    title_obj = create_obj(
        id=len(objects), 
        text=grobid["title"], 
        type=TITLE, 
        cluster_id=-1
    )
    if title_obj != None:
        objects.append(title_obj)
        title_id = 0

    # authors
    for author in grobid["authors"]:
        author_obj = create_obj(
            id=len(objects), 
            text=handle_author(author), 
            type=TEXT,
            cluster_id=title_id
        )
        if author_obj != None:
            objects.append(author_obj)

    # abstract
    abstract_id = title_id
    if grobid["abstract"].strip() != "":
        abstract_obj_h = create_obj(
            id=len(objects),
            text="Abstract",
            type=SECTION_HEADER,
            cluster_id=title_id
        )
        assert abstract_obj_h != None
        abstract_id = abstract_obj_h["id"]
        objects.append(abstract_obj_h)
    
        abstract_obj_t = create_obj(
            id=len(objects),
            text=grobid["abstract"],
            type=TEXT,
            cluster_id=abstract_id
        )
        assert abstract_obj_t != None
        objects.append(abstract_obj_t)

    # body
    paragraphs = grobid["pdf_parse"]["body_text"] + grobid["pdf_parse"]["back_matter"] # list
    m_ref_entries = grobid["pdf_parse"]["ref_entries"] # dict
    assert all(["first_para_idx" not in ref_entry for ref_entry in m_ref_entries.values()])

    m_sec_heading = {}
    m_sec_oid = {}
    m_heading_oid = {}

    last_heading_oid = abstract_id

    for para_idx, para in enumerate(paragraphs):
        sec = _extract_sec(para["sec_num"])
        heading = _extract_heading(para['section'])
        # valid sec
            # new: add heading_object, update m_sec_heading & m_sec_oid, add para_object
            # old: check m_sec_heading, add para_object
        # invalid sec
            # valid heading
                # new: adding heading_object, update m_heading_oid, add para_object
                # old: add para_object
            # invalid heading: add para_object
        if len(sec) > 0:
            assert heading != ""
            if sec not in m_sec_heading:
                heading_obj = create_obj(
                    id=len(objects),
                    text=".".join([str(sid) for sid in sec]) + " " + heading,
                    type=SECTION_HEADER,
                    cluster_id=_get_oid(m_sec_oid, sec, title_id)
                )
                assert heading_obj != None
                m_sec_heading[sec] = heading
                m_sec_oid[sec] = heading_obj["id"]
                objects.append(heading_obj)
                para_object_cluster_id = heading_obj["id"]
                last_heading_oid = heading_obj["id"]
            else:
                if heading == m_sec_heading[sec]:
                    para_object_cluster_id = m_sec_oid[sec]
                else:
                    if heading not in m_heading_oid:
                        heading_obj = create_obj(
                            id=len(objects),
                            text=heading,
                            type=SECTION_HEADER,
                            cluster_id=last_heading_oid
                        )
                        assert heading_obj != None
                        m_heading_oid[heading] = heading_obj["id"]
                        objects.append(heading_obj)
                        para_object_cluster_id = heading_obj["id"]
                        last_heading_oid = heading_obj["id"]
                    else:
                        para_object_cluster_id = m_heading_oid[heading]
        else:
            if heading != "":
                if heading not in m_heading_oid:
                    heading_obj = create_obj(
                        id=len(objects),
                        text=heading,
                        type=SECTION_HEADER,
                        cluster_id=title_id
                    )
                    assert heading_obj != None
                    m_heading_oid[heading] = heading_obj["id"]
                    objects.append(heading_obj)
                    para_object_cluster_id = heading_obj["id"]
                    last_heading_oid = heading_obj["id"]
                else:
                    para_object_cluster_id = m_heading_oid[heading]
            else:
                para_object_cluster_id = title_id
        para_object = create_obj(
            id=len(objects),
            text=handle_paragraph(para, m_ref_entries, para_idx),
            type=TEXT,
            cluster_id=para_object_cluster_id
        )
        if para_object != None:
            objects.append(para_object)

    # remaining references
    for ref_entry in m_ref_entries.values():
        if "first_para_idx" not in ref_entry:
            ref_obj = create_obj(
                id=len(objects),
                text=handle_ref_entry(ref_entry, None),
                type=TEXT, 
                cluster_id=title_id
            )
            if ref_obj != None:
                objects.append(ref_obj)

    # reference
    reference_id = title_id
    sorted_bib_entries = sorted(list(grobid["pdf_parse"]["bib_entries"].values()), key=lambda b: int(b["ref_id"].replace("b", "")))
    if any([b["raw_text"].strip() != "" for b in sorted_bib_entries]):
        reference_obj_h = create_obj(
            id=len(objects),
            text="References",
            type=SECTION_HEADER,
            cluster_id=title_id,
        )
        assert reference_obj_h != None
        reference_id = reference_obj_h["id"]
        objects.append(reference_obj_h)
        
        for bib_entry in sorted_bib_entries:
            bib_entry_obj = create_obj(
                id=len(objects),
                text=bib_entry['raw_text'],
                type=LIST_ITEM,
                cluster_id=reference_id
            )
            if bib_entry_obj != None:
                objects.append(bib_entry_obj)

    return objects


def grobid2sht(dataset):
    grobid_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "grobid")
    grobid_extraction_dir = os.path.join(grobid_dir, "grobid")
    assert os.path.exists(grobid_extraction_dir)
    grobid_clustering_dir = os.path.join(grobid_dir, "node_clustering")
    os.makedirs(grobid_clustering_dir, exist_ok=True)
    sht_skeleton_dir = os.path.join(grobid_dir, "sbert.gpt-4o-mini.c100.s100", "sht_skeleton")
    sht_dir = os.path.join(grobid_dir, "sbert.gpt-4o-mini.c100.s100", "sht")
    sht_vis_dir = os.path.join(grobid_dir, "sbert.gpt-4o-mini.c100.s100", "sht_vis")
    os.makedirs(sht_skeleton_dir, exist_ok=True)
    os.makedirs(sht_dir, exist_ok=True)
    os.makedirs(sht_vis_dir, exist_ok=True)

    # # build_tree_skeleton
    # for json_name in os.listdir(grobid_extraction_dir):
    #     if os.path.exists(os.path.join(sht_skeleton_dir, json_name)):
    #         print(f"SHT skeleton for {json_name} is already existed.")
    #         continue
    #     print("Building SHT skeleton for", json_name)
    #     assert json_name.endswith(".json")
    #     grobid_extraction_path = os.path.join(grobid_extraction_dir, json_name)
    #     assert os.path.exists(grobid_extraction_path)
    #     with open(grobid_extraction_path, 'r') as file:
    #         grobid = json.load(file)
    
    #     objects = parse_grobid(grobid)
    #     # store to node_clustering/
    #     with open(os.path.join(grobid_clustering_dir, json_name), 'w') as file:
    #         json.dump(objects, file, indent=4)

    #     sht_builder = SHTBuilder(
    #         config=SHTBuilderConfig(
    #             store_json=os.path.join(sht_skeleton_dir, json_name),
    #             load_json=None,
    #             chunk_size=100,
    #             summary_len=100,
    #             embedding_model_name="sbert",
    #             summarization_model_name="gpt-4o-mini",
    #         )
    #     )
    #     logging.info(f"Built SHT skeleton for {json_name}")
    #     sht_builder.build(objects)
    #     sht_builder.check()
    #     sht_builder.store2json()
    #     sht_builder.visualize(vis_path=os.path.join(sht_vis_dir, json_name.replace(".json", ".vis")))

    # # add summaries and embeddings
    # for json_name in os.listdir(grobid_extraction_dir):
    #     try:
    #         assert json_name.endswith(".json")
    #         sht_skeleton_path = os.path.join(sht_skeleton_dir, json_name)
    #         sht_path = os.path.join(sht_dir, json_name)
    #         sht_builder = SHTBuilder(
    #             config=SHTBuilderConfig(
    #                 store_json=sht_path,
    #                 load_json=sht_skeleton_path,
    #                 chunk_size=100,
    #                 summary_len=100,
    #                 embedding_model_name="sbert",
    #                 summarization_model_name="gpt-4o-mini",
    #             )
    #         )
    #         logging.info(f"Processing {json_name} with sbert")
    #         sht_builder.build(None)
    #         sht_builder.check()
    #         logging.info(f"\tAdding summaries for {json_name}")
    #         sht_builder.add_summaries()
    #         sht_builder.check()
    #         node_ids = list(range(len(sht_builder.tree["nodes"])))
    #         logging.info(f"\tAdding embeddings for {json_name}")
    #         sht_builder.add_embeddings(node_ids)
    #         sht_builder.store2json()
    #         with open(sht_builder.store_json, 'w') as file:
    #             json.dump(sht_builder.tree, file, indent=4)

    #         cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         print(f"\t[{cur_time}]✅ Finished processing {json_name} with sbert")
    #     except Exception as e:
    #         cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         print(f"\t[{cur_time}]❌ Error processing {json_name} with sbert: {str(e)}")
    #         continue


    # index / generate context for queries
    root_dir = os.path.dirname(grobid_dir)
    queries_path = os.path.join(root_dir, "queries.json")
    with open(queries_path, 'r') as file:
        queries_info = json.load(file)

    for qid, query_info in enumerate(queries_info):
        file_name = query_info["file_name"]
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{cur_time}] Processing {qid}...")
        try:
            sht_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset, "grobid", "sbert.gpt-4o-mini.c100.s100", "sht", f"{file_name}.json")
            assert os.path.exists(sht_path), f"No SHT: {sht_path} doesn't exist"
            rag = StructuredRAG(
                root_dir=grobid_dir,
                chunk_size=100,
                summary_len=100,
                node_embedding_model="sbert",
                query_embedding_model="sbert",
                summarization_model="gpt-4o-mini",
                embed_hierarchy=True,
                distance_metric="cosine",
                context_hierarchy=True,
                context_raw=True,
                context_len=1000
            )
            ### index
            rag.index(
                name=query_info["file_name"],
                query=query_info["query"],
                query_id=qid,
            )
            ### generate context
            # rag.generate_context(
            #     name=query_info["file_name"],
            #     query=query_info["query"],
            #     query_id=qid,
            # )
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\t[{cur_time}]✅ Finished processing {qid} with sbert")
        except Exception as e:
            cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\t[{cur_time}]❌ Error processing {qid}: {e}\n{traceback.print_exc()}")
            continue
            

if __name__ == "__main__": 
    grobid2sht("finance")