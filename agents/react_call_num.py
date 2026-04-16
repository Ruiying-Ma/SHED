import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
import logging_config
import time
from collections import Counter
import re
import pandas as pd
from calendar import month_name
import pickle
from datetime import datetime

from langgraph.types import StateSnapshot
from langchain.messages import ToolMessage, HumanMessage, AIMessage, SystemMessage, AnyMessage
from langchain_core.messages.base import BaseMessage
from pathlib import Path
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ModelCallLimitMiddleware, ToolRetryMiddleware, AgentMiddleware, ToolCallLimitMiddleware
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware.types import ResponseT, ContextT
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage, HumanMessage, AIMessage, SystemMessage, AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from config import DATASET_LIST, DATA_ROOT_FOLDER
from agents.utils import get_result_path, llm, get_doc_txt, get_toc_numbered
from agents.accuracy import INVALID_CIVIC_QUERY_IDS

def compute_latency(start_str, end_str):
    fmt = "%Y-%m-%d__%H-%M-%S"
    
    start = datetime.strptime(start_str, fmt)
    end = datetime.strptime(end_str, fmt)
    
    latency = end - start
    return latency.total_seconds()

def get_trajectory_call_num(dataset, query_id, model, method, sht_type):
    answer_path = get_result_path(dataset, model, method, sht_type)
    msg_path = Path(str(answer_path).replace("/core/", "/other/messages/")).parent / dataset / f"query{query_id}.txt"
    checkpoint_path = Path(str(msg_path).replace("/messages/", "/checkpoints/").replace(".txt", ".pkl"))
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = list(pickle.load(f))

    latest_state: StateSnapshot = checkpoint_data[0]
    messages = latest_state.values['messages']
    tool_call_num = sum(1 for msg in messages if isinstance(msg, ToolMessage))
    tool_call_latencies = [compute_latency(msg.artifact['start_timestamp'], msg.artifact['end_timestamp']) for msg in messages if isinstance(msg, ToolMessage)]
    avg_tool_call_latency = sum(tool_call_latencies) / len(tool_call_latencies) if len(tool_call_latencies) > 0 else None

    read_section_tool_call_num = 0
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "read_section":
                    read_section_tool_call_num += 1
    model_call_num = sum(1 for msg in messages if isinstance(msg, AIMessage))

    model_call_latencies = [compute_latency(msg.response_metadata['start_timestamp'], msg.response_metadata['end_timestamp']) for msg in messages if isinstance(msg, AIMessage)]
    avg_model_call_latency = sum(model_call_latencies) / len(model_call_latencies) if len(model_call_latencies) > 0 else None

    return tool_call_num, model_call_num, read_section_tool_call_num, avg_model_call_latency, avg_tool_call_latency

    
    


if __name__ == "__main__":
    model = "gpt-5.4"
    # model = "gpt-5-mini"
    method_list = [
        "react_agent_grep_next_chunk_notoc",
        "react_agent_clean", 
        "react_agent_grep_next_chunk_clean",
    ]
    sht_type = 'intrinsic'
    results = []
    for method in method_list:
        for dataset in DATASET_LIST[:-1]:
            queries_path = Path(DATA_ROOT_FOLDER) / dataset / "queries.json"
            with open(queries_path, 'r') as file:
                queries = json.load(file)
            query_id_list = [qinfo["id"] for qinfo in queries] 
            if dataset == "civic":
                query_id_list = [qid for qid in query_id_list if qid not in INVALID_CIVIC_QUERY_IDS] 
            tool_call_num_list = []
            model_call_num_list = []
            read_section_tool_call_num_list = []
            avg_model_call_latency_list = []
            avg_tool_call_latency_list = []
            for query_id in query_id_list:
                try:
                    tc_num, md_call_num, rs_tc_num, avg_md_call_lat, avg_tc_lat = get_trajectory_call_num(dataset, query_id, model, method, sht_type)
                    tool_call_num_list.append(tc_num)
                    model_call_num_list.append(md_call_num)
                    read_section_tool_call_num_list.append(rs_tc_num)
                    if avg_md_call_lat != None:
                        avg_model_call_latency_list.append(avg_md_call_lat)
                    if avg_tc_lat != None:
                        avg_tool_call_latency_list.append(avg_tc_lat)
                except Exception as e:
                    # pass
                    logging.warning(f"Failed to get tool call num for {dataset} query {query_id}: {e}")
            avg_tool_call_num = sum(tool_call_num_list) / len(tool_call_num_list)
            avg_model_call_num = sum(model_call_num_list) / len(model_call_num_list)
            avg_read_section_tool_call_num = sum(read_section_tool_call_num_list) / len(read_section_tool_call_num_list)
            avg_model_call_latency = sum(avg_model_call_latency_list) / len(avg_model_call_latency_list)
            avg_tool_call_latency = sum(avg_tool_call_latency_list) / len(avg_tool_call_latency_list)
            assert len(tool_call_num_list) == len(model_call_num_list) == len(read_section_tool_call_num_list)

            results.append({
                "dataset": dataset,
                "method": method,
                "avg_model_call_num": avg_model_call_num,
                "avg_tool_call_num": avg_tool_call_num,
                "avg_read_section_tool_call_num": avg_read_section_tool_call_num,
                "avg_model_call_latency": avg_model_call_latency,
                "avg_tool_call_latency": avg_tool_call_latency,
            })

    df = pd.DataFrame(results)
    df['method'] = pd.Categorical(df['method'], categories=method_list, ordered=True)
    df = df.sort_values('method')
    print(df.pivot(index='method', columns='dataset', values='avg_model_call_num').round(4).to_csv())
    print(df.pivot(index='method', columns='dataset', values='avg_tool_call_num').round(4).to_csv())
    # print(df.pivot(index='method', columns='dataset', values='avg_read_section_tool_call_num').round(4).to_csv())
    print(df.pivot(index='method', columns='dataset', values='avg_model_call_latency').round(4).to_csv())
    print(df.pivot(index='method', columns='dataset', values='avg_tool_call_latency').round(4).to_csv())

