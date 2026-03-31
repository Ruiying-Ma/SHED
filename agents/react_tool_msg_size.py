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

def get_tool_result_size(dataset, filename, query_id, model):
    answer_path = get_result_path(dataset, model, "react_agent")
    msg_path = Path(str(answer_path).replace("/core/", "/other/messages/")).parent / dataset / f"query{query_id}.txt"
    checkpoint_path = Path(str(msg_path).replace("/messages/", "/checkpoints/").replace(".txt", ".pkl"))
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = list(pickle.load(f))

    latest_state: StateSnapshot = checkpoint_data[0]
    messages = latest_state.values['messages']
    tool_msg_list = [msg.content for msg in messages if isinstance(msg, ToolMessage)]
    doc_txt = get_doc_txt(dataset, query_id)
    return sum(len(tool_msg) for tool_msg in tool_msg_list) / len(doc_txt)



if __name__ == "__main__":
    model = "gpt-5-mini"

    for dataset in DATASET_LIST:
        queries_path = Path(DATA_ROOT_FOLDER) / dataset / "queries.json"
        with open(queries_path, 'r') as file:
            queries = json.load(file)
        query_id_list = [qinfo["id"] for qinfo in queries]  
        tool_msg_size_list = []
        for query_id in query_id_list:
            try:
                tc_size = get_tool_result_size(dataset, query_id, model)
                tool_msg_size_list.append(tc_size)
            except Exception as e:
                pass
                # logging.warning(f"Failed to get tool call num for {dataset} query {query_id}: {e}")
        
        avg_tool_msg_size = sum(tool_msg_size_list) / len(tool_msg_size_list) if tool_msg_size_list else 0
        print(f"Model: {model}, Dataset: {dataset}, Average Tool Message Size: {avg_tool_msg_size:.4f}")