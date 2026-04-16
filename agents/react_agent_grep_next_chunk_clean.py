import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
import logging_config
from typing import TypedDict, Annotated
import time
import pickle
from dotenv import load_dotenv
import traceback
import re
import operator

from config import DATASET_LIST, DATA_ROOT_FOLDER, get_max_window
from agents.utils import get_toc_system_message, get_system_message, get_timestamp, get_toc_textspan_clean, get_toc_numbered_clean, get_result_path, LLMResponse, grep_search, get_toc_greptext_clean

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ModelCallLimitMiddleware, ToolRetryMiddleware, AgentMiddleware, AgentState 
from langchain.messages import ToolMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool, ToolRuntime
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain.chat_models import init_chat_model
from langgraph.types import Command

from agents.react_agent import(
    create_model,
    read_section,
    DocAgentModelCallTimestampMiddleware,
    DocAgentToolArtifactMiddleware,
    messages_to_response,
)
from agents.accuracy import INVALID_CIVIC_QUERY_IDS
from agents.utils import get_doc_txt

DEBUG = False

AGENT_NAME = "react_agent_grep_next_chunk_clean"

class DocChunkAgentContext(TypedDict):
    toc_textgrep: dict # {level_str: text_span}
    toc_textspan: dict # {level_str: text_span}

class DisableParallelToolCallsMiddleware(AgentMiddleware):
    
    def wrap_model_call(self, request, handler):
        request.model_settings["parallel_tool_calls"] = False
        return handler(request)
    
    async def awrap_model_call(self, request, handler):
        request.model_settings["parallel_tool_calls"] = False
        return await handler(request)

class GrepSearchNextState(AgentState):
    patterns_matches: Annotated[list, operator.add] # [..., [pattern, [matching_chunks], ...]
    patterns_cursor: list # [..., [pattern, last_returned_idx], ...]

def get_agent_system_message(dataset):
    system_msg = get_system_message(dataset).strip() + "\n\n"+ get_toc_system_message(verbose=False).strip() + "\n\n"

    system_msg += "Initially, you are only provided with:\n" \
    "- The query\n" \
    "- The table of contents of the document\n\n" \
    "You are provided with the following tools to retrieve more information from the document:\n" \
    "- You can use the tool `read_section(section_id)` to retrieve the content of a specific section in the document. The `section_id` corresponds to the number prefix in the TOC (the part before `|`). Use the TOC to identify relevant sections, and call `read_section` **as needed** to retrieve their content.\n" \
    "- You can use `grep_search_next(pattern)` to locate relevant content in the document using regular expression matching. The tool searches the document and returns matching chunks incrementally across tool calls. On the first call with a given pattern, it returns the first matching chunk in the document. On subsequent calls with the same pattern, the tool returns the next matching chunk for that pattern, following the document's reading order. If you switch to a different pattern, the search for that new pattern will start from the beginning of the document. If you switch back to a previously used pattern, the search will resume from where it left off for that pattern. Each returned chunk contains the match(es) near its center, along with surrounding context, is prefixed with the corresponding section's number prefix and title. Use this tool when you want to find text containing specific keywords or phrases. This tool allows you to stop retrieving results for a given pattern at any time, and resume later from exactly where you left off, whenever needed.\n\n" \
    "RULES:\n" \
    "- Use tool calls to solve the task.\n" \
    "- Do not output any plain-text message until you are ready to provide the final answer.\n" \
    "- Any plain-text output will be treated as the final answer and will immediately terminate execution.\n" \
    "- Before the final answer, you may only produce tool calls.\n" \
    "- When you have obtained enough information, output a single plain-text answer with no additional text.\n" \
    "- Use tool calls **only when necessary**. If the query can be answered using the TOC alone, return the answer directly without calling any tools."

    return SystemMessage(content=system_msg.strip())

grep_search_next_schema = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": """A regular expression used to search for matches within the document contents."""},
    },
    "required": ["pattern"],
}
@tool(args_schema=grep_search_next_schema)
def grep_search_next(
    pattern: str,
    runtime: ToolRuntime,
) -> str:
    """Searches the document using full regular expression syntax and returns matching chunks, together with their corresponding section titles and number prefixes. The search results are returned incrementally across tool calls. 

A matching chunk contains at least one match of the `pattern`, with the match(es) near the center and surrounding context included. Each matching chunk is returned with its corresponding section number prefix and title at the beginning of the chunk.

How results are returned:
- On the first `grep_search_next` call with a given `pattern`, the tool returns the first matching chunk in the document.
- On the subsequent calls of `grep_search_next` with the same `pattern`, the tool returns the next matching chunk in the document, following the reading order of these chunks in the document.

Pattern switching behavior:
- If the `pattern` in the current call is different from the pattern used in the immediately previous `grep_search_next` call, the search for this new `pattern` will start from the beginning of the document, and the tool will return the first matching chunk for this new `pattern`.
- If the `pattern` in the current call has been used earlier (but not in the immediately previous `grep_search_next` call), the search will resume from where it left off for this `pattern`, and the tool will return the next matching chunk for this `pattern`.

If there are no more matching chunks to return for the given `pattern`, the tool returns 'No more matches found.'"""
    try:
        re.compile(pattern)
    except re.error as e:
        return f"Invalid regex pattern: {e}"
    
    patterns_matches : list = runtime.state['patterns_matches']
    patterns_cursor : list = runtime.state['patterns_cursor']
    m_pattern_state = dict()
    for p, c in patterns_cursor:
        m_pattern_state[p] = [c]
    for p, m in patterns_matches:
        m_pattern_state[p].append(m)

    # previous pattern 
    if pattern in m_pattern_state:
        last_returned_idx = m_pattern_state[pattern][0]
        matched_chunks = m_pattern_state[pattern][1]
        if last_returned_idx + 1 >= len(matched_chunks):
            return "No more matches found."
        
        cur_chunk = matched_chunks[last_returned_idx + 1]

        new_patterns_cursor = []
        for p, c in patterns_cursor:
            if p == pattern:
                new_patterns_cursor.append([p, last_returned_idx + 1])
            else:
                new_patterns_cursor.append([p, c])

        return Command(
            update={
                "messages": [ToolMessage(content=cur_chunk, tool_call_id=runtime.tool_call_id)],
                "patterns_cursor": new_patterns_cursor,
            }
        )
    

    # new pattern
    # return all matching chunks for the new pattern as a list in reading order
    all_matching_chunks = []
    for num_pre, text_info in runtime.context['toc_textgrep'].items():
        heading = text_info['heading']
        # try heading matches
        heading_matches = list(re.finditer(pattern, heading))
        if len(heading_matches) > 0:
            all_matching_chunks.append(f"{num_pre} | {heading.strip()}")
        # try content matches
        doc_txt = text_info['text']
        matches = list(re.finditer(pattern, doc_txt))
        matched_chunks = []
        last_chunk_end = -1
        window_len = 500
        for m in matches:
            start_idx = m.start()
            end_idx = m.end()

            if end_idx <= last_chunk_end:
                continue # skip matches that are fully contained within the previously returned chunk to avoid redundancy

            # expand to include surrounding context, with the match near the center
            chunk_start = max(0, start_idx - window_len)
            chunk_end = min(len(doc_txt), end_idx + window_len)
            chunk_txt = doc_txt[chunk_start:chunk_end].strip()
            if chunk_start != 0:
                chunk_txt = "..." + chunk_txt
            if chunk_end != len(doc_txt):
                chunk_txt = chunk_txt + "..."
            matched_chunk = f"{num_pre} | {heading}\n{chunk_txt}"
            matched_chunks.append(matched_chunk)
            last_chunk_end = chunk_end
        all_matching_chunks.extend(matched_chunks)

    if len(all_matching_chunks) == 0:
        return "No more matches found."
    
    new_patterns_cursor = [[p, c] for p, c in patterns_cursor] + [[pattern, 0]]
    return Command(
        update={
            "messages": [ToolMessage(content=f"{all_matching_chunks[0]}", tool_call_id=runtime.tool_call_id)],
            "patterns_matches": [[pattern, all_matching_chunks]], # use reducer
            "patterns_cursor": new_patterns_cursor,
        }
    )


def run_agent_per_query(dataset, qinfo, model, sht_type):
    query = qinfo["query"]
    toc_numbered : str = get_toc_numbered_clean(dataset, qinfo["file_name"], sht_type)
    toc_textspan: dict = get_toc_textspan_clean(dataset, qinfo["file_name"], sht_type)
    # doc_txt = get_doc_txt(dataset, qinfo["file_name"])
    toc_greptext: dict = get_toc_greptext_clean(dataset, qinfo["file_name"], sht_type)

    if DEBUG == True:
        print(get_agent_system_message(dataset).content)

    tools = [read_section, grep_search_next]
    agent = create_agent(
        model=create_model(model),
        tools=tools,
        middleware=[
            DisableParallelToolCallsMiddleware(),
            ModelCallLimitMiddleware(thread_limit=100, exit_behavior="error"),
            DocAgentModelCallTimestampMiddleware(),
            ModelRetryMiddleware(on_failure='error'),
            DocAgentToolArtifactMiddleware(),
            ToolRetryMiddleware(max_retries=0, on_failure='continue'),
        ],
        system_prompt=get_agent_system_message(dataset),
        # state_schema=AgentState,
        state_schema=GrepSearchNextState,
        # context_schema=DocAgentContext,
        context_schema=DocChunkAgentContext,
        checkpointer=InMemorySaver(),
    )

    runnable_config = {"configurable": {"thread_id": "1"}, "recursion_limit": 10000} 

    answer_path = get_result_path(dataset, model, AGENT_NAME, sht_type)
    msg_path = Path(str(answer_path).replace("/core/", "/other/messages/")).parent / dataset / f"query{qinfo['id']}.txt"
    os.makedirs(msg_path.parent, exist_ok=True)
    checkpoint_path = Path(str(msg_path).replace("/messages/", "/checkpoints/").replace(".txt", ".pkl"))
    os.makedirs(checkpoint_path.parent, exist_ok=True)
    
    prefix = "QUERY"
    if dataset == "contract":
        prefix = "HYPOTHESIS"
    start_time = time.time()
    try:
        final_state: AgentState = agent.invoke(
            # input=AgentState(messages=[HumanMessage(content=f"{prefix}:\n{query}\n\nTABLE OF CONTENTS (TOC):\n{toc_numbered}")]),
            input=GrepSearchNextState(messages=[HumanMessage(content=f"{prefix}:\n{query}\n\nTABLE OF CONTENTS (TOC):\n{toc_numbered}")], patterns_matches=[], patterns_cursor=[]),
            context=DocChunkAgentContext(toc_textgrep=toc_greptext, toc_textspan=toc_textspan),
            config=runnable_config,
        )
        end_time = time.time()
        llm_response = messages_to_response(
            messages=final_state['messages'],
            is_success=True,
            latency=end_time - start_time,
            err_msg=None
        )
        llm_response['id'] = qinfo['id']
        
        final_messages = final_state['messages']
        

    except Exception as e:
        end_time = time.time()
        err_msg = f"{type(e).__name__}: {str(e)}"
        logging.warning(f"Agent execution error: {type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}")

        llm_response = messages_to_response(
            messages=None,
            is_success=False,
            latency=end_time - start_time,
            err_msg=err_msg
        )
        llm_response['id'] = qinfo['id']

        try:
            final_messages = list(agent.get_state_history(config=runnable_config))[0].values['messages']
        except Exception as e:
            final_messages = []
        
    finally:
        with open(answer_path, 'a') as file:
            contents = json.dumps(llm_response) + "\n"
            file.write(contents)

        formatted_msg = ""
        for msg in final_messages:
            msg_str = msg.pretty_repr()
            if len(msg_str) <= 10000:
                formatted_msg += msg_str.strip() + "\n\n"
            else:
                formatted_msg += msg_str[:10000].strip() + "...(truncated)\n\n"
        with open(msg_path, 'w') as file:
            file.write(formatted_msg.strip())

        with open(checkpoint_path, 'wb') as file:
            pickle.dump(list(agent.get_state_history(config=runnable_config)), file)

        
if __name__ == "__main__":
    for sht_type in [
        # 'deep',
        # 'wide',
        # 'grobid',
        # '',
        # 'llm_txt_sht',
        'intrinsic',
    ]:
        for model in [
            "gpt-5.4", 
            # "gpt-5-mini"
        ]:
            for dataset in DATASET_LIST[3:]:
            # for dataset in ['office']:
                # if sht_type == 'grobid' and dataset in ['civic', 'contract']:
                #     continue

                print(f"{AGENT_NAME} ({model}, {sht_type}): {dataset}")
                queries_path = Path(DATA_ROOT_FOLDER) / dataset / "queries.json"
                with open(queries_path, 'r') as file:
                    queries = json.load(file)
                
                result_jsonl_path = get_result_path(dataset, model, AGENT_NAME, sht_type)

                if dataset == "civic_rand_v1":
                    start_id = 0
                    end_id = len(queries)
                elif dataset == 'finance_rand_v1':
                    start_id = 74
                    end_id = 100
                elif dataset == 'contract_rand_v0_1':
                    start_id = 0
                    end_id = 248
                elif dataset == 'qasper_rand_v1':
                    start_id = 290
                    end_id = 500

                for qinfo in queries[start_id:end_id]:
                    run_agent_per_query(dataset, qinfo, model, sht_type)