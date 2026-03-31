import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import json
import logging
import logging_config
from typing import TypedDict
import time
import pickle
from dotenv import load_dotenv
import traceback
import re

from config import DATASET_LIST, DATA_ROOT_FOLDER, get_max_window
from agents.utils import get_toc_system_message, get_system_message, get_timestamp, get_toc_textspan, get_toc_numbered, get_result_path, LLMResponse, grep_search

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ModelCallLimitMiddleware, ToolRetryMiddleware, AgentMiddleware, AgentState 
from langchain.messages import ToolMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool, ToolRuntime
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain.chat_models import init_chat_model

from agents.react_agent import(
    create_model,
    read_section,
    DocAgentModelCallTimestampMiddleware,
    DocAgentToolArtifactMiddleware,
    DocAgentContext,
    messages_to_response,
)
from agents.accuracy import INVALID_CIVIC_QUERY_IDS

DEBUG = False

AGENT_NAME = "react_agent_grep_all"

def get_agent_system_message(dataset):
    system_msg = get_system_message(dataset).strip() + "\n\n"+ get_toc_system_message(verbose=False).strip() + "\n\n"

    system_msg += "Initially, you are only provided with:\n" \
    "- The query\n" \
    "- The table of contents of the document\n\n" \
    "You are provided with the following tools to retrieve more information from the document:\n" \
    "- You can use the tool `read_section(section_id)` to retrieve the content of a specific section in the document. The `section_id` corresponds to the number prefix in the TOC (the part before `|`). Use the TOC to identify relevant sections, and call `read_section` **as needed** to retrieve their content.\n" \
    "- You can use the tool `grep_search_all(pattern)` to locate relevant content in the document using regular expression matching. The tool searches the entire document and returns all minimum matching sections, where each result includes the section's number prefix, title, and full content. A minimum matching section is defined as a section that contains at least one match to the pattern, while none of its subsections (as defined by the document's table of contents) contain any matches. Use this tool when you want to find relevant sections that contain specific keywords or phrases, especially when you are not sure which sections might be relevant based on the TOC alone.\n\n" \
    "RULES:\n" \
    "- Use tool calls to solve the task.\n" \
    "- Do not output any plain-text message until you are ready to provide the final answer.\n" \
    "- Any plain-text output will be treated as the final answer and will immediately terminate execution.\n" \
    "- Before the final answer, you may only produce tool calls.\n" \
    "- When you have obtained enough information, output a single plain-text answer with no additional text.\n" \
    "- Use tool calls **only when necessary**. If the query can be answered using the TOC alone, return the answer directly without calling any tools."

    return SystemMessage(content=system_msg.strip())

grep_search_all_schema = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": """A regular expression used to search for matches within the document contents."""},
    },
    "required": ["pattern"],
}
@tool(args_schema=grep_search_all_schema)
def grep_search_all(
    pattern: str,
    runtime: ToolRuntime,
) -> str:
    """Searches the document using full regular expression syntax and returns all minimum matching sections, each including its number prefix, section title, and full content. A minimum matching section is one that contains at least one match to the pattern, while none of its subsections contain any matches."""
    try:
        re.compile(pattern)
    except re.error as e:
        return f"Invalid regex pattern: {e}"
    
    toc_textspan = runtime.context['toc_textspan']
    matched_section_ids = grep_search(pattern, toc_textspan)

    if len(matched_section_ids) == 0:
        return "No matches found."

    results = []
    for section_id in matched_section_ids:
        section_content = toc_textspan[section_id]
        results.append(f"{section_id} | {section_content}")
    
    return "\n\n".join(results)


def run_agent_per_query(dataset, qinfo, model):
    query = qinfo["query"]
    toc_numbered : str = get_toc_numbered(dataset, qinfo["file_name"])
    toc_textspan = get_toc_textspan(dataset, qinfo["file_name"])
    assert len(toc_numbered.strip().splitlines()) == len(toc_textspan)

    if DEBUG == True:
        print(get_agent_system_message(dataset).content)

    tools = [read_section, grep_search_all]
    agent = create_agent(
        model=create_model(model),
        tools=tools,
        middleware=[
            ModelCallLimitMiddleware(thread_limit=100, exit_behavior="error"),
            DocAgentModelCallTimestampMiddleware(),
            ModelRetryMiddleware(on_failure='error'),
            DocAgentToolArtifactMiddleware(),
            ToolRetryMiddleware(max_retries=0, on_failure='continue'),
        ],
        system_prompt=get_agent_system_message(dataset),
        state_schema=AgentState,
        context_schema=DocAgentContext,
        checkpointer=InMemorySaver(),
    )

    runnable_config = {"configurable": {"thread_id": "1"}, "recursion_limit": 10000} 

    answer_path = get_result_path(dataset, model, AGENT_NAME)
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
            input=AgentState(messages=[HumanMessage(content=f"{prefix}:\n{query}\n\nTABLE OF CONTENTS (TOC):\n{toc_numbered}")]),
            context=DocAgentContext(toc_textspan=toc_textspan),
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

        final_messages = list(agent.get_state_history(config=runnable_config))[0].values['messages']
        
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

    for model in ["gpt-5.4", "gpt-5-mini"]:
        for dataset in DATASET_LIST[3:4]:
            print(f"{AGENT_NAME} ({model}): {dataset}")
            queries_path = Path(DATA_ROOT_FOLDER) / dataset / "queries.json"
            with open(queries_path, 'r') as file:
                queries = json.load(file)
            
            num_queries = int(len(queries) * 0.2)

            result_jsonl_path = get_result_path(dataset, model, AGENT_NAME)

            if dataset == "civic":
                start_id = 0
                end_id = len(queries)
            elif dataset == "contract":
                start_id = 6
                end_id = num_queries
            elif dataset == 'finance':
                start_id = 30
                end_id = 74
            else:
                start_id = 0
                end_id = num_queries

            for qinfo in queries[start_id:end_id]:
                if dataset == "civic" and qinfo['id'] in INVALID_CIVIC_QUERY_IDS:
                    print(f"\tquery id: {qinfo['id']}: SKIPPED")
                    continue
                print(f"\tquery id: {qinfo['id']}")
                run_agent_per_query(dataset, qinfo, model)