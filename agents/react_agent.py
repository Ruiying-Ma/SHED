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
from agents.utils import get_toc_system_message, get_system_message, get_timestamp, get_toc_textspan, get_toc_numbered, get_result_path, LLMResponse

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ModelCallLimitMiddleware, ToolRetryMiddleware, AgentMiddleware, AgentState 
from langchain.messages import ToolMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool, ToolRuntime
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain.chat_models import init_chat_model
from langgraph.types import Command

DEBUG = False

class DocAgentContext(TypedDict):
    toc_textspan: dict


class ToolArtifact(TypedDict):
    start_timestamp: str
    end_timestamp: str

class DocAgentModelCallTimestampMiddleware(AgentMiddleware):
    """Capture model call artifacts such as start timestamp and end timestamp. This wrapper should be applied outside the ModelRetryMiddleware."""
    def wrap_model_call(self, request, handler):
        start_ts = get_timestamp()
        response = handler(request)
        end_ts = get_timestamp()
        for msg in response.result:
            assert isinstance(msg, AIMessage)
            msg.response_metadata['start_timestamp'] = start_ts
            msg.response_metadata['end_timestamp'] = end_ts
        return response

class DocAgentToolArtifactMiddleware(AgentMiddleware):
    """Capture tool call artifacts such as caller, start timestamp, and end timestamp. This wrapper should be applied outside the ToolRetryMiddleware."""
    def wrap_tool_call(self, request, handler):
        start_ts = get_timestamp()
        response = handler(request)
        end_ts = get_timestamp()
        artifact = ToolArtifact(
            start_timestamp=start_ts,
            end_timestamp=end_ts
        )
        if isinstance(response, ToolMessage):
            response.artifact = artifact
        elif isinstance(response, Command):
            msgs = response.update.get("messages", [])
            if len(msgs) > 0:
                for msg in msgs:
                    if isinstance(msg, ToolMessage):
                        msg.artifact = artifact
        return response

def get_agent_system_message(dataset):
    system_msg = get_system_message(dataset).strip() + "\n\n"+ get_toc_system_message(verbose=False).strip() + "\n\n"

    system_msg += "Initially, you are only provided with:\n" \
    "- The query\n" \
    "- The table of contents of the document\n\n" \
    "You can use the tool `read_section(section_id)` to retrieve the content of a specific section in the document. The `section_id` corresponds to the number prefix in the TOC (the part before `|`).\n" \
    "Use the TOC to identify relevant sections, and call `read_section` **as needed** to retrieve their content. Then, answer the query following the instructions above.\n\n" \
    "RULES:\n" \
    "- Use tool calls to solve the task.\n" \
    "- Do not output any plain-text message until you are ready to provide the final answer.\n" \
    "- Any plain-text output will be treated as the final answer and will immediately terminate execution.\n" \
    "- Before the final answer, you may only produce tool calls.\n" \
    "- When you have obtained enough information, output a single plain-text answer with no additional text.\n" \
    "- Use tool calls **only when necessary**. If the query can be answered using the TOC alone, return the answer directly without calling any tools."

    return SystemMessage(content=system_msg.strip())

read_section_schema = {
    "type": "object",
    "properties": {
        "section_id": {"type": "string", "description": """The number prefix in the TOC (the part before `|`) that identifies the section to read. For example, if the TOC entry is `2.1 | 1.1 Subsection A1`, the `section_id` would be `2.1`."""},
    },
    "required": ["section_id"],
}
@tool(args_schema=read_section_schema)
def read_section(section_id: str, runtime: ToolRuntime) -> str:
    """Given the `section_id`, return the content of the corresponding section in the document."""
    toc_textspan = runtime.context['toc_textspan']
    if section_id not in toc_textspan:
        raise RuntimeError(f"Section ID {section_id} not found in TOC.")
    section_content = toc_textspan[section_id]
    return section_content.strip()



def messages_to_response(messages, is_success, latency, err_msg):
    if is_success == False:
        final_answer = err_msg
    else:
        try:
            final_answer = messages[-1].content
        except Exception as e:
            final_answer = ""
            logging.warning(f"Error extracting final answer from messages: {type(e).__name__}: {str(e)}")
        
    llm_response = LLMResponse(
        is_success=is_success,
        message=final_answer,
        latency=latency,
        input_tokens=0,  # to be updated below
        cached_tokens=0,   # to be updated below
        output_tokens=0,   # to be updated below
    )
    if is_success == True:
        for msg in messages:
            if isinstance(msg, AIMessage):
                token_usage = msg.response_metadata.get('token_usage', {})
                input_tokens = token_usage.get('prompt_tokens', 0)
                cached_tokens = token_usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
                input_tokens = input_tokens - cached_tokens
                output_tokens = token_usage.get('completion_tokens', 0)

                llm_response["input_tokens"] += input_tokens
                llm_response['cached_tokens'] += cached_tokens
                llm_response['output_tokens'] += output_tokens

    return llm_response

def create_model(model_name):
    load_dotenv()
    if "gpt" in model_name:
        model_provider = "azure_openai"
    else:
        raise NotImplementedError(f"Unsupported model provider for model {model_name}")
    if "azure" in model_provider.lower():
        model = init_chat_model(
            model=model_name.strip().lower(),
            model_provider=model_provider.strip().lower(),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            openai_api_version=os.getenv("AZURE_API_VERSION"),
            api_key=os.getenv("AZURE_API_KEY"),
            timeout=60,
        )
    else:
        raise NotImplementedError(f"Model provider {model_provider} is not supported.")
    return model

def run_agent_per_query(dataset, qinfo, model):
    query = qinfo["query"]
    toc_numbered : str = get_toc_numbered(dataset, qinfo["file_name"])
    toc_textspan = get_toc_textspan(dataset, qinfo["file_name"])
    assert len(toc_numbered.strip().splitlines()) == len(toc_textspan)

    if DEBUG == True:
        print(get_agent_system_message(dataset).content)

    agent = create_agent(
        model=create_model(model),
        tools=[read_section],
        middleware=[
            ModelCallLimitMiddleware(thread_limit=len(toc_textspan), exit_behavior="error"),
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

    answer_path = get_result_path(dataset, model, "react_agent")
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
    for model in ['gpt-5.4', 'gpt-5-mini']:
        for dataset in DATASET_LIST[3:4]:
            print(f"ReAct DocAgent: {dataset}")
            queries_path = Path(DATA_ROOT_FOLDER) / dataset / "queries.json"
            with open(queries_path, 'r') as file:
                queries = json.load(file)
            
            num_queries = int(len(queries) * 0.2)

            result_jsonl_path = get_result_path(dataset, model, "react_agent")

            if dataset == "civic":
                start_id = 1
                end_id = len(queries)
            elif dataset == "finance":
                start_id = 30
                end_id = 74
            else:
                start_id = 0
                end_id = num_queries

            for qinfo in queries[start_id:end_id]:
                print(f"\tquery id: {qinfo['id']}")
                run_agent_per_query(dataset, qinfo, model)


    # print(get_agent_system_message("qasper").content)