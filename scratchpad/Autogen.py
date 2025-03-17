from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen import AssistantAgent
import chromadb
import autogen

llm_config={
    "config_list": [
        {
            "model": "mistralai/Mistral-Small-24B-Instruct-2501", # Same as in vLLM command
            "api_key": "NotRequired", # Not needed
            "base_url": "http://<IP>:9150/v1"  # Your vLLM URL, with '/v1' added
        }
    ],
    "cache_seed": None # Turns off caching, useful for testing different models
}


# autogen.ChatCompletion.start_logging()
termination_msg = lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

boss = autogen.UserProxyAgent(
    name="Manager",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    system_message="Manager who assigns tasks to other agents.",
    code_execution_config=False,  # we don't want to execute code in this case.
)

insurance_expert = RetrieveUserProxyAgent(
    name="insurance_expert",
    is_termination_msg=termination_msg,
    system_message="Assistant who has extra content retrieval power for solving difficult problems.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    retrieve_config={
        "task": "code",
        "docs_path": "/home/vikasnr/codebase/crsl/pdfs/insurance.pdf",
        "chunk_token_size": 1500,
        "model": llm_config.get("config_list")[0]["model"],
        "client": chromadb.PersistentClient(path="insurance"),
        "collection_name": "groupchat",
        "get_or_create": True,
    },
    code_execution_config=False,  # we don't want to execute code in this case.
)

wildlife_expert = RetrieveUserProxyAgent(
    name="wildlife_expert",
    is_termination_msg=termination_msg,
    system_message="Assistant who has extra content retrieval power for solving difficult problems.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    retrieve_config={
        "task": "code",
        "docs_path": "/home/vikasnr/codebase/crsl/pdfs/wildlife.pdf",
        "chunk_token_size": 1500,
        "model": llm_config.get("config_list")[0]["model"],
        "client": chromadb.PersistentClient(path="wildlife"),
        "collection_name": "groupchat",
        "get_or_create": True,
    },
    code_execution_config=False,  # we don't want to execute code in this case.
)


river_expert = RetrieveUserProxyAgent(
    name="river_expert",
    is_termination_msg=termination_msg,
    system_message="Assistant who has extra content retrieval power for solving difficult problems.",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    retrieve_config={
        "task": "code",
        "docs_path": "/home/vikasnr/codebase/crsl/pdfs/river.pdf",
        "chunk_token_size": 1500,
        "model": llm_config.get("config_list")[0]["model"],
        "client": chromadb.PersistentClient(path="river"),
        "collection_name": "groupchat",
        "get_or_create": True,
    },
    code_execution_config=False,  # we don't want to execute code in this case.
)
# boss_aid = RetrieveUserProxyAgent(
#     name="Boss_Assistant",
#     is_termination_msg=termination_msg,
#     system_message="Assistant who has extra content retrieval power for solving difficult problems.",
#     human_input_mode="NEVER",
#     max_consecutive_auto_reply=10,
#     retrieve_config={
#         "task": "code",
#         "docs_path": "/home/vikasnr/codebase/crsl/pdfs/insurance.pdf",
#         "chunk_token_size": 1500,
#         "model": llm_config.get("config_list")[0]["model"],
#         "client": chromadb.PersistentClient(path="owais_experiments/chromadb"),
#         "collection_name": "groupchat",
#         "get_or_create": True,
#     },
#     code_execution_config=False,  # we don't want to execute code in this case.
# )

# owais_coder = AssistantAgent(
#     name="Insurance_expert",
#     is_termination_msg=termination_msg,
#     system_message="You are an expert in provided context. Only answer based on  Reply `TERMINATE` in the end when everything is done.",
#     llm_config=llm_config,
# )

# pm = autogen.AssistantAgent(
#     name="Product_Manager",
#     is_termination_msg=termination_msg,
#     system_message="You are a product manager. Reply `TERMINATE` in the end when everything is done.",
#     llm_config=llm_config,
# )

# reviewer = autogen.AssistantAgent(
#     name="Code_Reviewer",
#     is_termination_msg=termination_msg,
#     system_message="You are a code reviewer. Reply `TERMINATE` in the end when everything is done.",
#     llm_config=llm_config,
# )

PROBLEM = "What is the claim process for insurance?"

def _reset_agents():
    boss.reset()
    wildlife_expert.reset()
    insurance_expert.reset()
    river_expert.reset()
    # owais_coder.reset()
    # pm.reset()
    # reviewer.reset()

def rag_chat():
    _reset_agents()
    groupchat = autogen.GroupChat(
        agents=[wildlife_expert,insurance_expert,river_expert], messages=[], max_round=30
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # Start chatting with boss_aid as this is the user proxy agent.
    boss.initiate_chat(
        manager,
        problem=PROBLEM,
        n_results=5,)
    
    
    
rag_chat()