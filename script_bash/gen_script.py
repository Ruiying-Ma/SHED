import os

# #### Gen scripts for tmux_graphrag_answer
# COMMAND_TMP = '''tmux new-session -d -s "graphrag_answer_{i}"
# tmux send-keys "cd /home/ruiying/SHTRAG/graphrag" C-m
# tmux send-keys "source ./miniconda3/bin/activate" C-m
# tmux send-keys "conda activate graphrag" C-m
# tmux send-keys "python test_{i}.py --terminal_id {i} > test_{i}.log 2> test_{i}.error" C-m'''

# #### Gen scripts for tmux_graphrag_build
# COMMAND_TMP = '''tmux new-session -d -s "graphrag_build_{i}"
# tmux send-keys "cd /home/ruiying/SHTRAG/graphrag" C-m
# tmux send-keys "source ./miniconda3/bin/activate" C-m
# tmux send-keys "conda activate graphrag" C-m
# tmux send-keys "python test{i}.py --terminal_id {i} > test{i}.log 2> test{i}.error" C-m'''

# command = ""
# for i in range(14):
#     id = i + 1
#     command += COMMAND_TMP.format(i=id) + "\n\n"

# with open("/home/ruiying/SHTRAG/script_bash/tmux_graphrag_build.sh", "w") as f:
#     f.write(command)


# #### Gen scripts for tmux_hipprag_index
# m_dataset_ntmux = {
#     "civic": 1,
#     "contract": 2,
#     "qasper": 13,
# }
# m_dataset_docnum = {
#     "civic": 19,
#     "contract": 73,
#     "qasper": 416
# }

# COMMAND_TMP = '''tmux new-session -d -s "hipporag_index_{ds}_{i}"
# tmux send-keys "cd /home/ruiying/HippoRAG" C-m
# tmux send-keys "source ./miniconda3/bin/activate" C-m
# tmux send-keys "conda activate hipporag" C-m
# tmux send-keys "python test_{ds}_{i}.py --dataset {ds} --start_doc_id {sid} --end_doc_id {eid} > test_{ds}_{i}.log 2> test_{ds}_{i}.error" C-m'''


# command = ""
# for dataset, ntmux in m_dataset_ntmux.items():
#     docnum = m_dataset_docnum[dataset]
#     doc_num_per_tmux_list = [0 for _ in range(ntmux)]
#     for i in range(docnum):
#         doc_num_per_tmux_list[(i % ntmux)] += 1

#     assert sum(doc_num_per_tmux_list) == docnum

#     for tmux_id, doc_num in enumerate(doc_num_per_tmux_list):
#         sid = sum(doc_num_per_tmux_list[:tmux_id])
#         eid = sid + doc_num_per_tmux_list[tmux_id]
#         command += COMMAND_TMP.format(
#             i=tmux_id,
#             ds=dataset,
#             sid=sid,
#             eid=eid,
#         ) + "\n\n"

# with open("/home/ruiying/SHTRAG/script_bash/tmux_hipporag_index.sh", "w") as f:
#     f.write(command)



#### Gen scripts for HippoRAG/text.py
m_dataset_ntmux = {
    "civic": 1,
    "contract": 2,
    "qasper": 13,
}
m_dataset_docnum = {
    "civic": 19,
    "contract": 73,
    "qasper": 416
}

COMMAND_TMP = '''cp -p ~/HippoRAG/test.py ~/HippoRAG/test_{ds}_{i}.py'''


command = ""
for dataset, ntmux in m_dataset_ntmux.items():
    docnum = m_dataset_docnum[dataset]
    doc_num_per_tmux_list = [0 for _ in range(ntmux)]
    for i in range(docnum):
        doc_num_per_tmux_list[(i % ntmux)] += 1

    assert sum(doc_num_per_tmux_list) == docnum

    for tmux_id, doc_num in enumerate(doc_num_per_tmux_list):
        sid = sum(doc_num_per_tmux_list[:tmux_id])
        eid = sid + doc_num_per_tmux_list[tmux_id]
        command += COMMAND_TMP.format(
            i=tmux_id,
            ds=dataset,
        ) + "\n\n"

with open("/home/ruiying/SHTRAG/script_bash/copy_file.sh", "w") as f:
    f.write(command)