tmux new-session -d -s "finance"
tmux send-keys "cd /home/ruiying/SHTRAG" C-m
tmux send-keys "source ./miniconda3/bin/activate" C-m
tmux send-keys "conda activate shtrag" C-m
tmux send-keys "python run_structured_rag.py > run_structured_rag.log 2> run_structured_rag.error" C-m
tmux send-keys "python run_raptor.py > run_raptor.log 2> run_raptor.error" C-m
tmux send-keys "python run_vanilla.py > run_vanilla.log 2> run_vanilla.error" C-m
tmux send-keys "python run_grobid.py > run_grobid.log 2> run_grobid.error" C-m
tmux send-keys "python run_bm25.py > run_bm25.log 2> run_bm25.error" C-m