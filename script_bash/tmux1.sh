tmux new-session -d -s "raptor-build-finance"
tmux send-keys "cd /home/ruiying/SHTRAG" C-m
tmux send-keys "source ./miniconda3/bin/activate" C-m
tmux send-keys "conda activate shtrag" C-m
tmux send-keys "python run_raptor.py > run_raptor.log 2> run_raptor.error" C-m