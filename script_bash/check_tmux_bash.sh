#!/bin/bash

# Get list of all tmux sessions
sessions=$(tmux list-sessions -F '#S' 2>/dev/null)

if [ -z "$sessions" ]; then
    echo "No tmux sessions found."
    exit 0
fi


for session in $sessions; do
    pane_pids=$(tmux list-panes -t "$session" -F '#{pane_pid}')

    for pid in $pane_pids; do
        # Get the full command line of the pane's main process
        parent_cmd=$(ps -p "$pid" -o args=)

        # Check for child processes of this PID
        child_cmds=$(pgrep -P "$pid" | xargs -r ps -o cmd= -p)

        if [ -z "$child_cmds" ]; then
            # echo "✅ Session $session (PID $pid): idle -> $parent_cmd"
            echo "✅ Session $session (PID $pid): idle"
        else
            # echo "⏳ Session $session (PID $pid): running -> "
            # echo "$child_cmds" | sed 's/^/     └── /'
            echo "⏳ Session $session (PID $pid): running -> $child_cmds"
            ((active_count++))
        fi
    done
done
