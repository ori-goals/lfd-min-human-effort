#!/bin/bash
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# cd into the src directory
cd "$parent_path"

# kill existing tmux sessions and the completed file



# loop through runs
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20;
do
	echo "Starting run $i"

	rm -f .run_completed.txt
	tmux kill-server

	sleep 15

	rosrun learn_to_manipulate learn_to_manipulate.tmux &

	sleep 200

	while [ -f .run_completed.txt ];
	do
		rm -f .run_completed.txt
  		sleep 100;
	done
	echo "No run completed file. Restarting everything."
done
