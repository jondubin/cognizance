#!/bin/bash

#
# Pull the latest code from the repository onto the system.
#

SCRIPT=""
while :
	sudo kill $(ps aux | grep '[p]ython csp_build.py' | awk '{print $2}')
	sudo git pull
	python $SCRIPT
	sleep 3
done