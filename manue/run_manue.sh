#!/bin/bash
GAME_HOST=`cat /etc/hosts | awk 'END{print $1}' | sed -r -e 's/[0-9]+$/1/g'`
mjai-manue --name ${1} mjsonp://${GAME_HOST}:48000/${2} 