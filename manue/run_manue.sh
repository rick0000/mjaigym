#!/bin/bash
GAME_HOST=`cat /etc/hosts | awk 'END{print $1}' | sed -r -e 's/[0-9]+$/1/g'`
mjai-manue mjsonp://${GAME_HOST}:48000/${1} ${2}