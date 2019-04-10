#!/bin/bash

while true; do
    find comics -type f | wc -l
    du -sh comics
    sleep 0.5

    tput cuu1 # move cursor up by one line
    tput el # clear the line
    tput cuu1
    tput el
done