#!/bin/bash

killall php
make
php -S localhost:8080 -t web/ &

while [ true ]
do
    inotifywait -e modify -r pages/
    make
done