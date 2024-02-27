#!/usr/bin/env bash
pid=2162624

if [ "$#" -gt 0 ]; then
    # At least 1 argument was passed in, so assume it is the PID
    pid="$1"
fi

# Try to print the process (`ps`) information for this PID. Send it to
# /dev/null, however, so we don't actually have to look at it. We just want
# the return code, `$?`, which will be 0 if the process exists and some other
# number if not.
ps --pid "$pid" > /dev/null
# shellcheck disable=SC2181
if [ "$?" -eq 0 ]; then
    echo "PID $pid exists and is running."
else
    echo "PID $pid does NOT exist."
fi