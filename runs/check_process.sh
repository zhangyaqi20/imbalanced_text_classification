# #!/usr/bin/env bash
# pid=2162624

# if [ "$#" -gt 0 ]; then
#     # At least 1 argument was passed in, so assume it is the PID
#     pid="$1"
# fi

# # Try to print the process (`ps`) information for this PID. Send it to
# # /dev/null, however, so we don't actually have to look at it. We just want
# # the return code, `$?`, which will be 0 if the process exists and some other
# # number if not.
# ps --pid "$pid" > /dev/null
# # shellcheck disable=SC2181
# if [ "$?" -eq 0 ]; then
#     echo "PID $pid exists and is running."
# else
#     echo "PID $pid does NOT exist."
# fi

# Move files
for file in $(cat /mounts/Users/cisintern/zhangyaq/imbalanced_text_classification/runs/files.txt); do 
  mv /mounts/Users/cisintern/zhangyaq/imbalanced_text_classification/mlruns/24/"$file" /mounts/data/proj/zhangyaq/logs/civil-comments-5k-7p5; done

# # Delete files
# for f in $(cat /mounts/Users/cisintern/zhangyaq/imbalanced_text_classification/files.txt) ; do 
#   rm -r /mounts/Users/cisintern/zhangyaq/imbalanced_text_classification/mlruns/4/"$f"
# done