#!/bin/bash
# poll.sh <max_minutes> <kernels...> ; exits when any listed kernel is no longer running
end=$(( $(date +%s) + $1*60 )); shift
while [ $(date +%s) -lt $end ]; do
  line=""; done_any=0
  for k in "$@"; do s=$(python3 kjob.py status $k | grep -o "'status': '[a-z]*'" | cut -d"'" -f4); line="$line $k:$s"; [ "$s" != "running" ] && [ "$s" != "queued" ] && done_any=1; done
  echo "$(date +%H:%M) $line"; [ $done_any = 1 ] && exit 0; sleep 60
done
