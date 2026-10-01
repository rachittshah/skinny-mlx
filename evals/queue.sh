#!/bin/zsh
# Run hill-climb configs once the machine is free: no lohia_qc / Blender batch jobs and an idle GPU
# for 60 s. Each run still gates every measured pair on idle windows (evals/run.py).
cd "$(dirname "$0")/.."
MODEL=${MODEL:-4b}
CONFIGS=(${=CONFIGS:-mlx_greedy dflash dflash_dq4 dflash_b8})
quiet=0
until [ $quiet -ge 60 ]; do
  busy=0
  pgrep -f "lohia_qc" >/dev/null && busy=1
  pgrep -f "Blender -b" >/dev/null && busy=1
  u=$(ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2)
  [ "${u:-0}" -gt 5 ] && busy=1
  if [ $busy -eq 0 ]; then quiet=$((quiet+5)); else quiet=0; fi
  sleep 5
done
echo "machine free at $(date)"
for c in $CONFIGS; do
  echo "=== $c"
  .venv/bin/python -u -m evals.run --model $MODEL --config $c 2>&1 | grep -vE "Fetching|Warn|it/s\]"
done
.venv/bin/python -m evals.run --model $MODEL --summary
