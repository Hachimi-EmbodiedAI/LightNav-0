#!/usr/bin/env bash
# Log VDD_IN power while a long-running test executes, so a hard power-off leaves
# the last reading on disk. On Jetson the regulator is INA3221 at i2c 1-0040.
OUT="${1:-/tmp/power.log}"
: > "$OUT"
while true; do
  v=$(cat /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/in1_input 2>/dev/null || echo 0)
  c=$(cat /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1/curr1_input 2>/dev/null || echo 0)
  t=$(cat /sys/devices/virtual/thermal/thermal_zone1/temp 2>/dev/null || echo 0)
  g=$(cat /sys/devices/platform/gpu.0/devfreq/*/cur_freq 2>/dev/null | head -1)
  printf "%s VDD_IN=%smV I=%smA P=%.1fW soc=%sC gpu=%sHz\n" \
    "$(date +%H:%M:%S)" "$v" "$c" "$(awk "BEGIN{print $v*$c/1000000}")" \
    "$((t/1000))" "$g" | tee -a "$OUT"
  sleep 0.5
done
