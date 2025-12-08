#!/bin/bash
# Monitor hardware (hi) and software (si) interrupt time per CPU
# Usage: ./monitor_interrupts.sh [interval_seconds] [cpu_filter]
# Example: ./monitor_interrupts.sh 1 "0 1 2"  # Monitor CPUs 0,1,2 every 1 second

INTERVAL=${1:-1}
CPU_FILTER=${2:-}

# Parse /proc/stat line and extract interrupt percentages
parse_cpu_line() {
    local line=$1
    local cpu_name=$(echo $line | awk '{print $1}')

    # Fields: user nice system idle iowait irq softirq steal guest guest_nice
    local user=$(echo $line | awk '{print $2}')
    local nice=$(echo $line | awk '{print $3}')
    local system=$(echo $line | awk '{print $4}')
    local idle=$(echo $line | awk '{print $5}')
    local iowait=$(echo $line | awk '{print $6}')
    local irq=$(echo $line | awk '{print $7}')
    local softirq=$(echo $line | awk '{print $8}')
    local steal=$(echo $line | awk '{print $9}')
    local guest=$(echo $line | awk '{print $10}')
    local guest_nice=$(echo $line | awk '{print $11}')

    # Calculate total time
    local total=$((user + nice + system + idle + iowait + irq + softirq + steal))

    echo "$cpu_name $irq $softirq $total"
}

# Read initial state
declare -A prev_irq prev_softirq prev_total

echo "Reading initial CPU state..."
while read -r line; do
    if [[ $line =~ ^cpu[0-9]+ ]]; then
        read cpu_name irq softirq total <<< $(parse_cpu_line "$line")
        cpu_num=${cpu_name#cpu}

        # Skip if CPU filter is specified and this CPU is not in it
        if [ -n "$CPU_FILTER" ] && [[ ! " $CPU_FILTER " =~ " $cpu_num " ]]; then
            continue
        fi

        prev_irq[$cpu_name]=$irq
        prev_softirq[$cpu_name]=$softirq
        prev_total[$cpu_name]=$total
    fi
done < /proc/stat

sleep "$INTERVAL"

echo ""
echo "Monitoring interrupt time (update every ${INTERVAL}s, Ctrl+C to stop)"
echo "Format: CPU  %HI (hardware)  %SI (software)  Total%"
echo "--------------------------------------------------------"

while true; do
    timestamp=$(date '+%H:%M:%S')
    echo ""
    echo "=== $timestamp ==="
    printf "%-8s %8s %8s %8s\n" "CPU" "%HI" "%SI" "Total%"

    while read -r line; do
        if [[ $line =~ ^cpu[0-9]+ ]]; then
            read cpu_name irq softirq total <<< $(parse_cpu_line "$line")
            cpu_num=${cpu_name#cpu}

            # Skip if CPU filter is specified and this CPU is not in it
            if [ -n "$CPU_FILTER" ] && [[ ! " $CPU_FILTER " =~ " $cpu_num " ]]; then
                continue
            fi

            # Calculate deltas
            delta_irq=$((irq - prev_irq[$cpu_name]))
            delta_softirq=$((softirq - prev_softirq[$cpu_name]))
            delta_total=$((total - prev_total[$cpu_name]))

            # Calculate percentages (avoid division by zero)
            if [ $delta_total -gt 0 ]; then
                pct_irq=$(awk "BEGIN {printf \"%.2f\", ($delta_irq / $delta_total) * 100}")
                pct_softirq=$(awk "BEGIN {printf \"%.2f\", ($delta_softirq / $delta_total) * 100}")
                pct_both=$(awk "BEGIN {printf \"%.2f\", (($delta_irq + $delta_softirq) / $delta_total) * 100}")
            else
                pct_irq="0.00"
                pct_softirq="0.00"
                pct_both="0.00"
            fi

            printf "%-8s %7s%% %7s%% %7s%%\n" "$cpu_name" "$pct_irq" "$pct_softirq" "$pct_both"

            # Update previous values
            prev_irq[$cpu_name]=$irq
            prev_softirq[$cpu_name]=$softirq
            prev_total[$cpu_name]=$total
        fi
    done < /proc/stat

    sleep "$INTERVAL"
done
