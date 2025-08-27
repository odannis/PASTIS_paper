#!/bin/bash

# Function to display help
Help() {
    echo "Syntax: script.sh [-h|-n|-s|-q]"
    echo "options:"
    echo "h     Print this Help."
    echo "n     Specify the number of CPUs to use."
    echo "s     Name of the script."
    echo "q     Queue type (ib6.q, ib5.q, or ib4.q)."
}

# Initialize variables
nump_cpu_use=1  # default value for number of CPUs
name_script=""  # default value for script name
queue_type="ib4.q"  # default value for queue type
pe_qlogic="12"  # default value for -pe qlogic
number_simu=100  # default value for number of simulations

# Process command-line options
while getopts ":hn:s:q:d:" option; do
    case $option in
        h) # display Help
            Help
            exit;;
        n) # Enter a number
            nump_cpu_use=$OPTARG;;
        s) # Enter a script name
            name_script=$OPTARG;;
        q) # Enter a queue type
            queue_type=$OPTARG;;
        d) # Enter a number
            number_simu=$OPTARG;;
        \?) # Invalid option
            echo "Error: Invalid option"
            exit;;
    esac
done

# Determine the appropriate -pe qlogic value based on the queue type
case $queue_type in
    ib8.q)
        pe_qlogic=32;;
    ib7.q)
        pe_qlogic=24;;
    ib6.q)
        pe_qlogic=24;;
    ib5.q)
        pe_qlogic=16;;
    ib4.q)
        pe_qlogic=12;;
    ib3.q)
        pe_qlogic=1
        nump_cpu_use=1;;
    *)
        echo "Error: Invalid queue type. Choose from ib6.q, ib5.q, or ib4.q."
        exit 1
esac

number_script=$((number_simu*$nump_cpu_use/$pe_qlogic))
number_script=$(((number_simu*nump_cpu_use+$pe_qlogic-1)/$pe_qlogic))

# Rest of your script...
echo "Number of CPUs to use: $nump_cpu_use"
echo "Name of the script: $name_script"
echo "Queue type: $queue_type"
echo "Parallel environment setting: -pe qlogic $pe_qlogic"
echo "Number script launch : $number_script"

# Loop to submit jobs
for ((i = 0 ; i < $number_script ; i++)); do
    if [[ $queue_type == "ib3.q" ]]; then
        qsub -N ${name_script} -q all.q script_all_q.sh ${i} ${name_script} 
        #qsub -N ${name_script} -q all.q script_ib_q.sh ${i} ${name_script} ${nump_cpu_use}
    else
        qsub -N ${name_script} -q $queue_type -pe qlogic $pe_qlogic script_ib_q.sh ${i} ${name_script} ${nump_cpu_use}
    fi
done
