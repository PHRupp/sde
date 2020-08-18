# Run python program
python main.py &

# get the process id for the python program
export main_pid="$!"

# define the output file
export outfile='mem_time_out.csv'

# delete the file to start clean
rm -f ${outfile}
touch ${outfile}

# keep grabbing the memory usage for the program until it stops running
while kill -0 ${main_pid} >/dev/null 2>&1
do
    mem_k=$(pmap ${main_pid} | tail -n 1 | awk '/[0-9]K/{print $2}')
    unix_nanosec=$(date +"%s%N")
    
    # if the memory was successfully retrived write it to file
    if [ ! -z "${mem_k}" ]
    then
        echo "${unix_nanosec},${mem_k}" >> ${outfile}
    fi
done
