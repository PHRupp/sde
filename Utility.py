
import numpy as np
import re
import subprocess

#pmap 4814 | tail -n 1 | awk '/[0-9]K/{print $2}'
def get_pid_memory_usage( pid: str ):

    cmd = ['pmap', str(pid)]

    out = subprocess.check_output(cmd)

    my_list = out.split(b'\n')

    size_str = str(my_list[-2])

    size_reg = r'\s*total\s*(\d+)K\s*'

    match = re.search( size_reg, size_str )

    # convert kilobytes to bytes
    size_bytes = int( match.group(1) ) * 1000 if match else None
    
    return size_bytes


