import subprocess
from datetime import datetime as dt


def callSPMF(spmfPath, command):
    # java -jar spmf.jar run PrefixSpan contextPrefixSpan.txt output.txt 50%
    comm = ' '.join(['java -jar', spmfPath, 'run', command])
    print(comm)
    p = subprocess.Popen(comm,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
    std_out, std_err = p.communicate()  # wait for completion
    print("std out: ", std_out)
    print("std err: ", std_err)
