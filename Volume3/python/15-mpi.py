[style=ShellInput]
$ mpirun -n 5 python hello.py
[style=ShellOutput]
Hello world! I'm process number 3.
Hello world! I'm process number 2.
Hello world! I'm process number 0.
Hello world! I'm process number 4.
Hello world! I'm process number 1.

from sys import argv

# Pass in the first command line argument as n
n = int(argv[1])

mpirun -n 2 python passVector.py 3
