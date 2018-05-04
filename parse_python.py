#!/usr/bin/env python
# File to parse the python environments from Jeff's latex files
import re
import sys

# error check to ensure that there are two arguments, and that the infile exists
infile = str(sys.argv[1])
outfile = str(sys.argv[2])

words = open(infile).read()
results = re.findall(r'\\begin{lstlisting}(.*?)\\end{lstlisting}', words, re.S)
output = open(outfile, 'w')
print ("Writing python environments to %s" % outfile)
for line in results:
	output.write(line)
