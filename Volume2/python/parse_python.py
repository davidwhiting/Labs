import re
data = open('Wavelets.py').read()
results = re.findall(r'\\begin{lstlisting}(.*?)\\end{lstlisting}', data, re.S)
output = open('Wavelets_new.py', 'w')
for line in results:
	output.write(line)
