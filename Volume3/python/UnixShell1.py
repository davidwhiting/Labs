[language=bash]
$ cd Test
$ touch data.txt				# create new empty file data.txt
$ mkdir New						# create directory New
$ ls							# list items in test directory
New 	data.txt
$ cp data.txt New/				# copy data.txt to New directory
$ cd New/						# enter the New directory
$ ls							# list items in New directory
data.txt
$ mv data.txt new_data.txt		# rename data.txt new_data.txt
$ ls							# list items in New directory
new_data.txt
$ cd ..							# Return to test directory
$ rm -rv New/					# Remove New directory and its contents
removed 'New/data.txt'
removed directory: 'New/'
$ clear							# Clear terminal screen
[language=bash]
$ ls
File1.txt	File2.txt	File3.jpg	text_files
$ mv -v *.txt text_files/
File1.txt -> text_files/File1.txt
File2.txt -> text_files/File2.txt
$ ls
File3.jpg	text_files
[language=bash]
$ cd Shell1/Files/Feb
$ cat assignments.txt | wc -l
9

$ ls -s | sort -nr
12 project3.py
12 project2.py
12 assignments.txt
 4 pics
total 40
[language=bash]
$ wc -l < assignments.txt
9
[language=bash]
$ wc -l < assignements.txt >> word_count.txt
[language=bash]
$ cd Shell1/Documents
$ zip zipfile.zip doc?.txt
 adding: doc1.txt (deflated 87%)
 adding: doc2.txt (deflated 90%)
 adding: doc3.txt (deflated 85%)
 adding: doc4.txt (deflated 97%)

# use -l to view contents of zip file
$ unzip -l zipfile.zip
Archive:  zipfile.zip
  Length     Date     Time    Name
---------  ---------- -----   ----
     5234  2015-08-26 21:21   doc1.txt
     7213  2015-08-26 21:21   doc2.txt
     3634  2015-08-26 21:21   doc3.txt
     4516  2015-08-26 21:21   doc4.txt
---------                     -------
    16081                     3 files

$ unzip zipfile.zip
  inflating: doc1.txt
  inflating: doc2.txt
  inflating: doc3.txt
  inflating: doc4.txt
[language=bash]
$ ls
doc1.txt	doc2.txt	doc3.txt	doc4.txt

# use -c to create a new archive
$ tar -zcvf docs.tar.gz doc?.txt
doc1.txt
doc2.txt
doc3.txt
doc4.txt

$ ls
docs.tar.gz

# use -t to view contents
$ tar -ztvf <archive>
-rw-rw-r-- username/groupname 5119 2015-08-26 16:50 doc1.txt
-rw-rw-r-- username/groupname 7253 2015-08-26 16:50 doc2.txt
-rw-rw-r-- username/groupname 3524 2015-08-26 16:50 doc3.txt
-rw-rw-r-- username/groupname 4516 2015-08-26 16:50 doc4.txt

# use -x to extract
$ tar -zxvf <archive>
doc1.txt
doc2.txt
doc3.txt
doc4.txt
[language=bash]
$ vim my_file.txt
