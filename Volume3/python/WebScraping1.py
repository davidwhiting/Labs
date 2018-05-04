[language=HTML]
<html>                                  <!-- Opening tags -->
   <body>
       <p>
           Click <a id='info' href='http://www.example.com'>here</a>
           for more information.
       </p>                             <!-- Closing tags -->
   </body>
</html>
[language=HTML]
<html><body><p>Click <a id='info' href='http://www.example.com/info'>here</a>
  for more information.</p></body></html>

>>> from bs4 import BeautifulSoup

>>> small_example_html = """
<html><body><p>
    Click <a id='info' href='http://www.example.com'>here</a>
    for more information.
</p></body></html>
"""

>>> small_soup = BeautifulSoup(small_example_html, 'html.parser')
>>> print(small_soup.prettify())
<<<html>
 <body>
  <p>
   Click
   <a href="http://www.example.com" id="info">
    here
   </a>
   for more information.
  </p>
 </body>
</html>>>

# Get the <p> tag (and everything inside of it).
>>> small_soup.p
<<<p>
    Click <a href="http://www.example.com" id="info">here</a>
    for more information.
</p>>>

# Get the <a> sub-tag of the <p> tag.
>>> a_tag = small_soup.p.a
>>> print(a_tag, type(a_tag), sep='\n')
<<<a href="http://www.example.com" id="info">here</a>
<class 'bs4.element.Tag'>>>

# Get just the name, attributes, and text of the <a> tag.
>>> print(a_tag.name, a_tag.attrs, a_tag.string, sep="\n")
<<a
{'id': 'info', 'href': 'http://www.example.com'}
here>>

>>> pig_html = """
<html><head><title>Three Little Pigs</title></head>
<body>
<p class="title"><b>The Three Little Pigs</b></p>
<p class="story">Once upon a time, there were three little pigs named
<a href="http://example.com/larry" class="pig" id="link1">Larry,</a>
<a href="http://example.com/mo" class="pig" id="link2">Mo</a>, and
<a href="http://example.com/curly" class="pig" id="link3">Curly.</a>
<p>The three pigs had an odd fascination with experimental construction.</p>
<p>...</p>
</body></html>
"""

>>> pig_soup = BeautifulSoup(pig_html, "html.parser")
>>> pig_soup.p
<<<p class="title"><b>The Three Little Pigs</b></p>>>

>>> pig_soup.a
<<<a class="pig" href="http://example.com/larry" id="link1">Larry,</a>>>

>>> print(pig_soup.prettify())
<<<html>
 <head>>>                     # <head> is the parent of the <title>
  <<<title>
   Three Little Pigs
  </title>
 </head>
 <body>>>                     # <body> is the sibling of <head>
  <<<p class="title">>>         # and the parent of two <p> tags (title and story).
   <<<b>
    The Three Little Pigs
   </b>
  </p>
  <p class="story">
   Once upon a time, there were three little pigs named
   <a class="pig" href="http://example.com/larry" id="link1">
    Larry,
   </a>
   <a class="pig" href="http://example.com/mo" id="link2">
    Mo
   </a>
   , and
   <a class="pig" href="http://example.com/curly" id="link3">
    Curly.>>                  # The preceding <a> tags are siblings with each
   </a>                     # other and the following two <p> tags.
   <<<p>
    The three pigs had an odd fascination with experimental construction.
   </p>
   <p>
    ...
   </p>
  </p>
 </body>
</html>>>

# Start at the first <a> tag in the soup.
>>> a_tag = pig_soup.a
>>> a_tag
<<<a class="pig" href="http://example.com/larry" id="link1">Larry,</a>>>

# Get the names of all of <a>'s parent tags, traveling up to the top.
# The name '[document]' means it is the top of the HTML code.
>>> [par.name for par in a_tag.parents]     # <a>'s parent is <p>, whose
<<['p', 'body', 'html', '[document]']>>         # parent is <body>, and so on.

# Get the next siblings of <a>.
>>> a_tag.next_sibling
<<'\n'>>                                        # The first sibling is just text.
>>> a_tag.next_sibling.next_sibling         # The second sibling is a tag.
<<<a class="pig" href="http://example.com/mo" id="link2">Mo</a>>>

# Alternatively, get all siblings past <a> at once.
>>> list(a_tag.next_siblings)
<<['\n',
 <a class="pig" href="http://example.com/mo" id="link2">Mo</a>,
 ', and\n',
 <a class="pig" href="http://example.com/curly" id="link3">Curly.</a>,
 '\n',
 <p>The three pigs had an odd fascination with experimental construction.</p>,
 '\n',
 <p>...</p>,
 '\n']>>

# Get to the <p> tag that has class="story".
>>> p_tag = pig_soup.body.p.next_sibling.next_sibling
>>> p_tag.attrs["class"]                # Make sure it's the right tag.
<<['story']>>

# Iterate through the child tags of <p> and print hrefs whenever they exist.
>>> for child in p_tag.children:
...     if hasattr(child, "attrs") and "href" in child.attrs:
...         print(child.attrs["href"])
<<http://example.com/larry
http://example.com/mo
http://example.com/curly>>

>>> pig_soup.head
<<<head><title>Three Little Pigs</title></head>>>

# Case 1: the <title> tag's only child is a string.
>>> pig_soup.head.title.string
<<'Three Little Pigs'>>

# Case 2: The <head> tag's only child is the <title> tag.
>>> pig_soup.head.string
<<'Three Little Pigs'>>

# Case 3: the <body> tag has several children.
>>> pig_soup.body.string is None
<<True>>
>>> print(pig_soup.body.get_text().strip())
<<The Three Little Pigs
Once upon a time, there were three little pigs named
Larry,
Mo, and
Curly.
The three pigs had an odd fascination with experimental construction.
...>>

<<'More information...'>>

# Find the first <b> tag in the soup.
>>> pig_soup.find(name='b')
<<<b>The Three Little Pigs</b>>>

# Find all tags with a class attribute of 'pig'.
# Since 'class' is a Python keyword, use 'class_' as the argument.
>>> pig_soup.find_all(class_="pig")
<<[<a class="pig" href="http://example.com/larry" id="link1">Larry,</a>,
 <a class="pig" href="http://example.com/mo" id="link2">Mo</a>,
 <a class="pig" href="http://example.com/curly" id="link3">Curly.</a>]>>

# Find the first tag that matches several attributes.
>>> pig_soup.find(attrs={"class": "pig", "href": "http://example.com/mo"})
<<<a class="pig" href="http://example.com/mo" id="link2">Mo</a>>>

# Find the first tag whose text is 'Mo'.
>>> pig_soup.find(string='Mo')
<<'Mo'>>                                # The result is the actual string,
>>> soup.find(string='Mo').parent       # so go up one level to get the tag.
<<<a class="pig" href="http://example.com/mo" id="link2">Mo</a>>>

>>> pig_soup.find(href="http://example.com/curly")
<<<a class="pig" href="http://example.com/curly" id="link3">Curly.</a>>>

>>> import re

# Find the first tag with an href attribute containing 'curly'.
>>> pig_soup.find(href=re.<<compile>>(r"curly"))
<<<a class="pig" href="http://example.com/curly" id="link3">Curly.</a>>

# Find the first tag with a string that starts with 'Cu'.
>>> pig_soup.find(string=re.<<compile>>(r"^Cu")).parent
<<<a class="pig" href="http://example.com/curly" id="link3">Curly.</a>>>

# Find all tags with text containing 'Three'.
>>> [tag.parent for tag in pig_soup.find_all(string=re.<<compile>>(r"Three"))]
<<[<title>Three Little Pigs</title>, <b>The Three Little Pigs</b>]>>

# Find all tags with an 'id' attribute.
>>> pig_soup.find_all(<<id>>=True)
<<[<a class="pig" href="http://example.com/larry" id="link1">Larry,</a>,
 <a class="pig" href="http://example.com/mo" id="link2">Mo</a>,
 <a class="pig" href="http://example.com/curly" id="link3">Curly.</a>]>>

# Final the names all tags WITHOUT an 'id' attribute.
>>> [tag.name for tag in pig_soup.find_all(<<id>>=False)]
<<['html', 'head', 'title', 'body', 'p', 'b', 'p', 'p', 'p']>>
