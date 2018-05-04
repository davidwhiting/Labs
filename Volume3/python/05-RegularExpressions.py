
>>> import re
>>> pattern = re.<<compile>>("cat")     # Make a pattern for finding 'cat'.
>>> bool(pattern.search("cat"))     # 'cat' matches 'cat', of course.
<<True>>
>>> bool(pattern.search("catfish")) # 'catfish' also contains 'cat'.
<<True>>
>>> bool(pattern.search("hat"))     # 'hat' does not contain 'cat'.
<<False>>

>>> pattern = re.<<compile>>("cat")
>>> bool(pattern.match("catfish"))
<<True>>
>>> bool(pattern.match("fishcat"))
<<False>>
>>> bool(pattern.search("fishcat"))
<<True>>

>>> bool(re.<<compile>>("cat").search("catfish"))
<<True>>
>>> bool(re.search("cat", "catfish"))
<<True>>

>>> pattern = re.<<compile>>("cat")
>>> positive = ["cat", "catfish", "fish cat", "your cat ran away"]
>>> assert all(pattern.search(p) for p in positive)

>>> dollar = re.<<compile>>(r"\$3\.99\? Thanks\.")
>>> bool(dollar.search("$3.99? Thanks."))
<<True>>
>>> bool(dollar.search("$3\.99? Thanks."))
<<False>>
>>> bool(dollar.search("$3.99?"))   # Doesn't contain the entire pattern.
<<False>>

>>> has_x, just_x = re.<<compile>>(r"x"), re.<<compile>>(r"^x$")
>>> for test in ["x", "xabc", "abcx"]:
...     print(test + ':', bool(has_x.search(test)), bool(just_x.search(test)))
...
<<x: True True
xabc: True False>>                    # Starts with 'x', but doesn't end with it.
<<abcx: True False>>                    # Ends with 'x', but doesn't start with it.

>>>bool(re.search("^W","Hello\nWorld"))
<<False>>
>>>bool(re.search("^W","Hello\nWorld", re.MULTILINE))
<<True>>
>>>pattern1 = re<<.compile>>("^W")
>>>pattern2 = re<<.compile>>("^W", re.MULTILINE)
>>>bool(pattern1.search("Hello\nWorld"))
<<False>>
>>>bool(pattern2.search("Hello\nWorld"))
<<True>>

>>> rb, rbg = re.<<compile>>(r"^red$|^blue$"), re.<<compile>>(r"^red$|^blue$|^green$")
>>> for test in ["red", "blue", "green", "redblue"]:
...     print(test + ":", bool(rb.search(test)), bool(rbg.search(test)))
<<red: True True
blue: True True
green: False True
redblue: False False>>                # The line anchors prevent matching here.

>>> fish = re.<<compile>>(r"^(one|two) fish$")
>>> for test in ["one fish", "two fish", "red fish", "one two fish"]:
...     print(test + ':', bool(fish.search(test)))
...
<<one fish: True
two fish: True
red fish: False
one two fish: False>>

>>> p1, p2 = re.<<compile>>(r"^[a-z][^0-7]$"), re.<<compile>>(r"^[^abcA-C][0-27-9]$")
>>> for test in ["d8", "aa", "E9", "EE", "d88"]:
...     print(test + ':', bool(p1.search(test)), bool(p2.search(test)))
...
<<d8: True True>>
<<aa: True False>>                      # a is not in [^abcA-C] or [0-27-9].
<<E9: False True>>                      # E is not in [a-z].
<<EE: False False>>                     # E is not in [a-z] or [0-27-9].
<<d88: False False>>                    # Too many characters.

# Match any three-character string with a digit in the middle.
>>> pattern = re.<<compile>>(r"^.\d.$")
>>> for test in ["a0b", "888", "n2%", "abc", "cat"]:
...     print(test + ':', bool(pattern.search(test)))
...
<<a0b: True
888: True
n2%: True
abc: False
cat: False>>

# Match two letters followed by a number and two non-newline characters.
>>> pattern = re.<<compile>>(r"^[a-zA-Z][a-zA-Z]\d..$")
>>> for test in ["tk421", "bb8!?", "JB007", "Boba?"]:
...     print(test + ':', bool(pattern.search(test)))
..
<<tk421: True
bb8!?: True
JB007: True
Boba?: False>>

# Match 0 or more 'a' characters, ending in a 'b'.
>>> pattern = re.<<compile>>(r"^a*b$")
>>> for test in ["b", "ab", "aaaaaaaaaab", "aba"]:
...     print(test + ':', bool(pattern.search(test)))
...
<<b: True>>                             # 0 'a' characters, then 1 'b'.
<<ab: True>>
<<aaaaaaaaaab: True>>                   # Several 'a' characters, then 1 'b'.
<<aba: False>>                          # 'b' must be the last character.

# Match an 'h' followed by at least one 'i' or 'a' characters.
>>> pattern = re.<<compile>>(r"^h[ia]+$")
>>> for test in ["ha", "hii", "hiaiaa", "h", "hah"]:
...     print(test + ':', bool(pattern.search(test)))
...
<<ha: True
hii: True>>
<<hiaiaa: True>>                        # [ia] matches 'i' or 'a'.
<<h: False>>                            # Need at least one 'i' or 'a'
<<hah: False>>                          # 'i' or 'a' must be the last character.

# Match an 'a' followed by 'b' followed by 0 or 1 'c' characters.
>>> pattern = re.<<compile>>(r"^abc?$")
>>> for test in ["ab", "abc", "abcc", "ac"]:
...     print(test + ':', bool(pattern.search(test)))
...
<<ab: True
abc: True>>
<<abcc: False>>                         # Only up to one 'c' is allowed.
<<ac: False>>                           # Missing the 'b'.

# Match exactly 3 'a' characters.
>>> pattern = re.<<compile>>(r"^a{3}$")
>>> for test in ["aa", "aaa", "aaaa", "aba"]:
...     print(test + ':', bool(pattern.search(test)))
...
<<aa: False>>                           # Too few.
<<aaa: True
aaaa: False>>                         # Too many.
<<aba: False>>

# Match exactly 3 'a' characters, hopefully.
>>> pattern = re.<<compile>>(r"a{3}")
>>> for test in ["aaa", "aaaa", "aaaaa", "aaaab"]:
...     print(test + ':', bool(pattern.search(test)))
...
<<aaa: True
aaaa: True>>                          # Should be too many!
<<aaaaa: True>>                         # Should be too many!
<<aaaab: True>>                         # Too many, and even with the 'b'?

>>> run match_function_definition.py
Enter a string>>> def compile(pattern,string):
<<True>>
Enter a string>>> def  space  ( ) :
<<True>>
Enter a string>>> def func(_dir, file_path='\Desktop\files', val=_PI):
<<True>>
Enter a string>>> def func(num=3., num=.5, num=0.0):
<<True>>
Enter a string>>> def func(num=., error,):
<<False>>
Enter a string>>> def variable:
<<False>>
Enter a string>>> def not.allowed(, *args):
<<False>>
Enter a string>>> def err*r('no parameter name'):
<<False>>
Enter a string>>> def func(value=_MY_CONSTANT, msg='%s' % _DEFAULT_MSG):
<<False>>
Enter a string>>> def func(s1='', a little tricky, s2=''):
<<False>>
Enter a string>>> def func(): Remember your line anchors!
<<False>>
Enter a string>>> deffunc()
<<False>>
Enter a string>>> func():
<<False>>
Enter a string>>> exit


>>> key_1 = "basic"
>>> print("This is a " + key_1 + " way to concatenate strings.")
This is a basic way to concatenate strings.
>>> format_dict = {"key_1": "basic", "key_2": "much more", "key_3": "advanced"}
>>> print("This is a {key_2} {key_3} way to concatenate strings. It's {key_2} flexible.".format(**format_dict))
This is a much more advanced way to concatenate strings. It's much more flexible.

# Find words that start with 'cat'.
>>> expr = re.<<compile>>(r"\bcat\w*")  # \b is the shortcut for a word boundary.

>>> target = "Let's catch some catfish for the cat"
>>> bool(expr.search(target))       # Check to see if there is a match.
<<True>>

>>> expr.findall(target)            # Get all matching substrings.
<<['catch' 'catfish', 'cat']>>

>>> expr.sub("DOG", target)         # Substitute 'DOG' for the matches.
<<"Let's DOG some DOG for the DOG">>

>>> expr.split(target)              # Split the target by the matches.
<<["Let's ", ' some ', ' for the ', '']>>

# Find words that start with 'cat', remembering what comes after the 'cat'.
>>> pig_latin = re.compile(r"\bcat(\w*)")
>>> target = "Let's catch some catfish for the cat"

>>> pig_latin.sub(r"at\1clay", target)  # \1 = (\w*) from the expression.
<<"Let's atchclay some atfishclay for the atclay">>

>>> target = "<abc> <def> <ghi>"

# Match angle brackets and anything in between.
>>> greedy = re.<<compile>>(r"^<.*>$")  # Greedy *
>>> greedy.findall(target)
<<['<abc> <def> <ghi>']>>               # The entire string matched!

# Try again, using the non-greedy version.
>>> nongreedy = re.<<compile>>(r"<.*?>")# Non-greedy *?
>>> nongreedy.findall(target)
<<['<abc>', '<def>', '<ghi>']>>         # Each <> set is an individual match.

# Match any line with 3 consecutive 'a' characters somewhere.
>>> pattern = re.<<compile>>("^.*a{3}.*$", re.MULTILINE)    # Search each line.
>>> pattern.findall("""
This is aaan example.
This is not an example.
Actually, it's an example, but it doesn't match.
This example does maaatch though.""")
<<['This is aaan example.', 'This example does maaatch though.']>>

# Match anything instance of 'cat', ignoring case.
>>> catfinder = re.compile("cat", re.IGNORECASE)
>>> catfinder.findall("cat CAT cAt TAC ctacATT")
<<['cat', 'CAT', 'cAt', 'cAT']>>

"""
k, i, p = 999, 1, 0
while k > i
    i *= 2
    p += 1
    if k != 999
        print("k should not have changed")
    else
        pass
print(p)
"""

# The string given above should become this string.
"""
k, i, p = 999, 1, 0
while k > i:
    i *= 2
    p += 1
    if k != 999:
        print("k should not have changed")
    else:
        pass
print(p)
"""

{
    "John Doe": {
        "birthday": "01/01/1990",
        "email": "john_doe90@hopefullynotarealaddress.com",
        "phone": "(123)456-7890"
        },
    "Jane Smith": {
        "birthday": None,
        "email": None,
        "phone": "(222)111-3333"
        },
# ...
}

$ grep 'regexp' filename

# List details of directories within current directory.
$ ls -l | grep ^d

$ ls -l | awk ' {if ($3 ~ /freddy/) print $9} '

$ ls -l | awk ' {if ($1 ~ /^d/) print $9} '

# Notice the semicolons delimiting the fields. Also, notice that in between the last and first name, that is a comma, not a semicolon.
<ORDER_ID>;<YEAR><MONTH><DAY>;<LAST>,<FIRST>;<ITEM_ID>
