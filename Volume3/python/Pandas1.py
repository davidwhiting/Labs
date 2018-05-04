
>>> import numpy as np
>>> import pandas as pd

# Initialize a Series of random entries with an index of letters.
>>> pd.Series(np.random.random(4), index=['a', 'b', 'c', 'd'])
<<a    0.474170
b    0.106878
c    0.420631
d    0.279713
dtype: float64>>

# The default index is integers from 0 to the length of the data.
>>> pd.Series(np.random.random(4), name="uniform draws")
<<0    0.767501
1    0.614208
2    0.470877
3    0.335885
Name: uniform draws, dtype: float64>>

>>> s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'], name="some ints")

>>> s1.values                           # Get the entries as a NumPy array.
<<array([1, 2, 3, 4])>>

>>> print(s1.name, s1.dtype, sep=", ")  # Get the name and dtype.
<<some ints, int64>>

>>> s1.index                            # Get the pd.Index object.
<<Index(['a', 'b', 'c', 'd'], dtype='object')>>

>>> s2 = pd.Series([10, 20, 30], index=["apple", "banana", "carrot"])
>>> s2
<<apple     10
banana    20
carrot    30
dtype: int64>>

# s2[0] and s2["apple"] refer to the same entry.
>>> print(s2[0], s2["apple"], s2["carrot"])
<<10 10 30>>

>>> s2[0] += 5              # Change the value of the first entry.
>>> s2["dewberry"] = 0      # Add a new value with label 'dewberry'.
>>> s2
<<apple       15
banana      20
carrot      30
dewberry     0
dtype: int64>>

# Initialize a Series from a dictionary.
>>> pd.Series({"eggplant":3, "fig":5, "grape":7}, name="more foods")
<<eggplant    3
fig         5
grape       7
Name: more foods, dtype: int64>>

>>> s3 = pd.Series({"lions":2, "tigers":1, "bears":3}, name="oh my")
>>> s3
<<bears     3
lions     2
tigers    1
Name: oh my, dtype: int64>>

# Get a subset of the data by regular slicing.
>>> s3[1:]
<<lions     2
tigers    1
Name: oh my, dtype: int64>>

# Get a subset of the data with fancy indexing.
>>> s3[np.array([len(i) == 5 for i in s3.index])]
<<bears    3
lions    2
Name: oh my, dtype: int64>>

# Get a subset of the data by providing several index labels.
>>> s3[ ["tigers", "bears"] ]
tigers    1                     # Note that the entries are reordered,
bears     3                     # and the name stays the same.
<<Name: oh my, dtype: int64>>

>>> val = 4     #desired constant value of Series
>>> n = 6       #desired length of Series
>>> s3 = pd.Series(val, index=np.arange(n))
>>> s3
0    4
1    4
2    4
3    4
4    4
5    4
dtype: int64

>>> s4 = pd.Series([1, 2, 4], index=['a', 'c', 'd'])
>>> s5 = pd.Series([10, 20, 40], index=['a', 'b', 'd'])
>>> 2*s4 + s5
<<a    12.0>>
<<b     NaN>>                           # s4 doesn't have an entry for b, and
<<c     NaN>>                           # s5 doesn't have an entry for c, so
<<d    48.0>>                           # the combination is Nan (np.nan / None).
<<dtype: float64>>

# Make an index of the first three days in July 2000.
>>> pd.date_range("7/1/2000", "7/3/2000", freq='D')
<<DatetimeIndex(['2000-07-01', '2000-07-02', '2000-07-03'],
                dtype='datetime64[ns]', freq='D')>>

>>> x = pd.Series(np.random.randn(4), ['a', 'b', 'c', 'd'])
>>> y = pd.Series(np.random.randn(5), ['a', 'b', 'd', 'e', 'f'])
>>> df1 = pd.DataFrame({"series 1": x, "series 2": y})
>>> df1
<<   series 1  series 2
a -0.365542  1.227960
b  0.080133  0.683523
c  0.737970       NaN
d  0.097878 -1.102835
e       NaN  1.345004
f       NaN  0.217523>>

>>> df1["series1"].dropna()
<<a   -0.365542
b    0.080133
c    0.737970
d    0.097878
Name: series 1, dtype: float64>>

>>> data = np.random.random((3, 4))
>>> pd.DataFrame(data, index=['A', 'B', 'C'], columns=np.arange(1, 5))

            1	        2	        3	        4
A	 0.065646	 0.968593	 0.593394	 0.750110
B	 0.803829	 0.662237	 0.200592	 0.137713
C	 0.288801	 0.956662	 0.817915	 0.951016
3 rows     4 columns

>>> grade=['eighth', 'ninth', 'tenth']
>>> subject=['math', 'science', 'english']
>>> myindex = pd.MultiIndex.from_product([grade, subject], names=['grade', 'subject'])
>>> myseries = pd.Series(np.random.randn(9), index=myindex)
>>> myseries
grade   subject
eighth  math       1.706644
        science   -0.899587
        english   -1.009832
ninth   math       2.096838
        science    1.884932
        english    0.413266
tenth   math      -0.924962
        science   -0.851689
        english    1.053329
dtype: float64

>>> df = pd.DataFrame(np.random.randn(4, 2), index=['a', 'b', 'c', 'd'],
                      columns = ['I', 'II'])
>>> df[:2]

          I        II
a  0.758867  1.231330
b  0.402484 -0.955039

[2 rows x 2 columns]

>>> # select rows a and c, column II
>>> df.loc[['a','c'], 'II']

a    1.231330
c    0.556121
Name: II, dtype: float64

>>> # select last two rows, first column
>>> df.iloc[-2:, 0]

c   -0.475952
d   -0.518989
Name: I, dtype: float64

>>> # get second column of df
>>> df['II']        # or, equivalently, df.II

a    1.231330
b   -0.955039
c    0.556121
d    0.173165
Name: II, dtype: float64

>>> # set second columns to zeros
>>> df['II'] = 0
>>> df['II']
a    0
b    0
c    0
d    0
Name: II, dtype: int64

>>> # add additional column of ones
>>> df['III'] = 1
>>> df
            I  II  III
a   -0.460457   0    1
b    0.973422   0    1
c   -0.475952   0    1
d   -0.518989   0    1

>>> import matplotlib.pyplot as plt
>>> N = 1000        # length of random walk
>>> s = np.zeros(N)
>>> s[1:] = np.random.binomial(1, .5, size=(N-1,))*2-1 #coin flips
>>> s = pd.Series(s)
>>> s = s.cumsum()  # random walk
>>> s.plot()
>>> plt.ylim([-50, 50])
>>> plt.show()
>>> plt.close()

>>> #build toy data for SQL operations
>>> name = ['Mylan', 'Regan', 'Justin', 'Jess', 'Jason', 'Remi', 'Matt', 'Alexander', 'JeanMarie']
>>> sex = ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'M', 'F']
>>> age = [20, 21, 18, 22, 19, 20, 20, 19, 20]
>>> rank = ['Sp', 'Se', 'Fr', 'Se', 'Sp', 'J', 'J', 'J', 'Se']
>>> ID = range(9)
>>> aid = ['y', 'n', 'n', 'y', 'n', 'n', 'n', 'y', 'n']
>>> GPA = [3.8, 3.5, 3.0, 3.9, 2.8, 2.9, 3.8, 3.4, 3.7]
>>> mathID = [0, 1, 5, 6, 3]
>>> mathGd = [4.0, 3.0, 3.5, 3.0, 4.0]
>>> major = ['y', 'n', 'y', 'n', 'n']
>>> studentInfo = pd.DataFrame({'ID': ID, 'Name': name, 'Sex': sex, 'Age': age, 'Class': rank})
>>> otherInfo = pd.DataFrame({'ID': ID, 'GPA': GPA, 'Financial_Aid': aid})
>>> mathInfo = pd.DataFrame({'ID': mathID, 'Grade': mathGd, 'Math_Major': major})

>>> mathInfo.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 5 entries, 0 to 4
Data columns (total 3 columns):
Grade         5 non-null float64
ID            5 non-null int64
Math_Major    5 non-null object
dtypes: float64(1), int64(1), object(1)

>>> mathInfo.head()
   Grade  ID Math_Major  ID  Age  GPA
0    4.0   0          y   0   20  3.8
1    3.0   1          n   2   18  3.0
2    3.5   5          y   4   19  2.8
3    3.0   6          n   6   20  3.8
4    4.0   3          n   7   19  3.4

>>> # SELECT ID, Age FROM studentInfo
>>> studentInfo[['ID', 'Age']]

>>> # SELECT ID, GPA FROM otherInfo WHERE Financial_Aid = 'y'
>>> otherInfo[otherInfo['Financial_Aid']=='y'][['ID', 'GPA']]

>>> # SELECT Name FROM studentInfo WHERE Class = 'J' OR Class = 'Sp'
>>>  studentInfo[studentInfo['Class'].isin(['J','Sp'])]['Name']
[language=SQL]
SELECT ID, Name from studentInfo WHERE Age > 19 AND Sex = 'M'

>>> # SELECT * FROM studentInfo INNER JOIN mathInfo ON studentInfo.ID = mathInfo.ID
>>> pd.merge(studentInfo, mathInfo, on='ID') # INNER JOIN is the default
   Age Class  ID    Name Sex  Grade Math_Major
0   20    Sp   0   Mylan   M    4.0          y
1   21    Se   1   Regan   F    3.0          n
2   22    Se   3    Jess   F    4.0          n
3   20     J   5    Remi   F    3.5          y
4   20     J   6    Matt   M    3.0          n
[5 rows x 7 columns]

>>> # SELECT GPA, Grade FROM otherInfo FULL OUTER JOIN mathInfo ON otherInfo.ID = mathInfo.ID
>>> pd.merge(otherInfo, mathInfo, on='ID', how='outer')[['GPA', 'Grade']]
   GPA  Grade
0  3.8    4.0
1  3.5    3.0
2  3.0    NaN
3  3.9    4.0
4  2.8    NaN
5  2.9    3.5
6  3.8    3.0
7  3.4    NaN
8  3.7    NaN
[9 rows x 2 columns]

>>> #create new DataFrame
>>> sibs = [0, 1, 0, 5, 2, 9]
>>> sibInfo = pd.DataFrame({'Siblings':sibs})
>>> sibInfo
   Siblings
0         0
1         1
2         0
3         5
4         2
5         9
[6 rows x 1 columns]

>>> #now join studentInfo with sibInfo
>>> studentInfo.join(sibInfo)
   Age Class  ID    Name Sex  Siblings
0   20    Sp   0    Mylan   M         0
1   21    Se   1   Regan   F         1
2   18    Fr   2     Justin   M         0
3   22    Se   3   Jess   F         5
4   19    Sp   4     Jason   M         2
5   20     J   5  Remi   F         9
6   20     J   6   Matt   M       NaN
7   19     J   7  Alexander   M       NaN
8   20    Se   8     JeanMarie   F       NaN
[9 rows x 6 columns]

>>> #join studentInfo and mathInfo on indexes
>>> studentInfo.join(mathInfo, lsuffix='_left', rsuffix='_right', how='inner')
   Age Class  ID_left   Name Sex  Grade  ID_right Math_Major
0   20    Sp        0   Mylan   M    4.0         0          y
1   21    Se        1  Regan   F    3.0         1          n
2   18    Fr        2    Justin   M    3.5         5          y
3   22    Se        3  Jess   F    3.0         6          n
4   19    Sp        4    Jason   M    4.0         3          n
[5 rows x 8 columns]

>>> #create another df holding more math info
>>> mathID2 = [0, 5, 2, 4]
>>> mathGd2 = [4.0, 3.5, 2.0, 3.0]
>>> major2 = ['y', 'y', 'n', 'y']
>>> mathInfo2 = pd.DataFrame({'Grade':mathGd2, 'ID':mathID2, 'Math_Major':major2})
>>> pd.concat([mathInfo, mathInfo2], ignore_index=True).drop_duplicates()
   Grade  ID Math_Major
0    4.0   0          y
1    3.0   1          n
2    3.5   5          y
3    3.0   6          n
4    4.0   3          n
7    2.0   2          n
8    3.0   4          y
[7 rows x 3 columns]

>>> x = pd.Series(np.random.randn(4), index=['a', 'b', 'c', 'd'])
>>> y = pd.Series(np.random.randn(5), index=['a', 'b', 'd', 'e', 'f'])
>>> x**2
a    1.710289
b    0.157482
c    0.540136
d    0.202580
dtype: float64
>>> z = x + y
>>> z
a    0.123877
b    0.278435
c         NaN
d   -1.318713
e         NaN
f         NaN
dtype: float64
>>> np.log(z)
a   -2.088469
b   -1.278570
c         NaN
d         NaN
e         NaN
f         NaN
dtype: float64

>>> df = pd.DataFrame(np.random.randn(4,2), index=['a', 'b', 'c', 'd'], columns=['I', 'II'])
>>> df
          I        II
a -0.154878 -1.097156
b -0.948226  0.585780
c  0.433197 -0.493048
d -0.168612  0.999194

[4 rows x 2 columns]

>>> df.transpose()
           a         b         c         d
I  -0.154878 -0.948226  0.433197 -0.168612
II -1.097156  0.585780 -0.493048  0.999194

[2 rows x 4 columns]

>>> # switch order of columns, keep only rows 'a' and 'c'
>>> df.reindex(index=['a', 'c'], columns=['II', 'I'])
         II         I
a -1.097156 -0.154878
c -0.493048  0.433197

[2 rows x 2 columns]

>>> # sort descending according to column 'II'
>>> df.sort_values('II', ascending=False)
          I        II
d -0.168612  0.999194
b -0.948226  0.585780
c  0.433197 -0.493048
a -0.154878 -1.097156

[4 rows x 2 columns]

>>> df.describe()
              I        II
count  4.000000  4.000000
mean  -0.209630 -0.001308
std    0.566696  0.964083
min   -0.948226 -1.097156
25%   -0.363516 -0.644075
50%   -0.161745  0.046366
75%   -0.007859  0.689133
max    0.433197  0.999194

[8 rows x 2 columns]

>>> df.mean(axis=1)
a   -0.626017
b   -0.181223
c   -0.029925
d    0.415291
dtype: float64

>>> df.cov()
           I        II
I   0.321144 -0.256229
II -0.256229  0.929456

[2 rows x 2 columns]

>>> x = pd.Series(np.arange(5))
>>> y = pd.Series(np.random.randn(5))
>>> x.iloc[3] = np.nan
>>> x + y
0    0.731521
1    0.623651
2    2.396344
3         NaN
4    3.351182
dtype: float64

>>> (x + y).dropna()
0    0.731521
1    0.623651
2    2.396344
4    3.351182
dtype: float64

>>> # fill missing data with 0, add
>>> x.fillna(0) + y
0    0.731521
1    0.623651
2    2.396344
3    1.829400
4    3.351182
dtype: float64

>>> df.to_csv("my_df.csv")
