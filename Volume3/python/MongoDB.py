
>>> from pymongo import MongoClient
# Create an instance of a client connected to a database running
# at the default host IP and port of the MongoDB service on your machine.
>>> client = MongoClient()

# Create a new database.
>>> db = client.db1

# Create a new collection in the db database.
>>> col = db.collection1

# Insert one document with fields 'name' and 'age' into the collection.
>>> col.insert_one({'name': 'Jack', 'age': 23})

# Insert another document. Notice that the value of a field can be a string,
# integer, truth value, or even an array.
>>> col.insert_one({'name': 'Jack', 'age': 22, 'student': True,
...                 'classes': ['Math', 'Geography', 'English']})

# Insert many documents simultaneously into the collection.
>>> col.insert_many([
...     {'name': 'Jill', 'age': 24, 'student': False},
...     {'name': 'John', 'nickname': 'Johnny Boy', 'soldier': True},
...     {'name': 'Jeremy', 'student': True, 'occupation': 'waiter'}  ])

>>> client = MongoClient()
>>> db = client.db1
>>> col = db.collection1

# Find all the documents that have a 'name' field containing the value 'Jack'.
>>> data = col.find({'name': 'Jack'})

# Find the FIRST document with a 'name' field containing the value 'Jack'.
>>> data = col.find_one({'name': 'Jack'})

# Search for documents containing True in the 'student' field.
>>> students = col.find({'student': True})
>>> students.count()                # There are 2 matching documents.
2

# List the first student's data.
# Notice that each document is automatically assigned an ID number as '_id'.
>>> students[0]
<<{'_id': ObjectId('59260028617410748cc7b8c7'),
 'age': 22,
 'classes': ['Math', 'Geography', 'English'],
 'name': 'Jack',
 'student': True}>>

# Get the age of the first student.
>>> students[0]['age']
22

# List the data for every student.
>>> list(students)
<<[{'_id': ObjectId('59260028617410748cc7b8c7'),
  'age': 22,
  'classes': ['Math', 'Geography', 'English'],
  'name': 'Jack',
  'student': True},
 {'_id': ObjectId('59260028617410748cc7b8ca'),
  'name': 'Jeremy',
  'occupation': 'waiter',
  'student': True}]>>

# Query for everyone that is either above the age of 23 or a soldier.
>>> results = col.find({'$or':[{'age':{'$gt': 23}},{'soldier': True}]})

# Query for everyone that is a student (those that have a 'student' attribute
# and haven't been expelled).
>>> results = col.find({'student': {'$not': {'$in': [False, 'Expelled']}}})

# Query for everyone that has a student attribute.
>>> results = col.find({'student': {'$exists': True}})

# Query for people whose name contains a the letter 'e'.
>>> import re
>>> results = col.find({'name': {'$regex': re.compile('e')}})

{'name': 'Jason', 'age': 16,
 'student': {'year':'senior', 'grades': ['A','C','A','B'],'flunking': False},
 'jobs':['waiter', 'custodian']}

# Query for student that are seniors
>>> results = col.find({'student.year': 'senior'})

# Query for students that have an A in their first class.
>>> results = col.find({'student.grades.0': 'A'})

# Find all the values in the names field.
>>> col.distinct("name")
<<['Jack', 'Jill', 'John', 'Jeremy']>>

# Delete the first person from the database whose name is Jack.
>>> col.delete_one({'name':'Jack'})

# Delete everyone from the database whose name is Jack.
>>> col.delete_many({'name':'Jack'})

# Clear the entire collection.
>>> col.delete_many({})

# Sort the students by name in alphabetic order.
>>> results = col.find().sort('name', 1)
>>> for person in results:
...     print(person['name'])
...
Jack
Jack
Jeremy
Jill
John

# Sort the students oldest to youngest, ignoring those whose age is not listed.
>>> results = col.find({'age': {'$exists': True}}).sort('age', -1)
>>> for person in results:
...    print(person['name'])
...
Jill
Jack
Jack

# Update the first person from the database whose name is Jack to include a
# new field 'lastModified' containing the current date.
>>> col.update_one({'name':'Jack'},
...                {'$currentDate': {'lastModified': {'$type': 'date'}}})

# Increment everyones age by 1, if they already have an age field.
>>> col.update_many({'age': {'$exists': True}}, {'$inc': {'age': 1}})

# Give the first John a new field 'best_friend' that is set to True.
>>> col.update_one({'name':'John'}, {'$set': {'best_friend': True}})

$pip install tweepy

import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
from pymongo import MongoClient
import json

#Set up the databse
client = MongoClient()
mydb = client.db1
twitter = mydb.collection1

f = open('trump.txt','w') #If you want to write to a file

consumer_key = #Your Consumer Key
consumer_secret = #Your Consumer Secret
access_token = #Your Access Token
access_secret = #Your Access Secret

my_auth = OAuthHandler(consumer_key, consumer_secret)
my_auth.set_access_token(access_token, access_secret)

class StreamListener(tweepy.StreamListener):
    def on_status(self, status):
        print(status.text)

    def on_data(self, data):
        try:
            twitter.insert_one(json.loads(data)) #Puts the data into your MongoDB
            f.write(str(data)) #Writes the data to an output file
            return True
        except BaseException as e:
            print(str(e))
            print("Error")
        return True

    def on_error(self, status):
        print(status)
        if status_code == 420: #This means twitter has blocked us temporarily, so we want to stop or they will get mad. Wait 30 minutes or so and try again. Running this code often in a short period of time will cause twitter to block you. But you can stream tweets for as long as you want without any problems.
            return False
        else:
            return True

stream_listener = StreamListener()
stream = tweepy.Stream(auth=my_auth, listener=stream_listener)
stream.filter(track=["trump"]) #This pulls all tweets that include the keyword "trump". Any number of keywords can be searched for.


col.update({'name': 'Jack','student': True})

f = list(col.find({'age': {'$lt': 24}, 'classes': {'$in': ['Art', 'English']}}))
