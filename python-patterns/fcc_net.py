### a simple web browser
# import socket
#
# mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# mysock.connect(('data.pr4e.org', 80))
# cmd = 'GET http://data.pr4e.org/romeo.txt HTTP/1.0\r\n\r\n'.encode()
# # print(cmd)
# mysock.send(cmd)
#
# while True:
#     data = mysock.recv(512)
#     if (len(data) < 1):
#         break
#     print(data.decode())
#     # print(data)
# mysock.close()



### Simplier Web Browser
# import urllib.request, urllib.parse, urllib.error
#
# fhand = urllib.request.urlopen('http://data.pr4e.org/romeo.txt')
# for line in fhand:
#     print(line.decode().strip())



### Count url words
# import urllib.request, urllib.parse, urllib.error
# fhand = urllib.request.urlopen('http://data.pr4e.org/romeo.txt')
# counts = dict()
# for line in fhand:
#     words = line.decode().split()
#     for word in words:
#         counts[word] = counts.get(word, 0) + 1
# common_words = dict()
# most_common = max(counts.values())
# for k,v in counts.items():
#     if v >= most_common:
#         common_words[k] = v
# for common_k,common_v in common_words.items():
#     print('Most common word is', '"'+common_k+'"', 'existing', common_v, 'times')



### Web Scraping with Beautifulsoup4
# import urllib.request, urllib.parse, urllib.error
# from bs4 import BeautifulSoup
# import ssl
#
# # Ignore SSL certificate errors
# ctx = ssl.create_default_context()
# ctx.check_hostname = False
# ctx.verify_mode = ssl.CERT_NONE
#
# # url = input('Enter url - ')
# # url = 'http://www.dr-chuck.com/page1.htm'
# url = 'http://www.si.umich.edu/'
# # html = urllib.request.urlopen(url).read()
# html = urllib.request.urlopen(url, context=ctx).read()
# soup = BeautifulSoup(html, 'html.parser')
#
# # Retrieve all of the anchor tags
# tags = soup('a')
# for tag in tags:
#     print(tag.get('href', None))


### Web Services : XML Schema
# import xml.etree.ElementTree as ET
#
# data = '''<person>
# <name>Chuck</name>
# <phone type="intl">
#     +1 734 303 4456
# </phone>
# <email hide="yes"/>
# </person>'''
#
# tree = ET.fromstring(data)
# print('Name:', tree.find('name').text)
# print('Attr:', tree.find('email').get('hide'))



### Web Services : JSON
# import json
#
# input = '''[
# {
#     "id":"001",
#     "x":"2",
#     "name":"Chuck"
# },
# {
#     "id":"009",
#     "x":"7",
#     "name":"Carol"
# }
# ]'''
#
# info = json.loads(input)
# print('User count:', len(info))
# for item in info:
#     print('Name:', item['name'])
#     print('Id:', item['id'])
#     print('Attribute:', item['x'])




### Web Services : APIs (Google Map API)
# import urllib.request, urllib.parse, urllib.error
# import json
# import ssl
#
# api_key = False
# # If you have a Google Places API key, enter it here
# # api_key = 'AIzaSy___IDByT70'
# # https://developers.google.com/maps/documentation/geocoding/intro
#
# if api_key is False:
#     api_key = 42
#     serviceurl = 'http://py4e-data.dr-chuck.net/json?'
# else :
#     serviceurl = 'https://maps.googleapis.com/maps/api/geocode/json?'
#
# # Ignore SSL certificate errors
# ctx = ssl.create_default_context()
# ctx.check_hostname = False
# ctx.verify_mode = ssl.CERT_NONE
#
# while True:
#     address = input('Enter location: ')
#     if len(address) < 1: break
#
#     parms = dict()
#     parms['address'] = address
#     if api_key is not False: parms['key'] = api_key
#     url = serviceurl + urllib.parse.urlencode(parms)
#
#     print('Retrieving', url)
#     uh = urllib.request.urlopen(url, context=ctx)
#     data = uh.read().decode()
#     print('Retrieved', len(data), 'characters')
#
#     try:
#         js = json.loads(data)
#     except:
#         js = None
#
#     if not js or 'status' not in js or js['status'] != 'OK':
#         print('==== Failure To Retrieve ====')
#         print(data)
#         continue
#
#     print(json.dumps(js, indent=4))
#
#     lat = js['results'][0]['geometry']['location']['lat']
#     lng = js['results'][0]['geometry']['location']['lng']
#     print('lat', lat, 'lng', lng)
#     location = js['results'][0]['formatted_address']
#     print(location)



### Web Services : APIs (Twitter API)
# import urllib.request, urllib.parse, urllib.error
# import twurl
# import json
# import ssl
#
# # https://apps.twitter.com/
# # Create App and get the four strings, put them in hidden.py
#
# TWITTER_URL = 'https://api.twitter.com/1.1/friends/list.json'
#
# # Ignore SSL certificate errors
# ctx = ssl.create_default_context()
# ctx.check_hostname = False
# ctx.verify_mode = ssl.CERT_NONE
#
# while True:
#     print('')
#     acct = input('Enter Twitter Account:')
#     if (len(acct) < 1): break
#     url = twurl.augment(TWITTER_URL,
#                         {'screen_name': acct, 'count': '5'})
#     print('Retrieving', url)
#     connection = urllib.request.urlopen(url, context=ctx)
#     data = connection.read().decode()
#
#     js = json.loads(data)
#     print(json.dumps(js, indent=2))
#
#     headers = dict(connection.getheaders())
#     print('Remaining', headers['x-rate-limit-remaining'])
#
#     for u in js['users']:
#         print(u['screen_name'])
#         if 'status' not in u:
#             print('   * No status found')
#             continue
#         s = u['status']['text']
#         print('  ', s[:50])




### Objects : A Sample Class
# class PartyAnimal:
#     x = 0
#
#     def party(self):
#         self.x = self.x + 1
#         print("So far", self.x)
#
#
# an = PartyAnimal()
#
# an.party()
# an.party()
# an.party()



### Object Lifecycle
# class PartyAnimal:
#     x = 0
#     name = ""
#
#     def __init__(self, z):
#         self.name = z
#         print(self.name, "constructed")
#
#     def party(self):
#         self.x = self.x + 1
#         print(self.name,"party count", self.x)
#
# s = PartyAnimal("Sally")
# s.party()
#
# j = PartyAnimal("Jim")
# j.party()
# j.party()



### Object Inheritance
# class PartyAnimal:
#     x = 0
#     def __init__(self,z):
#         self.name = z
#         print(self.name,"constructed")
#
#     def party(self):
#         self.x = self.x + 1
#         print(self.name,"party count", self.x)
#
# class FootballFan(PartyAnimal):
#     points = 0
#     def touchdown(self):
#         self.points = self.points + 7
#         self.party()
#         print(self.name,"points", self.points)
#
# s = PartyAnimal("Sally")
# j = FootballFan("Jim")
# j.touchdown()
# s.party()
# j.touchdown()

























#
