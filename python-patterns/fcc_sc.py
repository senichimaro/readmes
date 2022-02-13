### Find the largest number
# largest_so_far = -1
# print('Before', largest_so_far)
# for the_num in [9, 41, 12, 3, 74, 15] :
#     if the_num > largest_so_far :
#         largest_so_far = the_num
#     print(largest_so_far, the_num)
# print('After',largest_so_far)



### Below is code to find the smallest value
### from a list of values.
### One line has an error
### that will cause the code to not work
### as expected. Which line is it?:
### "break" line : cause force the exit of the look
### after the first value
# smallest = None
# print('Before:',smallest)
# for itervar in [300, 42, 12, 9, 74, 15] :
#     if smallest is None or itervar < smallest :
#         smallest = itervar
#         # break
#     print('Loop:', itervar, smallest)
# print('Smallest:', smallest)



### Parsig and Extracting
### Extract the host from an Email header.
# data = 'From stephen.marquard@uct.ac.za Sat Jan 5 09:14:16 2008'
# atpos = data.find('@')
# print(atpos)
# sppos = data.find(' ', atpos)
# print(sppos)
# host = data[atpos + 1 : sppos]
# print(host)



### File Handle
# try:
#     fhand = open('README.md')
#     for line in fhand:
#         print(line)
# except:
#     print('file wasnt open')



### Count Lines from a File
# fhand = open('README.md')
# count = 0
# for line in fhand:
#     count = count + 1
# print('Line Count:', count)



### Prompt the File Name
# fname = input('Enter the file name:')
# fpatt = input('Text pattern to find?:')
# try:
#     fhand = open(fname)
# except:
#     print('El nombre del Archivo no es correcto')
#     quit()
# count = 0
# for line in fhand :
#     if not line.startswith(fpatt) :
#         continue
#     count = count + 1
# print('There were', count, 'lines starts with', '"'+fpatt+'"', 'in', fname)



### loop calculation pattern
# total = 0
# count = 0
# while True :
#   inp = input('Enter a number: ')
#   if inp == 'done' : break
#   try :
#     value = float(inp)
#   except:
#     print('Enter a valid number')
#     quit()
#   total = total + value
#   count = count + 1
# average = total / count
# print('Average:', average)



### Counters with Dictionary
### Checking to see if a key is already in a dictionary and assuming a default value if is not there
# counts = dict()
# names = ['csev','cwen','cwen','zqian','csev']
# for name in names :
#   counts[name] = counts.get(name,0) + 1
# print(counts)



### Read a file and count all the words in the file
# name = input('Enter filename:')
# fhandle = open(name)
# counts = dict()
# for line in fhandle :
#   words = line.split()
#   for word in words :
#     counts[word] = counts.get(word,0) + 1
#
# bigcount = None
# bigword = None
# for word,count in counts.items():
#   if bigcount is None or count > bigcount :
#     bigword = word
#     bigcount = count
#
# print(bigword, bigcount)



# c = {'a':10, 'c':22, 'b':1}
# d = {'a':10, 'b':1, 'c':22}
# e = {'b':1, 'a':100, 'c':22}
# print('dic c: ',c)
# print('itm c: ',c.items())
# print('sor c: ',sorted(c.items()))






















#
