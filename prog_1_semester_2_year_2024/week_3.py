print("week_3")

print("-------------------------")

# len => count the total length of an string datatype or array datatype
characters_1 = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
text_1 = 'hello world'

print("Using len method")
print(len(characters_1))
print(len(text_1))

print("-------------------------")

numbers_1 = [5, 3, 2, 6, 7, 4, 1]

print("Using sort & sorted method")
print(sorted(numbers_1))  # cannot sort list
print(numbers_1)

numbers_1.sort()  # can sort list
print(numbers_1)

print("-------------------------")

numbers_2 = [1, 2, 3, 4, 5, 6, 7]

print("Using reversed method")
print(list(reversed(numbers_2)))
print(numbers_2)

numbers_2.reverse()
print(numbers_2)

print("-------------------------")

numbers_3 = [2, 3, 1, 5, 4, 5, 4, 1, 5, 1]

# count => count the given parameters based on total occurences
print("Using count method")
print(numbers_3.count(5))

text_1 = 'banana'
print(text_1.count('a'))

print("-------------------------")

print("Using index method")
print(numbers_3)

# index => locate the given parameters based on the current Index
print(numbers_3.index(5))

print(text_1)
print(text_1.index("n"))

print("-------------------------")

print("Using upper & lower method")

# upper => makes the string uppercase
# lower => makes the string lowercase
text_2 = "Hello World!"

print(text_2.upper())
print(text_2.lower())

print("-------------------------")
