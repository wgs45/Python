from collections import Counter
from collections import defaultdict
import re
import datetime as dt
import datetime

now = datetime.datetime.now()
print(now.ctime())

print("-------------------------")


class MyProduct:
    def __init__(self, n, p):
        self.name = n
        self.price = p

    def __str__(self):
        return f"MyProduct(name={self.name}, price={self.price})"


# Creating an instance of MyProduct with correct arguments
book = MyProduct("Book", 700)
# This will call the __str__ method and print a readable
# representation of the object
print(book)
print(book.name)    # This will print the name attribute
print(book.price)   # This will print the price attribute


print("-------------------------")


class MyProduct:
    def __init__(self, n, p):
        self.name = n
        self.price = p

    def summary(self):
        print("Item: {}\nPrice: {}".format(self.name, self.price))

    def discount(self, rate):
        self.price *= rate


# item 1
book = MyProduct("Book", 700)
book.discount(0.79)
book.summary()

# item 2
car = MyProduct("Van", 700000)
car.discount(0.9)
car.summary()

print("-------------------------")


x = dt.datetime(2024, 10, 22)
print(x)

# x = dt.time(year=2024, month=10, day=22)
# print(x)

# y = dt.time(2024, 10, 22, 10, 30, 45)
# print(y)

print("-------------------------")

# time delta

x = dt.datetime(year=2024, month=10, day=22, hour=1, minute=30)
print(x)

print("-------------------------")

x = dt.datetime(2024, 10, 22, 10, 30, 45)
y = dt.timedelta(days=1, hours=2, minutes=5)

print(x)
print(x + y)
print(x - y)
print(x + y * 2)

print("-------------------------")

# DATETIME time formatted output

x = dt.datetime(2024, 10, 22, 10, 30, 45)
s1 = x.strftime("%Y/%m/%d %H-%M-%S")
print(s1)

print("-------------------------")

s = "2024/10/22 10-30-45"
x = dt.datetime.strptime(s, "%Y/%m/%d %H-%M-%S")
print(s)
print(type(x))

print("-------------------------")

# Lambda function (Known as anonymous function)


def power(x): return x ** 2


print(power(10))


def add(a, b): return a + b


print(add(5, 3))

print("-------------------------")

# Use a line of IF conditional expressions in LAMBDA


def absolute(x): return x if x >= 0 else -x


print(absolute(5))
print(absolute(-5))


def func(x): return (x ** 2 - 40 * x + 350) if 10 <= x < 30 else 50


print(func(5))
print(func(11))
print(func(22))
print(func(33))

print("-------------------------")

# STR.SPLIT(): Split the string into LIST elements

sentence = "This is a test sentences"
print(sentence.split(" "))

print("-------------------------")

# Split string into LIST using string normalization
# re = regular expression


sentence = "This is a test senteces"
time_data = "2024/05/20_12:30:45"

print(re.split("[,.]", sentence))
print(re.split("[/_:]", sentence))

a = [1, -2, 3, -4, 5]
new = []

for x in a:
    new.append(abs(x))

print(new)

str_list = ['This', 'is', 'a', 'test', 'sentences']
print(list(map(str.upper, str_list)))

print("-------------------------")

# Use FILTER() to filter container elements

str_list = ['This', 'is', 'a', 'test', 'sentences']
print(list(filter(lambda x: len(x) >= 3, str_list)))

print("-------------------------")

# Revisiting SORTED(): Customizing the sorting method of target containers

str_list = ['This', 'is', 'a', 'test', 'sentences']
print(sorted(str_list, key=len, reverse=True))

nest_list = [
    [0, 9],
    [1, 8],
    [2, 7],
    [3, 6],
    [4, 5]
]

print(sorted(nest_list))
print(sorted(nest_list, key=lambda x: x[1]))
print(sorted(nest_list, key=lambda x: x[1], reverse=True))

print("-------------------------")

# LIST

a = [1, -2, 3, -4, 5]
print([abs(x) for x in a])

print([x ** 2 for x in a])

str_list = ['This', 'is', 'a', 'test', 'sentences']
print([s.upper() for s in str_list])

print("-------------------------")

# Use IF to filter elements in LIST generation

a = [1, -2, 3, -4, 5]
print([x for x in a if x > 0])

str_list = ['This', 'is', 'a', 'test', 'sentences']
print([x for x in str_list if len(x) >= 3])

print("-------------------------")

# Use ZIP() in LIST generation to access multiple containers at the same time

a = [1, -2, 3, -4, 5]
b = [9, 8, -7, -6, -5]

print([[x, y] for x, y in zip(a, b)])
print([x + y for x, y in zip(a, b)])

print([x + y for x, y in zip(a, b) if x + y >= 0])

print("-------------------------")

# Generate compound LIST using nested LIST generation

a = [1, 2, 3]
b = ['A', 'B']

print([[x, y] for x in a for y in b])

print("-------------------------")

# Default dictionary


letter = ['foo', 'bar', 'pop', 'foo', 'bar', 'foo']
d = defaultdict(int)

for item in letter:
    d[item] += 1
print(d)

print("-------------------------")

# Counter


letter = ['foo', 'bar', 'pop', 'foo', 'bar', 'foo']
c = Counter(letter)

print(c)

for item, counter in c.items():
    print(item, "Occur", counter, "times")

print("Most common occurences: ", c.most_common(1))

print("-------------------------")
