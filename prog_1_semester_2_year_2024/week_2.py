print("Color: ")
color_1 = ['red', 'blue', 'yellow']
print(color_1)
print(type(color_1))
print("-------------------------")

print("Data_1: ")
data_1 = ['apple', 3, [4.5, 'car', True]]
print(data_1)

print(data_1[0])
print(data_1[1])
print(data_1[2])
print(data_1[2][0])
print(data_1[2][2])

print(data_1[-1])
print(data_1[-2])
print(data_1[-3])
print(data_1[-1][-1])
print("-------------------------")

print("String_1: ")
string_1 = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
print(string_1[:])
print(string_1[:3])
print(string_1[3:])
print(string_1[2:5])
print(string_1[::2])
print(string_1[1:6:2])
print("-------------------------")

print("Fruits: ")
fruit_1 = ['Mango', 'Banana', 'Orange']
fruit_1.append('Grape')
print(fruit_1)

fruit_1 += ['Dragon fruits', 'Tomato']
print(fruit_1)

del fruit_1[0]
print(fruit_1)

fruit_1.remove('Tomato')
print(fruit_1)

print("-------------------------")

# set
capitals_1 = {"Jakarta", "Bangkok", "Kuala Lumpur",
              "Washington DC", "Tokyo", "Seoul"}
print(capitals_1)

# dictionary
language_1 = {"name": "Python", "version": "3.12",
              "types": ["int", "float", "str", "bool"]}
print(language_1)
print(language_1["name"])
print(language_1["types"])

language_1["name"] = "Lua"
print(language_1)

# update data into dictionary
language_1.update({"Created by": "Guido van Rossum", "Used by": "Developers"})
print(language_1)

# delete data from dictionary
del language_1["Used by"]
print(language_1)

print("-------------------------")

# while loop
n = 0
# while n < 5:
#    print(n)
#    n += 1

# if while loop
while True:
    print(n)
    n += 1
    if n >= 5:
        break

print("-------------------------")

# while continue loop
while n < 10:
    n += 1
    if n % 3 == 0:
        continue
    print(n)

print("-------------------------")

# for loop

animal_list_1 = ['dog', 'cat', 'monkey', 'bird', 'elephant']
for animal in animal_list_1:
    print(animal)

print("-------------------------")

# for loop using list

fruits_1 = [['apple', 'red'], ['banana', 'yellow'], ['guava', 'green']]
for fruit in fruits_1:
    for item in fruit:
        print(item)

print("-------------------------")

# for range loop
print(list(range(5)))

# looping through list of data
animal_list_1 = ['dog', 'cat', 'monkey', 'bird', 'elephant']
for index in range(5):
    print("item", index, '=', animal_list_1[index])

print("-------------------------")

# enumerate => a built-in function in python that allows you to keep track of the number of iterations (loops) in a loop.
print(list(enumerate(animal_list_1)))

animal_list_2 = ['dog', 'cat', 'monkey', 'bird', 'elephant']
for index, item in enumerate(animal_list_2):
    print('item', index, '=', item)

print("-------------------------")

# zip => used to combine two or more iterable dictionaries into a single iterable,
# where corresponding elements from the input iterable are paired together as tuples.
index_list_1 = ['a', 'b', 'c', 'd', 'e']
animal_list_3 = ['dog', 'cat', 'monkey', 'bird', 'elephant']
print(list(zip(index_list_1, animal_list_3)))

fruits_1 = ['apple', 'peach', 'banana', 'guava', 'papaya']
colors_1 = ['red', 'pink', 'yellow', 'green', 'orange']

for name, color in zip(fruits_1, colors_1):
    print(name, 'is', color)

print("-------------------------")

# dictionary data types

fruits_2 = {
    'apple': 'red',
    'peach': 'pink',
    'banana': 'yellow',
    'guava': 'green',
    'papaya': 'orange',
    'strawberry': 'red'
}

print(list(fruits_2.items()))

for name, colors in fruits_2.items():
    print(name, 'is', color)
