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

