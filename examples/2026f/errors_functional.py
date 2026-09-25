def square(error):
    return error ** 2

errors = [1, -2, 3]
total = sum(map(square, errors))
print(total)  # 14
