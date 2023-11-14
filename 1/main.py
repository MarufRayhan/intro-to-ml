import sys

# Read command line arguments and convert to a list of integers
arr = sys.argv[1].split(',')
my_numbers = [None]*len(arr)

for idx, arr_val in enumerate(arr):
    my_numbers[idx] = int(arr_val)

# Print
print(f'Before sorting {my_numbers}')

# My solution in Bubble Sort
index_length = len(my_numbers)
is_sorted = False

while not is_sorted:
    is_sorted = True
    for i in range(0, index_length-1):
        if my_numbers[i] > my_numbers[i+1]:
            is_sorted = False
            my_numbers[i], my_numbers[i+1] = my_numbers[i+1], my_numbers[i]

print(f'After sorting {my_numbers}')