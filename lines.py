file_path = 'omniweb data.txt'

print(f"--- Analyzing the first 10 lines of '{file_path}' ---")

try:
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            print(f"Line {i + 1}: {line.strip()}")
except FileNotFoundError:
    print(f"Error: Could not find the file '{file_path}'")