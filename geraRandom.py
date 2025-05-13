import random
import string

def generate_file(filename, length):
    with open(filename, "w") as f:
        for _ in range(length):
            f.write(random.choice(string.ascii_uppercase))

generate_file("fileA.in", 100000)
generate_file("fileB.in", 100000)
