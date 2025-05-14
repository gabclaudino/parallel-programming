import random
import string
import sys

def generate_file(filename, length):
    with open(filename, "w") as f:
        for _ in range(length):
            f.write(random.choice(string.ascii_uppercase))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python3 geraRandom.py <quantidade_de_caracteres>")
        sys.exit(1)
    
    try:
        length = int(sys.argv[1])
    except ValueError:
        print("Por favor, forneça um número válido.")
        sys.exit(1)

    generate_file("fileA.in", length)
    generate_file("fileB.in", length)

    print(f"Arquivos gerados com {length} caracteres cada.")
