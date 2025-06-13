from pathlib import Path

current_file = Path().resolve()
while current_file.name != "jcq_thesis" and current_file != current_file.parent:
    current_file = current_file.parent

BASE_DIR = str(current_file)+ "/"

# BASE_DIR = "/home/jeauscq/Desktop/jcq_thesis/"