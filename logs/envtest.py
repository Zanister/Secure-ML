import os
env = os.environ.copy()
with open("python_env.txt", "w") as f:
    for key, value in env.items():
        f.write(f"{key}={value}\n")
