from interpreter import NeuralInterpreter
from engine import NousModel

def test_strings_io():
    model = NousModel()
    interpreter = NeuralInterpreter(model)
    
    print("=== Testing String Manipulation ===")
    code_strings = """
s1 = "Hello"
s2 = "World"
joined = f"{s1}, {s2}!"
upper = joined.upper()
parts = upper.split(',')
return parts[0]
"""
    try:
        res = interpreter.execute(code_strings)
        print(f"String Result: {res} (Expected 'HELLO')")
    except Exception as e:
        print(f"String Failed: {e}")

    print("\n=== Testing File I/O ===")
    code_io = """
# Try to write to a file
with open('test_output.txt', 'w') as f:
    f.write("Nous wrote this!")

# Read it back
with open('test_output.txt', 'r') as f:
    content = f.read()
return content
"""
    try:
        res = interpreter.execute(code_io)
        print(f"I/O Result: {res} (Expected 'Nous wrote this!')")
    except Exception as e:
        print(f"I/O Failed: {e}")

if __name__ == "__main__":
    test_strings_io()
