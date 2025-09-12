import importlib.metadata
import tiktoken
print("tiktoken version:", importlib.metadata.version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
string = tokenizer.decode(integers)
print(string)

unknow_word = 'Akwirw ier'
integers = tokenizer.encode(unknow_word)
print(integers)
string = tokenizer.decode(integers)
print(string)

