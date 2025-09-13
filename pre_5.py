import torch
from pre_4 import GPTDatasetV1
#将单词 id 转换为连续的向量表示，即所谓的令牌嵌入

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
print(embedding_layer(torch.tensor([3])))
print(embedding_layer(input_ids))


vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
max_length = 4
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
dataloader = GPTDatasetV1.create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

#使用嵌入层将令牌id 嵌入到256 维向量中
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)                            #每个id 都嵌入为256 维向量


#GPT-2 使用绝对位置嵌入，创建另外一个嵌入层
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))         #torch.arange(max_length) 会生成一个从0到 max_length-1 的张量，即 [0, 1, 2, 3] ，这代表了序列中的位置索引
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
