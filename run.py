from models.net import FeedForward
from models.embedding import WikipediaEmbedding

if __name__ == '__main__':
    wk_emb = WikipediaEmbedding(vocab_size=20000)
    wk_emb.build()
    model = FeedForward(emb=wk_emb)
    model.train()
