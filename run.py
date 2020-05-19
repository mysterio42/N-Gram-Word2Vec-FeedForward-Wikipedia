from models.net import FeedForward
from models.embedding import WikipediaEmbedding
from utils.data import to_csv
from utils.model import load_model, load_embedding,top_neighs,analogy
import argparse


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str2bool, default=True, required=True,
                        help='True: Load trained model  False: Train model default: True')

    parser.print_help()

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.load:

        W, V = load_model()
        word2idx = load_embedding()
        idx2word = {i: w for w, i in word2idx.items()}
        for idx,We in enumerate((W, (W + V.T) / 2)):
            neighs = top_neighs('king', word2idx, idx2word, We)
            solution = analogy('world', 'city', 'population', word2idx, idx2word, We)
            print(solution)
            print(neighs)
            to_csv(f'neighs{idx}','king',neighs)


    else:
        wk_emb = WikipediaEmbedding(vocab_size=1000)
        wk_emb.build()

        model = FeedForward(emb=wk_emb)
        model.train()
