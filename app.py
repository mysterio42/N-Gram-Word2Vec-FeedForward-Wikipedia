from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from utils.model import top_neighs, load_model, load_embedding,analogy

app = Flask(__name__)
api = Api(app)

W, V = load_model()
word2idx = load_embedding()
idx2word = {i: w for w, i in word2idx.items()}
weight = (W + V.T) / 2


class Neighs(Resource):

    def post(self):
        posted_data = request.get_json()
        assert 'top_k' in posted_data
        assert 'word' in posted_data
        neighs = top_neighs(posted_data['word'], word2idx, idx2word, weight, posted_data['top_k'])
        return jsonify(neighs)

class Analogy(Resource):

    def post(self):

        posted_data = request.get_json()

        assert 'pos1' in posted_data
        assert 'neg1' in posted_data
        assert 'pos2' in posted_data

        rets = analogy(posted_data['pos1'],posted_data['neg1'],posted_data['pos2'],word2idx,idx2word,weight)
        return jsonify(rets)




api.add_resource(Neighs, '/neighs')
api.add_resource(Analogy,'/analogy')
if __name__ == '__main__':
    app.run(host='0.0.0.0')
