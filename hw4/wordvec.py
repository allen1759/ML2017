# -*- coding: utf-8 -*-

import word2vec
import numpy as np
import nltk


def train():
    # DEFINE your parameters for training
    MIN_COUNT = 10
    WORDVEC_DIM = 500
    WINDOW = 5
    NEGATIVE_SAMPLES = 5
    ITERATIONS = 10
    MODEL = 0
    LEARNING_RATE = 0.01

    # train model
    word2vec.word2vec(
        train='hp/all.txt',
        output='hp/model.bin',
        cbow=MODEL,
        size=WORDVEC_DIM,
        min_count=MIN_COUNT,
        window=WINDOW,
        negative=NEGATIVE_SAMPLES,
        iter_=ITERATIONS,
        alpha=LEARNING_RATE,
        verbose=True)


def test():
    plot_num = 750
    
    # load model for plotting
    model = word2vec.load('hp/model.bin')

    vocabs = []                 
    vecs = []                   
    for vocab in model.vocab:
        vocabs.append(vocab)
        vecs.append(model[vocab])
    vecs = np.array(vecs)[:plot_num]
    vocabs = vocabs[:plot_num]

    '''
    Dimensionality Reduction
    '''
    # from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(vecs)


    '''
    Plotting
    '''
    import matplotlib.pyplot as plt
    from adjustText import adjust_text

    # filtering
    use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
    puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]
    
    
    plt.figure(figsize=(8, 6), dpi=80)
    texts = []
    for i, label in enumerate(vocabs):
        pos = nltk.pos_tag([label])
        if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
                and all(c not in label for c in puncts)):
            x, y = reduced[i, :]
            texts.append(plt.text(x, y, label))
            plt.scatter(x, y)

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

    plt.savefig('hp.png', dpi=600)
    #plt.show()

    

# %%

if __name__ == "__main__":
    
    train()
    
# %%
    
    test()
