from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
KM = KMeans(n_clusters=2)
model = Word2Vec.load("C:\\Users\\Administrator\\Desktop\\word2vec\\Word60.model") #加载训练好的模型
corpus = []
B = model.wv.index2word  #获取word2vec训练过的词汇
gb = open('./data/动物词库.txt',encoding='utf-8').readlines()
for word in gb[:30]:    #为了方便，每个词库只取了前面30个单词
    word = word.split('\t')
    if word[0] in B:
        corpus.append(word[0])

fb = open('./data/地名词库.txt',encoding='utf-8').readlines()
for word in fb[:30]:
    word = word.split('\t')
    if word[0] in B:
        corpus.append(word[0])
vector = model[corpus]
vector_ = pca.fit_transform(vector)
y_ = KM.fit_predict(vector_)
plt.rcParams['font.sans-serif'] = ['FangSong']
plt.scatter(vector_[:,0],vector_[:,1],c=y_)
for i in range(len(corpus)):    #给每个点进行标注
    plt.annotate(s=corpus[i], xy=(vector_[:, 0][i], vector_[:, 1][i]),
                 xytext=(vector_[:, 0][i] + 0.1, vector_[:, 1][i] + 0.1))
plt.show()
