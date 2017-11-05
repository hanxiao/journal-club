# journal-club
## 05.11.2017
### Learning Deep Structured Semantic Models for Web Search using Clickthrough Data
**Author:** Po-Sen Huang, Xiaodong He, etc.

**Link:** https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/DSSM_cikm13_talk_v4.pdf

**Motivation:** Keyword-based matching often fails due to the lack of semantic level interpretation. They proposed a DL model to put query and document into a common low-dim space in a supervised way (i.e. using clickthorugh data).

**Approach:** Use DNN to map high-dimensional sparse text features into low-dim dense features in a semantic space. The relevance score between query and doc is measured by cosine. For word, they proposed a "word hashing" method, breaking down the word into character-level ngrams. 

They simply use (siamese) DNN for both doc and query, no fancy architecture in this version. 

**Algorithm:** minimize the following function:
![](assets/570556d6.png), where
![](assets/1860408b.png), and 
![](assets/6d60fb16.png)

**Evaluation:** Test how model can generalize from the popular URL-clicks to tail or new URLs. Baselines: TF-IDF, BM25, WTM, LSA, PLSA. Bilingual topic model with posterior regularization, discriminative projection model.

 

## 04.11.2017

### Do Deep Nets Really Need to be Deep?
**Author:** Lei Jimmy Ba, Rich Caruana

**Link:** https://arxiv.org/pdf/1312.6184.pdf

**Motivation:** exploring the source of improvement of DNN model, is it really because of the deeper topology? 

**Goal:** showing that shallow net *with the same number of parameters* can performs as good as deep net.

**Approach:** instead of learning a shallow model directly from the data, the author trains a complex model first and uses it as a "teacher" to teach shallow "student".

**Algorithm:** student learns from the logit output of the teacher (more information compare to softmax output), and then minimizing the square loss (take it as a regression problem). Namely,
![](assets/9b421263.png)
where ![](assets/cb89e723.png) is the model prediction. 

To speed up, the student decompose $W$ with two low rank matrix, namely.
![](assets/7179bbf9.png)

**Evaluation:** train different networks on speech recognition data and use the best model (ensemble of DNN) as teacher to train shallow model. 
