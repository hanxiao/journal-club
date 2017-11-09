# journal-club

## 09.11.2017
### Learning to Respond with Deep Neural Networks for Retrieval-Based Human-Computer Conversation System
**Author:** Rui Yan

**Link:** http://www.ruiyan.me/pubs/SIGIR2016.pdf

**Motivation:** proposed deep learning-to-respond framework for open-domain chatbot. Single query may not fully convey user intention in multi-turn conversations. Thus, they also use context to build reformulated query and then do retrieval.

**Approach:** different query reformulation strategies using context, 
- no context {q0}
- whole context (all sentences concate with current query), {q0, q0+C}
- add-one, add one sentence at time {q0, q0+c1, q0+c2, ...}.
- dropout, leave one out {q0, q0+C\c1, q0+C\c2, ...}
- combined, all above.

network: word-embedding -> bi-LSTM, convolution on the output of LSTM, max pooling, hinge rank loss

![](assets/dd248394.png)

unrelated, but i will keep it here since it introduce symmetric rank loss.
![](assets/301cb280.png)

**Evaluation:** p@1, MAP, nDCG, MRR. Neural Responding Machine, DeepMatch, ARC. 

## 08.11.2017
### On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima
**Author:** Nitish Shirish Keskar, Dheevatsa Mudigere

**Link:** https://arxiv.org/pdf/1609.04836.pdf

**Motivation:** Large batch is worse (in terms of generalization ability), but why? This paper investigates the numerical evidence: small-batch converges to flat-minima whereas large-batch converges to sharp-minima.

![](assets/5b612ab4.png)
**Evaluation:** Numerical experiment on real and artificial data.


**Conclusion:**
1. LB methods tend to be attracted to minimizers close to the starting point, whereas SB methods move away and locate minimizers that are farther away.
2. noise in the SB gradient pushes the iterates out of the basin of attraction of sharp minimizers and encourages movement towards a flatter
   minimizer where noise will not cause exit from that basin. In LB,  the noise in the stochastic gradient is not sufficient to cause ejection
   from the initial basin leading to convergence to sharper a minimizer.
3. Solution: the use of dynamic sampling where the batch size is increased gradually as the iteration progresses
   (Byrd et al., 2012; Friedlander & Schmidt, 2012). high testing accuracy is achieved using a large-batch method that is warm-start with a small-batch method.


## 07.11.2017
### SWISH: A Self-Gated Activation Function
**Author:** Prajit Ramachandran, etc. (Google Brain)

**Link:** https://arxiv.org/abs/1710.05941

**Motivation:** Relu is the default activation function in NN now. The author proposed a "Swish" activation function, i.e. f(x) = x * sigmoid(x) that consistently improving over different datasets.

![](assets/e4910474.png)
![](assets/f7ba4e45.png)

**Approach:** Swish is a smooth, non-linear, non-monotonic, unbounded (above) function, its parameterized version f(x) = 2x*sigmoid(beta*x) is a interpolate between linear and ReLU function. Based on following observations, they proposed Swish:
1. Unboundedness is good because it avoids saturation (unlike sigmoid and tanh)
2. Bounded below is good because it lets the network forgets large negative values. Bounded below zero improves the gradient flow
3. Smoothness is good because it is more traversable, reducing sensitivity to the initialization and learning rate.


**Evaluation:** Swish outperforms Relu in nearly all datasets/batch size/number of layers. On MNIST it is able to use Swish train very deep networks without residual connections. 



## 06.11.2017
### Beyond triplet loss: a deep quadruplet network for person re-identification
**Author:** Weihua Chen, Xiaotang Chen, etc.

**Link:** https://arxiv.org/pdf/1704.01719.pdf

**Motivation:** Quadruplet loss produces larger inter-class variation and smaller intra-class variation, tends to generalize better on the test set. 

**Approach:** optimize two aspects simultaneously, 1) the distance of pairs from same class (they call probe) should be smaller than from other class. If anchor is B1, relevant to B3, then B1B3 (intra-class) < B1A3 (inter-class). 2) the distance of negative pairs should be greater than any other pairs from the same class, i.e. B1A3 > C1C2, or equivalently B1B3 (intra-class) < C1A3 (inter-class). where C1A3 are all irrelevant to B1B3

They propose a margin-based sampling approach to sample difficult negative examples for training. 

**Algorithm:**
Triplet loss with l2-norm metric and hinge loss:![](assets/214c08b5.png), a more general is to learn the metric, e.g. the distance function g in ![](assets/db37a4ac.png), they proposed ![](assets/60393ecd.png)

![](assets/2728ca7a.png)

**Evaluation:** 
Top-1 classification accuracy (rank-1) on CUHK01, CUHK03, VIPeR

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
