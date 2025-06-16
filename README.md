# Abstract
Most of the existing multi-view clustering methods are based on the assumption that the data is complete. However, real-world data collection faces various problems. For example, the damage of data storage equipment, limited to the current technology lead to the lack of collection means and so on. Although some incomplete multi-view clustering methods have been proposed, the existing methods mainly have the following two problems: (1) they focus on the consistency of views, but cannot fully mine the consistency of views, and only use the semantic level consistency information, and do not explore the instance-level consistency and complementarity. The combined effect of the two features of views will improve the performance of the model. (2) Self-supervised contrastive learning algorithm mistakenly identifies negative and positive samples, which leads to false negative and false positive samples. These samples mislead the model and have a negative impact on the performance of the model. In order to solve the above problems, this paper proposes Affinity Matrix Guided Multiple Contrastive Learning Incomplete Multi-view Clustering (AMIMVC). First, we use a high-order random walk to construct a kinship matrix, and use the affinity matrix to guide the contrastive learning of the same view and different views at the instance level, the complementarity of views is utilized, and the low-dimensional consistency is mined. The guidance of the affinity matrix in contrastive learning alleviated the problem of false negative and false positive samples recognition. After a large number of experiments, it is confirmed that our proposed method has good performance on both complete views and missing views. After comparing multiple IMVC models, it is confirmed that our model has a significant improvement in performance.
# Model Flowchart
![model_struct](https://github.com/user-attachments/assets/49d9e7ad-6560-4c6a-865a-987339933c4e)

<p>After the incomplete multi-view features are completed by inference evaluation, z<sub>o</sub> and z<sub>t</sub> are encoded by the original encoder and the target encoder for comparison. The original encoded features are mapped to another view space through the cross-view decoder to obtain xr, and z<sub>t</sub> is cross-compared with xr, that is, xr<sup>1</sup> is compared with z<sub>t</sub><sup>2</sup>, and xr<sup>2</sup> is compared with z<sub>t</sub><sup>1</sup>. z<sub>o</sub> is also passed to the clustering module for semantic comparison.</p>

# Method

## Encoder

The view original features and target features are obtained by encoder **E<sub>o</sub><sup>(v)</sup>** and target encoder **E<sub>t</sub><sup>(v)</sup>**, **E<sub>t</sub><sup>(v)</sup>** is the momentum version of **E<sub>o</sub><sup>(v)</sup>**. Specifically **E<sub>t</sub><sup>(v)</sup>**, is not architecturally different from **E<sub>o</sub><sup>(v)</sup>** and self-adjusts using exponential moving average (EMA) of **E<sub>o</sub><sup>(v)</sup>**.

For each view v, we transmit instances in small batches to **E<sub>o</sub><sup>(v)</sup>** and **E<sub>t</sub><sup>(v)</sup>** to obtain the corresponding view-specific embeddings, i.e.,
<div align='center'>
  
![1750071079607](https://github.com/user-attachments/assets/d8d41d3b-5128-4268-b5f2-663dabd7baf8)
</div>

## Cross-view Decoders
<p>As mentioned above, in order to solve the problem that only semantic level contrastive learning cannot fully mine the consistency of views, it is not found that contrastive learning of instance-level low-dimensional information can better reflect the essential structure and potential pattern of data, and help to learn the general representation of data, so as to improve the generalization ability of the model and prevent the excessive reinforcement of the consistency of different view embeddings during learning, which leads to the loss of view <a href="#ref12" title="文献引用">complementarity</a>. We employ a special decoder, namely the cross-view decoder. It moderately enforces consistency across views, while preserving the complementary information of views due to its special architectural philosophy. Its special architecture idea is to use the cross-view decoder <span class="formula">F<sup>(a→b)</sup></span> to project the embedding <span class="formula">z<sub>oj</sub><sup>(a)</sup></span> of a view into the embedding space of another view <span class="formula">b</span> and output the reconstructed feature, i.e.,</p>
<div align='center'>

![1750071504493](https://github.com/user-attachments/assets/4a1bb796-c278-4bba-88f5-147cebe65fec)
</div>

## Instance-level Double Contrastive Learning
<p>
  Due to the unique encoder architecture, we use a special instance-level bi-contrastive loss function accordingly:
</p>
<div align='center'>
  
![1750071545352](https://github.com/user-attachments/assets/7a506f83-83a8-4523-bb41-1034368bd303)
</div>

<p>
  ℒ<sub>same</sub> and ℒ<sub>diff</sub> represent the intra-view contrast loss and the inter-view contrast loss. 
  These two losses aim to optimize the feature representation of the model by maximizing the similarity between pairs of positive samples 
  and minimizing the similarity between pairs of negative samples. 
  Where Q(d,e) is the cross-entropy function, 
  D ∈ ℝ<sup>N×n</sup> is the pseudo-target (Eq. 8) used to indicate positive and negative sample pairs, 
  and g(k<sub>i</sub>, l<sub>j</sub>) is the pairwise similarity s(k<sub>i</sub>, l<sub>j</sub>) with row normalization operator, that is,
</p>
<div align='center'>
  
![1750071759277](https://github.com/user-attachments/assets/a46a5b02-53d2-481f-b79e-f36be5cae225)
</div>
  If τ is too large, the model pays too much attention to difficult samples. 
  When τ is too small, the loss function is not sensitive to the similarity difference. 
  Therefore, τ is fixed as 0.5 in the experiment.
</p>


## The Affinity Matrix Guides Positive and Negative Pair Identification
<p>
  Our affinity matrix guidance method can not only distinguish the samples in the neighborhood according to the affinity 
  but also find the potential high-order neighbors. Fig. 2 shows that our affinity matrix is obtained using the random-walk algorithm 
  <sup>[13]</sup>. Let G be an undirected graph that contains n segments and n nodes. 
  Its random walk matrix is defined as:
</p>

<!-- 公式 (5): Y = A·D⁻¹ -->
<div align='center'>

  ![1750072255039](https://github.com/user-attachments/assets/782e7651-2ed6-455c-be33-1a412cc89bac)

</div>

<p>
  Where A is the adjacency matrix of this graph, A<sub>ij</sub> represents the edge weights, 
  D is the diagonal matrix, and D<sub>ii</sub> represents the sum of the values in the i-th row of A. 
  Y<sub>ij</sub> represents the probability of moving from the i-th node to the j-th node in one step.
</p>

<p>
  Let P(m) be the probability that node i moves to node j at step m. 
  We can deduce that the probability after the random walk at step m can be expressed as follows:
</p>

<!-- 公式 (6): P(m) = P(m-1)·Y = ... = P(0)·Yᵐ -->
<div align='center'>
  
  ![1750072282290](https://github.com/user-attachments/assets/adc69a64-6e74-4beb-986c-965f0bd91209)

</div>

<p>
  Where Y<sup>m</sup> is the m-th power of the mobility probability matrix Y. 
  If the step size m is too small, it will reduce the experimental results; 
  if it is too large, it will reduce the running speed of the model. 
  Therefore, we fix the step size as 5.
</p>

<p>
  We adopt heat kernel similarity <sup>[14]</sup> to define edge weights to build a fully connected affinity graph for in-batch instances, 
  and use the identity matrix to preserve the self-characteristics of views. 
  The heat kernel similarity formula is:
</p>

<!-- 公式 (7): A_ij = exp(-||z_ti^v - z_tj^v||² / η) -->
<div align='center'>
  
  ![1750072308011](https://github.com/user-attachments/assets/f2dd01aa-2956-4708-a477-8ae427503ddd)

</div>

<p>
  Here z<sub>ti</sub><sup>(v)</sup> is the anchor embedding of the i-th node, 
  and z<sub>tj</sub><sup>(v)</sup> is the corresponding negative embedding. 
  We keep η = 0.1 in the experiments.
</p>

<p>
  Each row of the matrix A<sup>(v)</sup> is normalized to ensure that the sum of each row is 1, 
  resulting in the matrix Y<sup>(v)</sup>. We obtain the m-step transition matrix Y<sup>(v^m)</sup>, 
  whose entries Y<sub>ij</sub><sup>(v^m)</sup> represent the probability that node j is a neighbor of anchor i at step m (denoted as FN). 
  We use Y<sup>m</sup> as a pseudo-target for Eq. 1 to achieve robustness to FN, i.e.:
</p>

<!-- 公式 (8): D^v = α·I + (1-α)·I -->
<div align='center'>
  
  
![1750072332751](https://github.com/user-attachments/assets/3f43b7f8-a927-45cf-815b-77479e292943)

</div>

<p>
  I is the identity matrix. Considering the balance, we set α = 0.5 in the experiment.
</p>

# DataSets
<p>In order to prove the performance of our model under datasets of the same type but different sample numbers, we choose Handwritten and MNIST-USPS datasets for experiments.Due to prove the performance of ATIMVC under different types of data sets with increasing sample numbers, we add BDGP and Fashion data sets for experiments.
You can obtain the required dataset by using this link.</p>
https://pan.baidu.com/s/1C194UFYTeF7Qx-Hf4Y67gw 提取码: 9u2y

# Quick Start
python train.py

# Results
<img width="448" alt="table" src="https://github.com/user-attachments/assets/52cd327f-4c2e-44c4-9aa6-52a4ca649707" />

# FastAPI
You can use FastAPI to start the server by running main.py and then calling the model by entering the latitude vector X corresponding to the data set selected by the model, which will return you the prediction Y corresponding to the vector, or you can directly simulate it using request.py provided here.
