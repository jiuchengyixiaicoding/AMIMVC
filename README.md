# Abstract
Most of the existing multi-view clustering methods are based on the assumption that the data is complete. However, real-world data collection faces various problems. For example, the damage of data storage equipment, limited to the current technology lead to the lack of collection means and so on. Although some incomplete multi-view clustering methods have been proposed, the existing methods mainly have the following two problems: (1) they focus on the consistency of views, but cannot fully mine the consistency of views, and only use the semantic level consistency information, and do not explore the instance-level consistency and complementarity. The combined effect of the two features of views will improve the performance of the model. (2) Self-supervised contrastive learning algorithm mistakenly identifies negative and positive samples, which leads to false negative and false positive samples. These samples mislead the model and have a negative impact on the performance of the model. In order to solve the above problems, this paper proposes Affinity Matrix Guided Multiple Contrastive Learning Incomplete Multi-view Clustering (AMIMVC). First, we use a high-order random walk to construct a kinship matrix, and use the affinity matrix to guide the contrastive learning of the same view and different views at the instance level, the complementarity of views is utilized, and the low-dimensional consistency is mined. The guidance of the affinity matrix in contrastive learning alleviated the problem of false negative and false positive samples recognition. After a large number of experiments, it is confirmed that our proposed method has good performance on both complete views and missing views. After comparing multiple IMVC models, it is confirmed that our model has a significant improvement in performance.
# Model Flowchart
![model_struct](https://github.com/user-attachments/assets/49d9e7ad-6560-4c6a-865a-987339933c4e)

<p>After the incomplete multi-view features are completed by inference evaluation, z<sub>o</sub> and z<sub>t</sub> are encoded by the original encoder and the target encoder for comparison. The original encoded features are mapped to another view space through the cross-view decoder to obtain xr, and z<sub>t</sub> is cross-compared with xr, that is, xr<sup>1</sup> is compared with z<sub>t</sub><sup>2</sup>, and xr<sup>2</sup> is compared with z<sub>t</sub><sup>1</sup>. z<sub>o</sub> is also passed to the clustering module for semantic comparison.</p>
# Method
## Encoder
The view original features and target features are obtained by encoder **E<sub>o</sub><sup>(v)</sup>** and target encoder **E<sub>t</sub><sup>(v)</sup>**, **E<sub>t</sub><sup>(v)</sup>** is the momentum version of **E<sub>o</sub><sup>(v)</sup>**. Specifically **E<sub>t</sub><sup>(v)</sup>**, is not architecturally different from **E<sub>o</sub><sup>(v)</sup>** and self-adjusts using exponential moving average (EMA) of **E<sub>o</sub><sup>(v)</sup>**.
![1750071079607](https://github.com/user-attachments/assets/a9d6a586-0c3f-4df9-9bac-a1dbeef1030b)

For each view v, we transmit instances in small batches to **E<sub>o</sub><sup>(v)</sup>** and **E<sub>t</sub><sup>(v)</sup>** to obtain the corresponding view-specific embeddings, i.e.,

**Z<sub>oj</sub><sup>(v)</sup> = E<sub>o</sub><sup>(v)</sup>(X<sub>i</sub><sup>(v)</sup>)**, &emsp; **Z<sub>tj</sub><sup>(v)</sup> = E<sub>t</sub><sup>(v)</sup>(X<sub>i</sub><sup>(v)</sup>)** &emsp; (1)
## Cross-view Decoders
<p>As mentioned above, in order to solve the problem that only semantic level contrastive learning cannot fully mine the consistency of views, it is not found that contrastive learning of instance-level low-dimensional information can better reflect the essential structure and potential pattern of data, and help to learn the general representation of data, so as to improve the generalization ability of the model and prevent the excessive reinforcement of the consistency of different view embeddings during learning, which leads to the loss of view <a href="#ref12" title="文献引用">complementarity</a>. We employ a special decoder, namely the cross-view decoder. It moderately enforces consistency across views, while preserving the complementary information of views due to its special architectural philosophy. Its special architecture idea is to use the cross-view decoder <span class="formula">F<sup>(a→b)</sup></span> to project the embedding <span class="formula">z<sub>oj</sub><sup>(a)</sup></span> of a view into the embedding space of another view <span class="formula">b</span> and output the reconstructed feature, i.e.,</p>

<!-- 公式部分用纯 HTML 构造，(2) 是公式编号 -->
<p class="formula-equation">
  xr<sub>j</sub><sup>(a→b)</sup> = F<sup>(a→b)</sup>(z<sub>o,j</sub><sup>(a)</sup>) 
  <span class="equation-tag">(2)</span>
</p>


## Instance-level Double Contrastive Learning
<h2>Instance-level Double Contrastive Learning</h2>
<p>
  Due to the unique encoder architecture, we use a special instance-level bi-contrastive loss function accordingly:
</p>
<!-- 公式 L = L_same + L_diff （编号 (3)） -->
<p class="formula">
  ℒ = ℒ<sub>same</sub> + ℒ<sub>diff</sub> 
  <span class="equation-tag">(3)</span>
</p>

<p>
  ℒ<sub>same</sub> and ℒ<sub>diff</sub> represent the intra-view contrast loss and the inter-view contrast loss. 
  These two losses aim to optimize the feature representation of the model by maximizing the similarity between pairs of positive samples 
  and minimizing the similarity between pairs of negative samples. 
  Where Q(d,e) is the cross-entropy function, 
  D ∈ ℝ<sup>N×n</sup> is the pseudo-target (Eq. 8) used to indicate positive and negative sample pairs, 
  and g(k<sub>i</sub>, l<sub>j</sub>) is the pairwise similarity s(k<sub>i</sub>, l<sub>j</sub>) with row normalization operator, that is,
</p>

<!-- 若有公式 (4)，可继续用类似结构扩展 -->
<!-- <p class="formula">
  g(k<sub>i</sub>, l<sub>j</sub>) = ... 
  <span class="equation-tag">(4)</span>
</p> -->

<p>
  If τ is too large, the model pays too much attention to difficult samples. 
  When τ is too small, the loss function is not sensitive to the similarity difference. 
  Therefore, τ is fixed as 0.5 in the experiment.
</p>


## The Affinity Matrix Guides Positive and Negative Pair Identification
# DataSets
<p>In order to prove the performance of our model under datasets of the same type but different sample numbers, we choose Handwritten and MNIST-USPS datasets for experiments.Due to prove the performance of ATIMVC under different types of data sets with increasing sample numbers, we add BDGP and Fashion data sets for experiments.
You can obtain the required dataset by using this link.</p>
https://pan.baidu.com/s/1C194UFYTeF7Qx-Hf4Y67gw 提取码: 9u2y
# Quick Start
python train.py
# Results
<img width="448" alt="table" src="https://github.com/user-attachments/assets/52cd327f-4c2e-44c4-9aa6-52a4ca649707" />
