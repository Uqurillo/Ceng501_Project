# Higher-Order Interaction Goes Neural: A Substructure Assembling Graph Attention Network for Graph Classification

This readme file is an outcome of the [CENG501 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/DL/) project for reproducing a paper without an implementation. See [CENG501 (Spring 42) Project List](https://github.com/CENG501-Projects/CENG501-Fall2024) for a complete list of all paper reproduction projects.

# 1. Introduction

This paper is written by Jianliang Gao, Jun Gao, Xiaoting Ying, Mingming Lu and Jianxin Wang and published in IEEE Transactions on Knowledge and Data Engineering where volume, issue and date are 35, 2 and 1 February 2023, respectively. 

Data can come in very different forms such as point clouds or graphs. Numerous applications in cheminformatics or social networks use classification of graphs to get results. Hence, the existence of Neural Networks working on graph-structured data is essential and Graph Neural Networks (GNNs) covers this need. However, existing GNN models tries to capture the information of first-order neighboring nodes within a single layer and mostly don't give enough importance to graph substructure and substructure information. In this paper, the proposed method SA-GAT not only investigate and extract linear information from higher-order substructures but also focus on their non-linear interaction information via a core module "Substructure Interaction Attention" (SIA). 

My aim is to fully understand the method and obtain the same experimental results by providing open code for the sake of community.

## 1.1. Paper summary

The paper introduces a method contributing to Graph Neural Networks to classify graphs. It can be seen as a function from the input space of graphs to the set of graph labels. The aim is to learn this function making loss smallest along the network. The existing literature takes substructures into account, but this process usually just covers the immediate neighbors for each node within a single layer. Passing to higher-order substructures by increasing number of layers enables receptive field to enlarge, but it can lead to failure of convergence or decrease in performance. Later works including k-GNN and SPA-GAT enable considering higher-order substructures within a single layer. However, they fail to investigate mutual influence among them. This paper introduces SIA to eliminate this deficiency and uses Max and SSA Pooling to extract local and global feature embeddings of graphs.

# 2. The method and our interpretation

## 2.1. The original method

The method has four different parts. It starts with the input and higher-order layer. We are given a graph $G$ which is a pair $(V,E)$ with a set of nodes $V$ and a set of edges $E$ where $E \subseteq \\{ (i,j) | i,j \in V, i \neq j \\} $. $V(G)$ and $E(G)$ denote the set of nodes and edges, respectively. Moreover, define the neighborhood of a node $i$ as $N(i) = \\{ j | (i,j) \in E(G) \\}$ and show the feature embedding encoding attributes of $i$ as $u_i$. After starting with such a graph $G$, substructures of $G$ which are $1$-order substructures are considered and they give $1$-order graph which is $G$ itself. To extract more information about $G$, higher-order graphs are defined in the following way.

**Definition.** For an integer $k \geq 2$, we denote any $k$ different connected nodes forming a connected subgraph in $G$ as $C_k=\\{v_1,\ldots,v_k\\}$. We identify $C_k$ as a node in $k$-order graph. $V(G)^k$ is denoted as the set of all nodes of $k$-order graph. The neighborhood of the node $C_k$ is defined as:
$$N(C_k)=\\{T_k \in V(G)^k | |C_k \bigcap T_k| = k-1\\}.$$

With this definition, we can create higher-order graphs for $k \geq 2$. The next step is to initialize node features of 1-order and higher order graphs. For $i \in V(G)$, the feature embedding $u_i \in ℝ^d$ is the concatanation of two one-hot vectors $e_i \in ℝ^{d_1}$ and $a_i \in ℝ^{d_2}$ based on label and attributes of the node $i$, respectively. Note that $d=d_1+d_2$. For node $C_k$ where $k \geq 2$,

```math
u(C_k)=\frac{1}{k}\sum_{C_1 \in C_k} u(C_1),
```
that is, $u(C_k)$ is just the average of feature embeddings of the nodes that constitute it.

Second step of the method includes Substructure Interaction Attention (SIA) Layer. The main aim is to train $u(C_k)$ for each substructure $C_k \in V(G)^k$ and it is done with the contribution of two parts: the neighbor structure aggregation (sa) and the neighbor interaction aggregation (ia). Let

```math 
u(C_k)^{'}_{sa} \textrm{ and } u(C_k)^{'}_{ia}
```

denote the new representation of $u(C_k)$ obtained from neighbor aggregation and neighbor interaction aggregation, respectively.

### Neighboring Substructure Aggregation

To create it, we need to take all feature embeddings of neighbors of the node $C_k$ into account using attention mechanism:

```math 
u(C_k)^{'}_{sa} = \sum_{T_k \in N(C_k)} \alpha(C_k,T_k) W_1 u(T_k) + \alpha(C_k,C_k) W_1 u(C_k),
```

where $W_1 \in ℝ^{d' x d}$ is a shared weight matrix with the desired dimension $d'$ of $u(C_k)^{'}_{sa}$ and $\alpha(C_k,T_k)$ is the attention coefficient which is computed as follows:

```math 
\alpha(C_k,T_k) = \frac{exp(f(u(C_k),u(T_k)))}{\sum_{T_k \in N(C_k) \cup {C_k}} exp(f(u(C_k),u(T_k)))},
```

where $f$ is the following feed-forward network with a single hidden layer:

```math 
f(u(C_k),u(T_k))= a^{T} tanh(W_2U+b),
```

where $a \in ℝ^d$ is the weight vector to obtain a scalar as a result of applying $f$ and $W_2 \in ℝ^{r x 2d}$ is the weight matrix, $U \in ℝ^{2d x 1}$ is the matrix obtained by first putting each entry of $u(C_k)$ in rows by obeying the order of it, and then doing same process for $u(T_k)$. Moreover, $b \in ℝ^d$ is the bias and $r$ is the dimension size of the hidden layer.

### Neighboring Substructure Interaction Aggregation

The neighbor interaction representation of node $C_k$ is given as:

```math 
u(C_k)^{'}_{ia} = \sum_{T_k \in N(C_k)} \sum_{S_k \in N(C_k)} \beta(T_k,S_k)(u(T_k)*u(S_k)),
```

where $*$ is the element-wise multiplication operator and $\beta(T_k,S_k)$ denotes the interaction coefficient between nodes $T_k$ and $S_k$. If $(T_k,S_k) \in E(G)^k$, we define $\beta(T_k,S_k)=\alpha(C_k,T_k)\alpha(C_k,S_k)$. Otherwise, it equals to 0.

Instead, their normalized versions $\beta(T_k,S_k)^*$ can be used to make coefficients easily comparable:

```math 
\beta(T_k,S_k)^* = \frac{\beta(T_k,S_k)}{\sum_{M_k,Q_k \in N(C_k),(M_k,Q_k) \in E(G)^k} \beta(M_k,Q_k)}.
```

### Combining Two Parts

The new representation of $u(C_k)$ which is denoted as $u(C_k)'$ combining neighbor information and neighbor interaction information is defined as:

```math 
u(C_k)' = \sigma(\alpha u(C_k)^{'}_{sa} + (1-\alpha)u(C_k)^{'}_{ia}), 
```

where $\alpha$ is a parameter to balance information coming from two parts.

## 2.2. Our interpretation

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
