# Unsupervised Learning
- Unsupervised learning is a type of machine learning that involves training a model on a dataset without labels. The goal of unsupervised learning is to find patterns in the data and to learn the underlying structure of the data.  
## Dimensionality Reduction
- Dimensionality reduction is the process of reducing the number of random variables under consideration, by obtaining a set of principal variables. It can be divided into feature selection and feature extraction.  

- Feature selection is the process of selecting a subset of relevant features for use in model construction. It is used to reduce the number of input variables to those that are believed to be most useful to a model in order to predict the target variable.
    e.g. Recursive Feature Elimination (RFE), L1 regularization (Lasso), Random Forest Feature Importance, etc.  

- Feature extraction is the process of transforming the data in such a way that it can be represented by a smaller number of features. It is used to reduce the dimensionality of the data while retaining as much of the information as possible.
    e.g. Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), t-distributed Stochastic Neighbor Embedding (t-SNE), Uniform Manifold Approximation and Projection (UMAP), etc.  

- Dimensionality reduction can be achieved by using techniques such as Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), t-distributed Stochastic Neighbor Embedding (t-SNE), and Uniform Manifold Approximation and Projection (UMAP). 
      e.g. PCA is a technique for reducing the dimensionality of the data by projecting it onto a lower-dimensional subspace that maximizes the variance of the data. It is used to identify patterns in data and to detect the correlation between variables.
        
- PCA is a technique for reducing the dimensionality of the data by projecting it onto a lower-dimensional subspace that maximizes the variance of the data. It is used to identify patterns in data and to detect the correlation between variables.
    
    ```python
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X)
    ```

- LDA is a technique for reducing the dimensionality of the data by projecting it onto a lower-dimensional subspace that maximizes the separation between classes. It is used to find the linear combinations of features that best separate the classes in the data.  
    ```python
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X, y)
    ```
  
- t-SNE is a technique for reducing the dimensionality of the data by projecting it onto a lower-dimensional space that preserves the local structure of the data. It is used to visualize high-dimensional data in a lower-dimensional space.  
    ```python
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    ```  
  
- UMAP is a technique for reducing the dimensionality of the data by projecting it onto a lower-dimensional space that preserves the global structure of the data. It is used to visualize high-dimensional data in a lower-dimensional space while preserving the global structure of the data.  
    ```python
    import umap
    reducer = umap.UMAP(n_components=2)
    X_umap = reducer.fit_transform(X)
    ```  
  
- Dimensionality reduction is useful for data visualization, data compression, and feature selection. It can help to reduce the computational cost of training machine learning models and to improve the performance of the models by removing irrelevant features.  
