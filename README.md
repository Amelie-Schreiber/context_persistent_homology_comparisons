# context_persistent_homology_comparisons
Comparing different models' preservation of persistent homology. 

In these notebooks we:

1. Choose a keyphrase, multiword expressions, or collocation.
2. Produce a list of contexts (texts) containing the keyphrase, multiword expressions, or collocation exactly once. 
3. Compute the context vectors for the keyphrase, multiword expressions, or collocation in each context for each head in each layer of the first model, and for the second model. 
4. Compute the persistent homology of the context vectors of the keyphrase, multiword expressions, or collocation for the first model and for the second model. 
5. Compute the Wasserstein distances between each pair of persistence diagrams for the first model (that is across all contexts), and for each attention head, and then do the same for the second model. 
6. Compare the distance matrix obtained in this way for each head of the first model to the distance matrix obtained in this way for each head of the second model. 
7. Compute the percentage of entries below the diagonal that are negative, effectively computing what percentage of the time the second model has higher Wasserstein distance compared to the first model. 
8. Compute statistics on the array of probabilities to determine the minimum, maximum, median, mean, variance, standard deviation, and quantiles to determine just how well one model preserves persistent homology compared to the other.

We note that `dicta-il/alephbertgimmel-base` outperforms all Hebrew and multilingual models at preserving persistent homology. 

## Some explanation of the notebooks

Towards the end of each notebook, we create a $144 \times 144$ array (or more accurately a $12 \times 12 \times 12 \times 12$ array, since there are $12$ layers with $12$ heads in each layer) to store the percentages indication what percentage of entries of the matrices `w_distances[layer_1][head_1] - w_distances_2[layer_2][head_2]` are negative. Here, we are comparing each head's Wasserstein distances in one model, that is `w_distances[layer_1][head_1]` to each head's Wasserstein distances in a second model, that is `w_distances_2[layer_2][head_2]`. This effectively compares how well the persistent homology is preserved by one `head` in the first model, to how well the persistent homology is preserved by a second `head` in a second model. 

If the percentage of negative entries is high in the matrix `w_distances[layer_1][head_1] - w_distances_2[layer_2][head_2]`, then the matrix `w_distances_2[layer_2][head_2]` is larger in more places than not, indcating that the second model's Wasserstein distances are higher much more often than the first model's Wasserstein distances. This means the second model is *worse* at preserving persistent homology. On the other hand, if the percentage of the entries in `w_distances[layer_1][head_1] - w_distances_2[layer_2][head_2]` that are negative is low, then the first model's attention head outperforms the second model's attention head at preserving the persistent homology. Doing this for every `head` in every`layer`, w get the $12 \times 12 \times 12 \times 12$ array `percentages`. Querying `percentages` by calling `percentages[layer_1][head_1][layer_2][head_2]` for a heads in specific layers gives us the percentage of entries below the diagonal in `w_distances[layer_1][head_1] - w_distances_2[layer_2][head_2]` that are negative. 

Running some basic statistics on the array `percentages` we get numbers such as the following for `TurkuNLP/wikibert-base-he-cased` as the first model, and `dicta-il/alephbertgimmel-base` as the second model:

```
Min:  0.0
Max:  90.47619047619048
Mean:  4.00201168430335
Median:  0.0
Standard Deviation:  9.971154321626278
25th percentile:  0.0
75th percentile:  4.761904761904762
```

From this we can tell that a large majority of the time the second model `dicta-il/alephbertgimmel-base` outperforms the first model `TurkuNLP/wikibert-base-he-cased` at preserving the persistent homology of the phrase "לצפות בזרחת השמש" in the various contexts given by each `text[i]`. This is because the mean and median are both near zero, indicating almost no entries below the diagonal of each matrix `w_distances[layer_1][head_1] - w_distances_2[layer_2][head_2]` are negative. That is, almost all entries below the diagonal are positive, indicating `w_distances_2[layer_2][head_2]` is smaller entrywise than `w_distances[layer_1][head_1]`. This means the Wasserstein distances between persistence diagrams computed by `dicta-il/alephbertgimmel-base` are *smaller*. Note however, there are some heads of the first model that outperform the second model, as can be seen by the `Max: 90.47619047619048`. 

Recall, the Wasserstein distances were computed for each pair of persistence diagrams for a fixed keyphrase (such as collocations, idioms, or multiword expression) placed in different contexts `text[i]`. This determines how well the model preserves the persistent homology of the keyphrase as it is placed in different contexts. While the context vectors of the keyphrase may change (as contextual embedding must do in order to be contextual), the relative distances between them, and thus their persistent homology, may be well preserved by a language model. In this case we find that `dicta-il/alephbertgimmel-base` does in fact outperform every other model it is compared to. We were unable to compare `xlm-roberta-large` to `dicta-il/alephbertgimmel-base` due to computational resource constraints, so this would be very straightforward next step to take. 

## Applications to Video

It seems as though computing persistence diagrams for context vectors associated to image patches for a text-to-video transformer model would allow us to improve the temporal coherence of the video produced by the model, by including the Wasserstein distance between persistence diagrams computed for consetutive frames. This is a very exciting potential application, still being explored. For a very similar idea applied to the token probability distributions (computed by applying softmax to individual attention matrices), please see [this repo](https://github.com/Amelie-Schreiber/emergent_topology_of_ideas_in_vision). 


## Applications to Language

### Morpheme Trees and Syntax Trees

We expect the simplex trees of persistent homology to mimic the behavior of the morpheme trees found in [Morphological Segmentation Inside-Out](https://arxiv.org/abs/1911.04916v2), as well as [constituency or dependency sentence parse trees](https://en.wikipedia.org/wiki/Parse_tree). This behavior is expected *on average*, and including a topological prior into a loss function for the model may improve performance on downstream tasks such as morphological segmentation. For example, we would want to have the simplex tree have the same structure as the morpheme tree for a character level transformer. This would be a loose constraint on the persistent homology, as it would not specify exactly *when* vertices representing characters merge to creat simplicial complexes, or at what scale those would then merge into each other to create more complicated words with multiple morphemes, only that they would merge in a particular order that is determined by the morpheme tree of a word. Similarly, we would want the simplex tree to mimic the constituency or dependency parse tree, where sub-simplicial ccomplexes representing words merge in a grammatically meaningful way according to the sentence's parse tree. 

### Anomaly Detection

We might also take Fréchet means of persistence diagrams of "normal text" and compare this to new potentially "anomalous text" to detect anomalies. This is done in [these notebook](https://github.com/Amelie-Schreiber/anomaly_detection_persistent_homology/tree/main). 

### Topic Modeling

Similar to BERTopic with [hierarchical topic modeling](https://maartengr.github.io/BERTopic/getting_started/hierarchicaltopics/hierarchicaltopics.html) using the hierarchical clustering method HDBSCAN, we can replace the HDBSCAN with persistent homology. This would provide a new take on topic modeling that is worth investigating further. 


