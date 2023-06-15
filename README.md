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
