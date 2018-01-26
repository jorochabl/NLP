## Example of Natural Language processing (Sentiment Analysis)

In this example, it's shown how to treat and train various models to analyze user reviews from Amazon. As trainset comments have a user value (from 1 to 5 stars), but we mostly want to know if a comment is positive or negative, we'll reduce it to a binary classification model:
- **1 to 3** --> user review is negative, dislikes the product.
- **4 to 5** --> user review is positive, likes the product.

As text valoration mechanism is based on how frequently meningful words appear, a number of techniques are to be put in place prior to train *sentiment analysys* machine learning models:
- `NLTK` --> toolkit for *stemming* (taking out meaningful, common parts of each word), as well as language detection (stems are different for each language). Stop words and non-words symbols are also put in place to reduce meaningless symbols and words.
- `CountVectorize` --> from `sklearn` library, it's used convert each user review into a sparse matrix, to be passed to machine learning models.
- `sklearn` models --> once pre processed, we'll train and test later on our models, both assuming all text is in Spanish and analyzing each comment's language first (*tokenizer* to split into meaningful word stems accepts a certain language dependant *stemmer*).
- *Dictionaries* --> both to handle functions (stemmers) and parameters / datasets, and to embrace model results for evaluation. Pretty *tricky* but, once reviewd, easy to understand.
