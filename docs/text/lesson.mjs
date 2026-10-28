// Scene copy stays brief; readingText and readingNote retain the continuous explanation.
export const chapters = [
  {
    "id": "ai-writing",
    "title": "AI writing",
    "subtitle": "An opening question",
    "stages": [
      {
        "title": "Does this sound like AI?",
        "text": "Sam Kriss asked ChatGPT to write a science-fiction story. His New York Times Magazine essay quotes this description of its hero, Kael. What features of the sentence might we measure?",
        "note": "A recognizable pattern gives us a hypothesis to test across texts.",
        "visual": "ai-example",
        "readingText": "Sam Kriss asked ChatGPT to write a science-fiction story. His New York Times Magazine essay quotes this description of its hero, Kael. Read the excerpt, then scroll to reveal some features you might notice."
      },
      {
        "title": "What caught your attention?",
        "text": "The contrast phrase sets up an ordinary description and replaces it with something more dramatic. An em dash makes the turn explicit, while the ghost imagery adds intensity. These are possible features to investigate.",
        "note": "The highlights describe this example. A comparison corpus would tell us whether such features help predict AI use.",
        "visual": "ai-example"
      },
      {
        "title": "Turn a hunch into measurable features.",
        "text": "We can count a word such as phantom, retain punctuation to count em dashes, or preserve a phrase to count a recurring construction. Each choice keeps different information about the text.",
        "note": "Begin with the simplest feature: how often each word appears. The cartoon makes that representation literal.",
        "visual": "ai-example",
        "readingNote": "Begin with the words themselves. Next, the cartoon makes that representation literal."
      }
    ]
  },
  {
    "id": "bag-of-words",
    "title": "Bag of words",
    "subtitle": "Documents → vocabulary counts",
    "stages": [
      {
        "title": "Catch the words as they are spoken.",
        "text": "Imagine catching each word of a speech in a bag. Our Lincoln-like speaker says “government of the people, by the people, for the people.” Follow the words into the bag.",
        "note": "The fragment is from the Gettysburg Address. Scroll slowly to follow individual words into the bag.",
        "visual": "speech",
        "readingText": "Imagine catching each word of a speech in a bag. Our Lincoln-like speaker says “government of the people, by the people, for the people.” Scroll to send each separate word from the speaker to the bag."
      },
      {
        "title": "Keep repetition. Let the order go.",
        "text": "The bag contains ten tokens but only six distinct terms. “The” and “people” each occur three times. Counts retain repetition while discarding where each word appeared.",
        "note": "A bag-of-words vector records one count for each vocabulary term.",
        "visual": "speech-counts"
      },
      {
        "title": "Begin with four short documents.",
        "text": "We can now build a bag for each document. These four constructed political texts are small enough to inspect directly. A shared vocabulary will let us compare their counts.",
        "note": "D1–D4 are teaching examples, not a sample of parliamentary speech.",
        "visual": "corpus",
        "readingText": "Suppose we want to compare what political texts discuss. These four constructed documents are small enough to inspect word by word. A computer needs a representation that lets the same operation compare every pair."
      },
      {
        "title": "Tokenize the text.",
        "text": "Tokenization splits text into units to count. Here our tokens are alphabetic words; we also lowercase them and drop punctuation. D1 yields eight tokens, with repetitions kept.",
        "note": "Tokens need not be whole words: we can use characters, subword pieces, or phrases such as “New York.”",
        "visual": "tokens",
        "readingText": "Tokenization splits a text into units called tokens. For these English examples, each uninterrupted sequence of letters is a token. We also lowercase the text and remove punctuation, so “The” and “the” count as the same term. D1 contains eight tokens, including three occurrences of “the” and two of “budget.” Each occurrence is a token; the shared vocabulary will contain each distinct term once.",
        "readingNote": "Tokens need not be whole words. Character tokens retain individual letters and symbols; subword tokens split words into smaller pieces; a multiword expression such as “New York” can be kept as one unit. Token boundaries, lowercasing, and punctuation handling are separate decisions. Retaining punctuation would be necessary to count the em dashes in our opening example. The research question determines which distinctions should survive."
      },
      {
        "title": "Group inflections with lemmatization.",
        "text": "",
        "note": "This step is optional. Our worked matrix keeps the original word forms.",
        "visual": "lemma",
        "readingText": "Lemmatization maps an inflected word to its dictionary form, or lemma. For example, “debated” and “debating” can be counted under “debate,” and “senators” under “senator.” Pooling these forms can make a representation less sparse, but it also removes distinctions in how the text was written. Choosing a lemma can depend on part of speech and context. Stemming is a different approach that typically uses rules to strip word endings; its output need not be a dictionary word.",
        "readingNote": "A lemma groups inflections, not all related words: “senate” and “parliament” remain different terms. We do not lemmatize or stem our four example documents."
      },
      {
        "title": "Decide whether to remove stop words.",
        "text": "Stop words are terms deliberately excluded from an analysis, often frequent words such as “the,” “and,” and “of.” Removing “the” here leaves five tokens. But common words can carry information about writing style.",
        "note": "There is no universal stop-word list. We retain these words in our worked matrix.",
        "visual": "stopwords",
        "readingText": "A stop-word list specifies terms to exclude from a representation. Such lists often contain frequent grammatical words such as “the,” “and,” and “of,” which may contribute little to a topical comparison. In D1, removing the three occurrences of “the” would leave five tokens. This is a choice about the information we want to retain, not a required cleaning step. Common words can help distinguish writing habits, and removing a negation such as “not” can erase a consequential difference in meaning.",
        "readingNote": "There is no universal list that suits every language, corpus, or research question. Our worked matrix retains all the tokenized words, including “the”; the Federalist example will show why common words can be useful."
      },
      {
        "title": "Give every document the same coordinates.",
        "text": "Each row is a document; each column is a vocabulary term. D1 has two instances of “budget” and three of “the.” The same coordinates let us compare all four documents.",
        "note": "We retain stop words and do not lemmatize. Most cells in a realistic document–term matrix are zero.",
        "visual": "matrix",
        "readingText": "For our worked example, we lowercase the texts and retain alphabetic tokens, without removing stop words or applying lemmatization. Each column is a vocabulary term; each row is a document. A cell counts how often that term occurs. D1 has two instances of “budget” and three of “the.” Most cells in a realistic document–term matrix are zero.",
        "readingNote": "Scroll back: the words return to their original positions. The count representation has one coordinate for every vocabulary term."
      },
      {
        "title": "The bag loses word order.",
        "text": "",
        "readingText": "“Senators support voters” and “Voters support senators” produce identical counts. Their meanings differ because the roles reverse. Counting can still be useful, but this representation cannot distinguish these sentences.",
        "note": "Bigrams can retain some local order, at the cost of a larger vocabulary.",
        "visual": "order"
      },
      {
        "title": "Small words can reveal writing habits.",
        "text": "Word counts can reveal writing habits. Mosteller and Wallace (1963) compared common-word frequencies in Hamilton’s and Madison’s writings. Their combined evidence favored Madison for the twelve disputed Federalist essays.",
        "links": [
          {
            "label": "Mosteller and Wallace (1963)",
            "url": "https://doi.org/10.1080/01621459.1963.10500849"
          },
          {
            "label": "Grimmer, Roberts, and Stewart (2022)",
            "url": "https://bstewart.scholar.princeton.edu/publications/text-data-new-framework-machine-learning-and-social-sciences"
          }
        ],
        "note": "These approximate “upon” rates include comparison writings outside the Federalist Papers. One word did not settle the case.",
        "visual": "federalist",
        "readingText": "In the classic Federalist Papers study, Mosteller and Wallace (1963) compared common-word frequencies in Hamilton’s and Madison’s writings. Their combined evidence favored Madison for the twelve disputed essays. Also discussed in Grimmer, Roberts, and Stewart (2022).",
        "readingNote": "The chart shows approximate reported rates of “upon” in their comparison writings, including material outside the Federalist Papers. One word did not settle the case."
      },
      {
        "title": "From writing style to subject matter.",
        "text": "The Federalist example keeps common words because they reveal style. To compare topics, we may instead give more weight to terms that distinguish documents. TF–IDF does this using their prevalence across our corpus.",
        "note": "We keep the same vocabulary; TF–IDF changes weights rather than applying a fixed stop-word list.",
        "visual": "purpose",
        "readingText": "Authorship and topical comparison ask different questions of the same counts. In the Federalist example, common function words carry stylistic information, so removing them automatically could discard the signal. To compare what documents discuss, we may instead emphasize terms that distinguish them from other texts in the corpus. TF–IDF combines repetition within a document with prevalence across documents to assign each term a weight. We will keep the same vocabulary and change the weights, rather than remove terms using a fixed stop-word list.",
        "readingNote": "Compare rates or account for length when texts differ in size. Counting words cannot by itself separate authorship from genre and topic. Feature selection and weighting should follow the research question."
      }
    ]
  },
  {
    "id": "tf-idf",
    "title": "TF–IDF",
    "subtitle": "Counts → distinctive term weights",
    "stages": [
      {
        "title": "A frequent word need not be distinctive.",
        "text": "Return to D1: “the” occurs three times, but it also appears in every other document. “Budget” occurs in only two. For topical comparison, repetition within a document and prevalence across documents tell us different things.",
        "note": "We keep the same four documents and vocabulary. First follow “the,” then “budget,” in D1.",
        "visual": "tf",
        "readingNote": "The corpus stays fixed throughout this chapter. We first inspect “the” in D1, then follow “budget” through document frequency, inverse document frequency, and TF–IDF weighting."
      },
      {
        "title": "Count documents, not occurrences.",
        "text": "Document frequency, \\(\\operatorname{df}(t)\\), counts documents containing a term. “Budget” occurs twice in D1, but D1 contributes only one to its document frequency. It appears in two of our four documents.",
        "note": "Term frequency, \\(\\operatorname{tf}(t,d)\\), is the raw count within a document. For “budget” in D1, it is 2.",
        "visual": "df",
        "readingText": "Document frequency, \\(\\operatorname{df}(t)\\), counts how many documents contain a term at least once. “Budget” occurs twice in D1, but D1 contributes only one to its document frequency. It appears in two of the four documents.",
        "readingNote": "Term frequency, \\(\\operatorname{tf}(t,d)\\), is the raw count within document \\(d\\) in this example."
      },
      {
        "title": "Rarity becomes an inverse-frequency weight.",
        "text": "A term appearing in every document receives zero inverse-frequency weight. “Budget” appears in two of four documents, so its weight is \\(\\ln(2)\\approx0.693\\). “Passed” appears in one, giving \\(\\ln(4)\\approx1.386\\).",
        "note": "This is one unsmoothed convention. Alternatives change the formula, term-frequency scaling, and normalization.",
        "visual": "idf",
        "readingText": "We use \\(\\operatorname{idf}(t)=\\ln\\!\\left(N/\\operatorname{df}(t)\\right)\\), where \\(N\\) is the number of documents. A term in every document gets \\(\\ln(1)=0\\). “Budget” gets \\(\\ln(2)\\approx0.693\\); “passed,” found in just one document, gets \\(\\ln(4)\\approx1.386\\)."
      },
      {
        "title": "Multiply within-document count by rarity.",
        "text": "Multiply each count by its term’s inverse-frequency weight. In D1, “budget” receives \\(2\\times0.693\\approx1.386\\), while “the” receives zero. The coordinates still name vocabulary terms; only their weights change.",
        "note": "This weighting targets distinctiveness in this corpus. Rare terms are not automatically meaningful, and typos can be rare too.",
        "visual": "weighted",
        "readingText": "The TF–IDF weight is \\(\\operatorname{tf}(t,d)\\times\\operatorname{idf}(t)\\). In D1, “budget” receives \\(2\\times0.693\\approx1.386\\). “The” receives \\(3\\times0=0\\). The coordinates still name vocabulary terms; only their weights have changed."
      },
      {
        "title": "Similarity still depends on shared words.",
        "text": "Cosine similarity compares vector direction, adjusting for vector length. But “senate” and “parliament” remain separate coordinates. Reweighting counts does not learn that words used in similar contexts may be related.",
        "note": "For predictive evaluation, learn the vocabulary and IDF weights on training documents, then apply them to held-out texts.",
        "visual": "cosine",
        "readingText": "Cosine similarity compares vector direction, dividing the dot product by both vector lengths. But “senate” and “parliament” remain separate coordinates. TF–IDF alone does not learn that words used in similar contexts may be related."
      }
    ]
  }
];
