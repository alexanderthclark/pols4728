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
        "title": "Choose what counts as a word.",
        "text": "Here we lowercase the text and split it into alphabetic tokens, removing punctuation. “The” and “the” become the same term. These choices define what differences our analysis can see.",
        "note": "We keep common words and do not stem words. Other research questions may call for different choices.",
        "visual": "tokens"
      },
      {
        "title": "Give every document the same coordinates.",
        "text": "Each row is a document; each column is a vocabulary term. D1 has two instances of “budget” and three of “the.” The same coordinates let us compare all four documents.",
        "note": "Most cells in a realistic document–term matrix are zero. Scroll back to retrace how the tokens became counts.",
        "visual": "matrix",
        "readingText": "Each column is a vocabulary term; each row is a document. A cell counts how often that term occurs. D1 has two instances of “budget” and three of “the.” Most cells in a realistic document–term matrix are zero.",
        "readingNote": "Scroll back: the words return to their original positions. The count representation has one coordinate for every vocabulary term."
      },
      {
        "title": "The bag loses word order.",
        "text": "“Senators support voters” and “Voters support senators” produce identical counts. Their meanings differ because the roles reverse. Counting can still be useful, but this representation cannot distinguish these sentences.",
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
        "title": "Compare Revelation with the Gospel of John.",
        "text": "Erwin and Oakes (2012) analyzed 500-word Greek samples using 75 common words from the Johannine corpus. Their correspondence analysis found distinct profiles for Revelation and the Gospel of John.",
        "note": "Distinct word profiles do not by themselves isolate authorship; genre also changes word use.",
        "visual": "bible"
      },
      {
        "title": "Choose features for the question.",
        "text": "For authorship, common function words can carry stylistic information. For topical comparison, we may instead emphasize terms that distinguish documents. Return to our four political texts to see how that changes their representation.",
        "note": "Compare rates or account for length when texts differ in size. A model cannot remove genre and topic differences simply by counting words.",
        "visual": "purpose",
        "readingText": "For authorship, frequent function words can carry stylistic information. Removing them automatically could discard the signal. For topical comparison, we may instead want to emphasize terms that distinguish documents."
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
        "text": "D1 contains “the” three times. Yet all four documents contain it. To compare topics, a term’s repetition within a document and its prevalence across documents answer different questions.",
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
