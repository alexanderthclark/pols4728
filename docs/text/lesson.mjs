import {pelotonBagStages,pelotonTfStages} from './peloton-lesson.mjs?v=20261108-peloton-1';
import {topicModelChapter} from './topic-model-lesson.mjs?v=20261108-peloton-1';
// Scene copy stays brief; readingText and readingNote retain the continuous explanation.
export const chapters = [
  {
    "id": "ai-writing",
    "title": "AI writing",
    "subtitle": "An opening question",
    "stages": [
      {
        "title": "Does this sound like AI?",
        "text": "",
        "note": "",
        "visual": "ai-example",
        "readingText": "Sam Kriss asked ChatGPT to write a science-fiction story. His New York Times Magazine essay quotes this description of its hero, Kael. Read the excerpt, then scroll to reveal some features you might notice.",
        "readingNote": "A recognizable pattern gives us a hypothesis to test across texts."
      },
      {
        "title": "What caught your attention?",
        "text": "Notice the contrast phrase, em dash, and ghost imagery.",
        "note": "These suggest features to test across texts, not proof of AI use.",
        "visual": "ai-example",
        "readingText": "The contrast phrase sets up an ordinary description and replaces it with something more dramatic. An em dash makes the turn explicit, while the ghost imagery adds intensity. These are possible features to investigate.",
        "readingNote": "The highlights describe this example. A comparison corpus would tell us whether such features help predict AI use."
      },
      {
        "title": "Turn features into counts.",
        "text": "We could count words, punctuation, or recurring phrases. Start with words.",
        "note": "",
        "visual": "ai-example",
        "readingNote": "Begin with the words themselves. Next, the cartoon makes that representation literal.",
        "readingText": "We can count a word such as phantom, retain punctuation to count em dashes, or preserve a phrase to count a recurring construction. Each choice keeps different information about the text."
      }
    ]
  },
  {
    "id": "bag-of-words",
    "title": "Bag of words",
    "subtitle": "Documents → vocabulary counts",
    "stages": [
      {
        "title": "Count the word frequencies.",
        "text": "",
        "note": "",
        "visual": "speech",
        "readingText": "Imagine catching each word of a speech in a bag. Our Lincoln-like speaker says “government of the people, by the people, for the people.” Scroll to send each separate word from the speaker to the bag.",
        "readingNote": "The fragment is from the Gettysburg Address. Scroll slowly to follow individual words into the bag."
      },
      {
        "title": "Keep repetition.",
        "text": "One row for the speech; one column per word.",
        "note": "",
        "visual": "speech-counts",
        "readingText": "The bag contains ten tokens but only six distinct terms. “The” and “people” each occur three times. Counts retain repetition while discarding where each word appeared.",
        "readingNote": "A bag-of-words vector records one count for each vocabulary term."
      },
      {
        "title": "But lose word order.",
        "text": "",
        "readingText": "Counts preserve repetition, but lose word order. “Senators support voters” and “Voters support senators” produce identical counts. Their meanings differ because the roles reverse. Counting can still be useful, but this representation cannot distinguish these sentences.",
        "note": "Same counts; different meanings.",
        "visual": "order",
        "readingNote": "Bigrams can retain some local order, at the cost of a larger vocabulary."
      },
      {
        "title": "Sentences 1 and 2 express similar ideas.",
        "text": "",
        "note": "",
        "visual": "short-sentences",
        "readingText": "The reverse problem also occurs: different words can express similar ideas. Compare three short sentences: 1. “Plebiscites are unwise.” 2. “Democracy is foolish.” 3. “Majority rule rules.” The first two criticize forms of democratic decision-making; the third praises majority rule. Can word counts capture that resemblance?",
        "readingNote": "The first two express a similar negative stance, though they are not exact synonyms."
      },
      {
        "title": "Count these words, too.",
        "text": "Lowercase and remove punctuation.",
        "note": "",
        "visual": "short-counts",
        "readingText": "Use the vocabulary plebiscites, are, unwise, democracy, is, foolish, majority, rule, rules, in that order. Each sentence contains three words, each occurring once. Their count vectors are \\(\\mathbf{x}_1=(1,1,1,0,0,0,0,0,0)\\), \\(\\mathbf{x}_2=(0,0,0,1,1,1,0,0,0)\\), and \\(\\mathbf{x}_3=(0,0,0,0,0,0,1,1,1)\\). No vocabulary coordinate is nonzero in more than one sentence.",
        "readingNote": "For this example, we only lowercase and remove punctuation. We do not lemmatize or remove stop words: “are” and “is” remain separate terms, as do “rule” and “rules.”"
      },
      {
        "title": "Compare with cosine similarity.",
        "text": "These counts miss the first two sentences’ similar stance.",
        "note": "",
        "visual": "short-cosine",
        "readingText": "Cosine similarity compares vector direction: \\(\\cos(\\mathbf{x}_i,\\mathbf{x}_j)=\\frac{\\mathbf{x}_i\\cdot\\mathbf{x}_j}{\\lVert\\mathbf{x}_i\\rVert\\lVert\\mathbf{x}_j\\rVert}\\). Each vector has length \\(\\sqrt{3}\\). No pair shares a nonzero coordinate, so every pairwise dot product is zero and every pairwise cosine similarity is zero. Define cosine distance as \\(d_{\\cos}(\\mathbf{x}_i,\\mathbf{x}_j)=1-\\cos(\\mathbf{x}_i,\\mathbf{x}_j)\\). Thus \\(d_{\\cos}(\\mathbf{x}_1,\\mathbf{x}_2)=d_{\\cos}(\\mathbf{x}_1,\\mathbf{x}_3)=1\\): bag of words places the first sentence equally far from the other two.",
        "readingNote": "Word overlap misses the shared negative stance of sentences 1 and 2. Different wording can obscure similar meanings, just as identical counts can obscure different meanings in the senators/voters example."
      },
      ...pelotonBagStages
    ]
  },
  {
    "id": "tf-idf",
    "title": "TF–IDF",
    "subtitle": "Counts → distinctive term weights",
    "stages": pelotonTfStages
  },
  {
    "id": "authorship",
    "title": "Application: Authorship",
    "subtitle": "Word counts as evidence",
    "application": true,
    "stages": [
      {
        "title": "Common words reveal writing habits.",
        "text": "Mosteller and Wallace (1963) compared frequencies to investigate the disputed Federalist essays.",
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
        "note": "Evidence from many words favored Madison.",
        "visual": "federalist",
        "readingText": "The usefulness of a word depends on the research question. Authorship and topical comparison ask different questions of word counts: common function words can carry stylistic information, so removing them automatically could discard the signal. In the classic Federalist Papers study, Mosteller and Wallace (1963) compared common-word frequencies in Hamilton’s and Madison’s writings. Their combined evidence favored Madison for the twelve disputed essays. Also discussed in Grimmer, Roberts, and Stewart (2022).",
        "readingNote": "The chart shows approximate reported rates of “upon” in their comparison writings, including material outside the Federalist Papers. One word did not settle the case. Compare rates or account for length when texts differ in size. Counting words cannot by itself separate authorship from genre and topic. Feature selection and weighting should follow the research question."
      }
    ]
  },
  {
    "id": "bertopic",
    "title": "Application: BERTopic",
    "subtitle": "From documents to topics",
    "stages": [
      {
        "title": "Use word counts to describe groups.",
        "text": "BERTopic groups embeddings, then pools each group’s words.",
        "note": "A separate example with two supplied groups.",
        "visual": "topic-pooling",
        "links": [
          {
            "label": "BERTopic",
            "url": "https://maartengr.github.io/BERTopic/algorithm/algorithm.html"
          }
        ],
        "readingNote": "We now use a separate synthetic corpus with four short texts in two supplied groups. These constructed texts are separate from the Peloton transcript comparison. The standard BERTopic workflow reduces embedding dimensions before clustering; its subsequent bag-of-words representation describes the resulting groups. The illustration supplies the groups to isolate that second task.",
        "readingText": "BERTopic first groups document embeddings into clusters. It then pools each cluster’s documents into one bag of words. We now weight a word within a whole cluster, rather than within an individual document."
      },
      {
        "title": "Then weight the pooled words.",
        "text": "Use class-based TF–IDF to rank words within each group.",
        "note": "The denominator counts occurrences, not documents.",
        "visual": "topic-words",
        "links": [
          {
            "label": "class-based TF–IDF",
            "url": "https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html"
          }
        ],
        "readingText": "BERTopic’s class-based TF–IDF assigns a weight to a word–cluster pair. Its default formula is \\(\\operatorname{cTFIDF}(t,c)=\\frac{f(t,c)}{L_c}\\ln\\!\\left(1+\\frac{A}{f(t,\\mathrm{all})}\\right)\\). Here \\(f(t,c)\\) counts the word’s occurrences in the pooled cluster, \\(L_c\\) is that cluster’s total token count, \\(A\\) is the average pooled cluster length (truncated to an integer in the implementation), and \\(f(t,\\mathrm{all})\\) counts the word’s occurrences across all clusters. This denominator counts occurrences, not documents or clusters containing the word. In this constructed example, “budget” occurs twice among cluster A’s six tokens, and twice across both clusters. With average cluster length six, its weight is \\(\\frac{2}{6}\\ln(1+6/2)\\approx0.462\\). This describes “budget” in cluster A; it is not a score for any one original document. BERTopic uses the highest-weighted words to describe each topic.",
        "readingNote": "Here “budget” in cluster A has weight 0.462. Both clusters have six tokens, so their average length is also six."
      }
    ],
    "application": true
  },
  topicModelChapter
];
