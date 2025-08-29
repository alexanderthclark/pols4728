(learning)=
# Learning

## What is Machine Learning?

I sometimes find the term "machine learning" to be frustrating. It sounds too impressive if you've already been running regressions to fulfill the modern duty to be data driven. Similarly, the use of words like "model" might frustrate you if you have a theorist's understanding of a model as more complete description of behavior. But we come to machine learning on machine learning's terms, and this requires some language acquisition. Let's first get our arms around the idea of "learning" and "machine learning," with the help of two foundational figures in the field.

Computer scientist Marvin Minsky offered the definition below, though with the preface that it was "too broad to have much use." 

> "Learning is making useful changes in our minds."  
> — {cite}`minsky1986society`

Computer scientist Tom Mitchell, in his widely cited textbook, provides a structured definition. 

> "Machine Learning is the study of computer algorithms that improve automatically through experience."  
> — {cite}`mitchell1997machine`

Mitchell's definition is common in the field and easily coexists with the idea of a well-posed learning problem, which we will introduce shortly. And there are less precise, but still useful definitions, like the one below from political scientists.

> "Machine learning is a class of flexible algorithmic and statistical techniques for prediction and dimension reduction."  
> — {cite}`grimmer2021machine`

With these rough boundaries, we can discuss learning problems and their key similarities in taking data, "learning" from it to produce a line in the case of linear regression, or a fitted model more abstractly, all guided by some measure of what makes that the best fit. But how do we know when a problem is actually suitable for machine learning? Mitchell offers a more structured framework for identifying well-posed learning problems.

## Well-Posed Learning Problems

**Definition:** A computer program is said to **learn** from experience *E* with respect to some class of tasks *T* and performance measure *P*, if its performance at tasks in *T*, as measured by *P*, improves with experience *E* {cite}`mitchell1997machine`.

This definition provides a framework for thinking about learning problems, though it can be somewhat narrow. Not all useful ML techniques fit perfectly into this framework, as we will see. 


### Example: Wage Prediction

Consider a system that predicts wages based on education level:

- **Task T**: Predicting wages from years of education
- **Performance measure P**: Mean squared error (MSE) between predicted and actual wages
- **Training experience E**: Observing education-wage pairs from survey data

If we observe $i=1,\dots,n$ and make prediction $\hat{y}_i$ for all $i$, our performance measure is $\frac{1}{n}\sum_{i=1}^{n} (y_i -\hat{y}_i)^2$. If we use a linear model, $\hat{y}_i=\hat{\beta}_0 + \hat{\beta}_1x_i$, then we are describing ordinary least squares. 

### Example: Playing Board Games

AlphaGo was trained to play Go by teaching itself. It used both supervised learning and self-play reinforcement learning.

- **Task T**: Playing the board game Go
- **Performance measure P**: Percent of games won against opponents
- **Training experience E**: Playing practice games against itself and against humans

Note that while AlphaGo's performance P is measured by win percentage, the actual training process optimizes a different objective that correlates with winning. P is an *evaluation metric* and not necessarily what is optimized during training.

Famously, AlphaGo would go on to defeat the reigning grandmaster of Go in four of five matches. This was celebrated by the AI community and even Rick Rubin {cite}`rubin2023creative`. The training experience, playing against itself, was advantageous because AlphaGo discovered a unique and unexpected way to play. 

## Types of Learning Tasks


We won't encounter many learning problems similar to playing Go in the social sciences. Instead, we'll see a lot of applications more similar to the wage prediction problem. Wage prediction is a case of *supervised learning*. 

Textbooks like {cite}`james2023introduction` often introduce a *supervised* vs *unsupervised* dichotomy. We will follow that convention, but note that more complete taxonomies might include reinforcement learning or semi-supervised learning.


### Supervised Learning

Supervised learning problems come with *supervising output* or we might say our data is labeled: we have both the inputs $x$ and the observed output $y$. 

- **Classification Tasks**: Predict discrete categories (e.g., spam vs. not spam)
- **Regression Tasks**: Predict continuous values (e.g., wages, house prices)

Linear regression is the most common form of supervised learning. Tree-based models, neural networks, and many more models fall under this category.

### Unsupervised Learning

Unsupervised learning problems come with no labels. This creates a different challenge: instead of predicting a known outcome, we're trying to discover structure in the data itself.

For example, you might have a large number of posts from a social media platform or the responses to a survey question. Consider the task of categorizing these inputs based on the topic. You have roughly three options.

1. Convert this to a supervised problem by hand-labeling some responses (using human coders or an LLM), then treating this as a classification task.

2. Use unsupervised methods to discover natural groupings in the responses. 

3. Use a hybrid approach where unsupervised methods suggest categories that humans then refine.

Possible unsupervised tasks include:

* Clustering: Grouping similar observations (e.g., identifying types of political donors based on giving patterns)
* Dimensionality reduction: Simplifying complex data (e.g., reducing high-dimensional voting data for legislators to a position on a left-right spectrum)
* Topic modeling: Discovering themes (e.g., finding emerging policy concerns in constituent emails)

Notice how these tasks don't fit neatly into Mitchell's framework - what exactly is the "performance measure P" for discovering topics? We might evaluate whether the topics seem interpretable to domain experts, but there's no single correct answer to optimize against.

## Performance Measures

Performance measures quantify how well a model performs its task. While often chosen without much thought, this choice can be consequential.

### Binary Classification Example

Judges make jail-or-release decisions. Based on the data available at the time of a hearing, a judge decides whether or not a defendant is granted pretrial release (whether on bail or release on recognizance). The judge's decision is to be based on a prediction of whether or not the defendant would fail to appear in court or be rearrested if released. This is the learning task of the judge, or perhaps of a machine when considering algorithmic decisions (and as studied in {cite}`kleinberg2018human`).

Note that in this classification problem, we define "positive" as high risk—that is, a defendant who would fail to appear or be rearrested if released.

The performance measure is slightly ambiguous, but any reasonable measure will weigh the types of errors. When making predictions, there are four possible outcomes:

|                        | **Predicted: High Risk** | **Predicted: Low Risk** |
|------------------------|--------------------------|-------------------------|
| **Actually: High Risk** | ✓ True Positive         | ✗ False Negative (Type II) |
| **Actually: Low Risk**  | ✗ False Positive (Type I) | ✓ True Negative        |

This matrix reveals a challenge: how do we combine these four numbers into a single performance score?

Some options include:
- Focus on overall accuracy
- Weight some errors more than others
- Consider multiple metrics (but how do we compare models?)
- Account for the cost of errors (but costs to whom and are the costs measurable?)

There's no correct answer. The choice of performance measure is itself a value judgment about what matters. The weight of the value judgment is clear in this example, whereas the nuances are easier to ignore if you're a marketer deciding who to email. 

Common performance measures in classification include

1. **Precision**  
   $\text{Precision} \;=\; \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}$
   > Of all the defendants the model predicted as high risk (would FTA or be rearrested), how many actually did?

2. **Recall** (aka True Positive Rate or Sensitivity)  
   $ \text{Recall} \;=\; \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}} $ 
   > Of all defendants who actually were high risk (would FTA or be rearrested), how many did the model correctly flag?

3. **F-1 Score**  
   Harmonic mean of Precision and Recall
   
   $ F_1 \;=\; 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $

4. **ROC curve & AUC**  
   The ROC curve plots the True Positive Rate, or Recall, vs the False Positive Rate as you vary a decision threshold. This is useful when you have a model that outputs a probability of being high risk and then you have to choose a threshold. The False Positive Rate is $\frac{\text{False Positives}}{\text{True Negatives + False Positives)}}$. The ROC curve always starts at (0,0) and ends at (1,1). As we gradually lower the classification threshold, more cases get labeled positive. This increases the True Positive Rate, but it also increases the False Positive Rate A perfectly random model traces a diagonal line, while a better model bows toward the top-left corner, achieving high TPR with low FPR. The area under the curve, **AUC**, summarizes the performance. If the area is 1, the model is perfect. If the area is 0.5, you could do the same from random guessing. See {cite}`fawcett2006roc`.


#### More Concrete

Let's imagine we have data on 5 defendants where we magically know what would have happened if they were all released:
- 2 would have failed to appear (FTA) or been rearrested
- 3 would have shown up to court and stayed out of trouble

Now let's compare three different models:

**Model A: The Oracle (Perfect)**
|                        | **Predicted: High Risk** | **Predicted: Low Risk** |
|------------------------|-------------------------|------------------------|
| **Actually would FTA** | 2 | 0 |
| **Actually wouldn't FTA** | 0 | 3 |

- Precision = 2/2 = **100%**
- Recall = 2/2 = **100%**
- F1 = **100%**

Perfect! But this is a nirvana no human can achieve.

**Model B: Nancy Grace (High Recall)**
|                        | **Predicted: High Risk** | **Predicted: Low Risk** |
|------------------------|-------------------------|------------------------|
| **Actually would FTA** | 2 | 0 |
| **Actually wouldn't FTA** | 2 | 1 |

- Precision = 2/4 = **50%**
- Recall = 2/2 = **100%**
- F1 = **66.7%**

This model catches everyone who would FTA, but detains 4 people total—2 unnecessarily.

**Model C: Alvin Bragg (High Precision)**
|                        | **Predicted: High Risk** | **Predicted: Low Risk** |
|------------------------|-------------------------|------------------------|
| **Actually would FTA** | 1 | 1 |
| **Actually wouldn't FTA** | 0 | 3 |

- Precision = 1/1 = **100%**
- Recall = 1/2 = **50%**
- F1 = **66.7%**

This model is perfectly selective—when it says someone is high risk, it's always right. But it only catches half of people who would FTA.

Which model is best? Model B protects public safety but jails more innocent people. Model C minimizes false imprisonment but lets half the risks go free. Notice that both models achieve the same F1 score (66.7%), yet they make very different tradeoffs. The F1 score treats precision and recall as equally important, but is that the right assumption for this decision? The answer is not in a machine learning textbook.


#### All Possible Classification Outcomes

With just 5 defendants, there are only 32 possible ways to classify them (2^5 = 32). The table below shows every possible prediction pattern and the resulting performance metrics. Each row represents a different classifier, with the binary string showing predictions for each defendant (1 = predict high risk, 0 = predict low risk).

<a id="all-classifiers-table"></a>
<div style="width: 100%; text-align: center;">
<style type="text/css">
#T_36472  {
  border-collapse: separate;
  border-spacing: 2px;
  margin: 25px auto;
  font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 14px;
  display: table;
}
#T_36472 th {
  background-color: #f0f0f0;
  color: #000000;
  padding: 10px 12px;
  border: 1px solid #ccc;
  text-align: center;
  font-weight: 600;
}
#T_36472 td {
  padding: 8px 10px;
  border: 1px solid #ccc;
  text-align: center;
}
#T_36472 caption {
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 1rem;
  font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
}
#T_36472_row0_col0, #T_36472_row1_col0, #T_36472_row2_col0, #T_36472_row3_col0, #T_36472_row4_col0, #T_36472_row5_col0, #T_36472_row6_col0, #T_36472_row7_col0, #T_36472_row8_col0, #T_36472_row9_col0, #T_36472_row10_col0, #T_36472_row11_col0, #T_36472_row12_col0, #T_36472_row13_col0, #T_36472_row14_col0, #T_36472_row15_col0, #T_36472_row16_col0, #T_36472_row17_col0, #T_36472_row18_col0, #T_36472_row19_col0, #T_36472_row20_col0, #T_36472_row21_col0, #T_36472_row22_col0, #T_36472_row23_col0, #T_36472_row24_col0, #T_36472_row25_col0, #T_36472_row26_col0, #T_36472_row27_col0, #T_36472_row28_col0, #T_36472_row29_col0, #T_36472_row30_col0, #T_36472_row31_col0 {
  text-align: center;
}
#T_36472_row0_col1, #T_36472_row0_col2, #T_36472_row1_col1, #T_36472_row2_col1, #T_36472_row3_col1, #T_36472_row4_col2, #T_36472_row5_col1, #T_36472_row6_col1, #T_36472_row7_col1, #T_36472_row8_col2, #T_36472_row9_col1, #T_36472_row31_col2 {
  background-color: #fff5eb;
  color: #000000;
  text-align: center;
}
#T_36472_row0_col3, #T_36472_row1_col3, #T_36472_row1_col4, #T_36472_row2_col3, #T_36472_row2_col4, #T_36472_row3_col3, #T_36472_row3_col4, #T_36472_row5_col3, #T_36472_row6_col3, #T_36472_row7_col3, #T_36472_row9_col3, #T_36472_row10_col4, #T_36472_row11_col4, #T_36472_row12_col4, #T_36472_row13_col4, #T_36472_row14_col4, #T_36472_row15_col4, #T_36472_row27_col4, #T_36472_row29_col4, #T_36472_row30_col4 {
  background-color: #37a055;
  color: #f1f1f1;
  text-align: center;
}
#T_36472_row0_col4, #T_36472_row4_col4, #T_36472_row8_col4, #T_36472_row31_col4 {
  background-color: #00441b;
  color: #f1f1f1;
  text-align: center;
}
#T_36472_row0_col5, #T_36472_row0_col6, #T_36472_row0_col7, #T_36472_row1_col6, #T_36472_row2_col6, #T_36472_row3_col6, #T_36472_row4_col3, #T_36472_row4_col5, #T_36472_row5_col4, #T_36472_row5_col6, #T_36472_row6_col4, #T_36472_row6_col6, #T_36472_row7_col4, #T_36472_row7_col6, #T_36472_row8_col3, #T_36472_row8_col5, #T_36472_row9_col6, #T_36472_row10_col3, #T_36472_row11_col3, #T_36472_row12_col3, #T_36472_row13_col3, #T_36472_row14_col3, #T_36472_row15_col3, #T_36472_row16_col3, #T_36472_row16_col4, #T_36472_row17_col3, #T_36472_row17_col4, #T_36472_row18_col3, #T_36472_row18_col4, #T_36472_row19_col3, #T_36472_row19_col4, #T_36472_row20_col3, #T_36472_row20_col4, #T_36472_row21_col3, #T_36472_row21_col4, #T_36472_row22_col3, #T_36472_row23_col3, #T_36472_row25_col4, #T_36472_row26_col4, #T_36472_row28_col4 {
  background-color: #aedea7;
  color: #000000;
  text-align: center;
}
#T_36472_row1_col2, #T_36472_row2_col2, #T_36472_row3_col2, #T_36472_row4_col1, #T_36472_row8_col1, #T_36472_row10_col1, #T_36472_row10_col2, #T_36472_row11_col1, #T_36472_row11_col2, #T_36472_row12_col1, #T_36472_row12_col2, #T_36472_row13_col1, #T_36472_row13_col2, #T_36472_row14_col1, #T_36472_row14_col2, #T_36472_row15_col1, #T_36472_row15_col2, #T_36472_row16_col1, #T_36472_row17_col1, #T_36472_row18_col1, #T_36472_row19_col1, #T_36472_row20_col1, #T_36472_row21_col1, #T_36472_row22_col1, #T_36472_row23_col1, #T_36472_row27_col2, #T_36472_row29_col2, #T_36472_row30_col2 {
  background-color: #fdb97d;
  color: #000000;
  text-align: center;
}
#T_36472_row1_col5, #T_36472_row2_col5, #T_36472_row3_col5, #T_36472_row4_col7, #T_36472_row5_col7, #T_36472_row6_col7, #T_36472_row7_col7, #T_36472_row8_col7 {
  background-color: #ceecc8;
  color: #000000;
  text-align: center;
}
#T_36472_row1_col7, #T_36472_row2_col7, #T_36472_row3_col7 {
  background-color: #c2e7bb;
  color: #000000;
  text-align: center;
}
#T_36472_row4_col6, #T_36472_row5_col5, #T_36472_row6_col5, #T_36472_row7_col5, #T_36472_row8_col6, #T_36472_row10_col5, #T_36472_row10_col6, #T_36472_row10_col7, #T_36472_row11_col5, #T_36472_row11_col6, #T_36472_row11_col7, #T_36472_row12_col5, #T_36472_row12_col6, #T_36472_row12_col7, #T_36472_row13_col5, #T_36472_row13_col6, #T_36472_row13_col7, #T_36472_row14_col5, #T_36472_row14_col6, #T_36472_row14_col7, #T_36472_row15_col5, #T_36472_row15_col6, #T_36472_row15_col7, #T_36472_row16_col6, #T_36472_row17_col6, #T_36472_row18_col6, #T_36472_row19_col6, #T_36472_row20_col6, #T_36472_row21_col6, #T_36472_row22_col6, #T_36472_row23_col6 {
  background-color: #dbf1d6;
  color: #000000;
  text-align: center;
}
#T_36472_row5_col2, #T_36472_row6_col2, #T_36472_row7_col2, #T_36472_row16_col2, #T_36472_row17_col2, #T_36472_row18_col2, #T_36472_row19_col2, #T_36472_row20_col2, #T_36472_row21_col2, #T_36472_row24_col1, #T_36472_row25_col1, #T_36472_row25_col2, #T_36472_row26_col1, #T_36472_row26_col2, #T_36472_row27_col1, #T_36472_row28_col1, #T_36472_row28_col2, #T_36472_row29_col1, #T_36472_row30_col1, #T_36472_row31_col1 {
  background-color: #e95e0d;
  color: #f1f1f1;
  text-align: center;
}
#T_36472_row9_col2, #T_36472_row22_col2, #T_36472_row23_col2, #T_36472_row24_col2 {
  background-color: #7f2704;
  color: #f1f1f1;
  text-align: center;
}
#T_36472_row9_col4, #T_36472_row22_col4, #T_36472_row23_col4, #T_36472_row24_col3, #T_36472_row24_col4, #T_36472_row24_col5, #T_36472_row24_col6, #T_36472_row25_col3, #T_36472_row25_col5, #T_36472_row25_col6, #T_36472_row26_col3, #T_36472_row26_col5, #T_36472_row26_col6, #T_36472_row27_col3, #T_36472_row27_col5, #T_36472_row27_col6, #T_36472_row28_col3, #T_36472_row28_col5, #T_36472_row28_col6, #T_36472_row29_col3, #T_36472_row29_col5, #T_36472_row29_col6, #T_36472_row30_col3, #T_36472_row30_col5, #T_36472_row30_col6, #T_36472_row31_col3, #T_36472_row31_col6 {
  background-color: #f7fcf5;
  color: #000000;
  text-align: center;
}
#T_36472_row9_col5, #T_36472_row16_col7, #T_36472_row17_col7, #T_36472_row18_col7, #T_36472_row19_col7, #T_36472_row20_col7, #T_36472_row21_col7 {
  background-color: #e3f4de;
  color: #000000;
  text-align: center;
}
#T_36472_row9_col7 {
  background-color: #d6efd0;
  color: #000000;
  text-align: center;
}
#T_36472_row16_col5, #T_36472_row17_col5, #T_36472_row18_col5, #T_36472_row19_col5, #T_36472_row20_col5, #T_36472_row21_col5, #T_36472_row22_col7, #T_36472_row23_col7 {
  background-color: #e7f6e3;
  color: #000000;
  text-align: center;
}
#T_36472_row22_col5, #T_36472_row23_col5 {
  background-color: #ebf7e7;
  color: #000000;
  text-align: center;
}
#T_36472_row24_col7, #T_36472_row25_col7, #T_36472_row26_col7, #T_36472_row27_col7, #T_36472_row28_col7, #T_36472_row29_col7, #T_36472_row30_col7, #T_36472_row31_col5, #T_36472_row31_col7 {
  background-color: #000000;
  color: #f1f1f1;
  background-color: lightgray;
  text-align: center;
}
</style>
<table id="T_36472">
  <caption>Possible Classifications</caption>
  <thead>
    <tr>
      <th id="T_36472_level0_col0" class="col_heading level0 col0" >predictions</th>
      <th id="T_36472_level0_col1" class="col_heading level0 col1" >fp</th>
      <th id="T_36472_level0_col2" class="col_heading level0 col2" >fn</th>
      <th id="T_36472_level0_col3" class="col_heading level0 col3" >tp</th>
      <th id="T_36472_level0_col4" class="col_heading level0 col4" >tn</th>
      <th id="T_36472_level0_col5" class="col_heading level0 col5" >precision</th>
      <th id="T_36472_level0_col6" class="col_heading level0 col6" >recall</th>
      <th id="T_36472_level0_col7" class="col_heading level0 col7" >f1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_36472_row0_col0" class="data row0 col0" >11 000</td>
      <td id="T_36472_row0_col1" class="data row0 col1" >0.00</td>
      <td id="T_36472_row0_col2" class="data row0 col2" >0.00</td>
      <td id="T_36472_row0_col3" class="data row0 col3" >2.00</td>
      <td id="T_36472_row0_col4" class="data row0 col4" >3.00</td>
      <td id="T_36472_row0_col5" class="data row0 col5" >1.00</td>
      <td id="T_36472_row0_col6" class="data row0 col6" >1.00</td>
      <td id="T_36472_row0_col7" class="data row0 col7" >1.00</td>
    </tr>
    <tr>
      <td id="T_36472_row1_col0" class="data row1 col0" >11 100</td>
      <td id="T_36472_row1_col1" class="data row1 col1" >0.00</td>
      <td id="T_36472_row1_col2" class="data row1 col2" >1.00</td>
      <td id="T_36472_row1_col3" class="data row1 col3" >2.00</td>
      <td id="T_36472_row1_col4" class="data row1 col4" >2.00</td>
      <td id="T_36472_row1_col5" class="data row1 col5" >0.67</td>
      <td id="T_36472_row1_col6" class="data row1 col6" >1.00</td>
      <td id="T_36472_row1_col7" class="data row1 col7" >0.80</td>
    </tr>
    <tr>
      <td id="T_36472_row2_col0" class="data row2 col0" >11 010</td>
      <td id="T_36472_row2_col1" class="data row2 col1" >0.00</td>
      <td id="T_36472_row2_col2" class="data row2 col2" >1.00</td>
      <td id="T_36472_row2_col3" class="data row2 col3" >2.00</td>
      <td id="T_36472_row2_col4" class="data row2 col4" >2.00</td>
      <td id="T_36472_row2_col5" class="data row2 col5" >0.67</td>
      <td id="T_36472_row2_col6" class="data row2 col6" >1.00</td>
      <td id="T_36472_row2_col7" class="data row2 col7" >0.80</td>
    </tr>
    <tr>
      <td id="T_36472_row3_col0" class="data row3 col0" >11 001</td>
      <td id="T_36472_row3_col1" class="data row3 col1" >0.00</td>
      <td id="T_36472_row3_col2" class="data row3 col2" >1.00</td>
      <td id="T_36472_row3_col3" class="data row3 col3" >2.00</td>
      <td id="T_36472_row3_col4" class="data row3 col4" >2.00</td>
      <td id="T_36472_row3_col5" class="data row3 col5" >0.67</td>
      <td id="T_36472_row3_col6" class="data row3 col6" >1.00</td>
      <td id="T_36472_row3_col7" class="data row3 col7" >0.80</td>
    </tr>
    <tr>
      <td id="T_36472_row4_col0" class="data row4 col0" >01 000</td>
      <td id="T_36472_row4_col1" class="data row4 col1" >1.00</td>
      <td id="T_36472_row4_col2" class="data row4 col2" >0.00</td>
      <td id="T_36472_row4_col3" class="data row4 col3" >1.00</td>
      <td id="T_36472_row4_col4" class="data row4 col4" >3.00</td>
      <td id="T_36472_row4_col5" class="data row4 col5" >1.00</td>
      <td id="T_36472_row4_col6" class="data row4 col6" >0.50</td>
      <td id="T_36472_row4_col7" class="data row4 col7" >0.67</td>
    </tr>
    <tr>
      <td id="T_36472_row5_col0" class="data row5 col0" >11 110</td>
      <td id="T_36472_row5_col1" class="data row5 col1" >0.00</td>
      <td id="T_36472_row5_col2" class="data row5 col2" >2.00</td>
      <td id="T_36472_row5_col3" class="data row5 col3" >2.00</td>
      <td id="T_36472_row5_col4" class="data row5 col4" >1.00</td>
      <td id="T_36472_row5_col5" class="data row5 col5" >0.50</td>
      <td id="T_36472_row5_col6" class="data row5 col6" >1.00</td>
      <td id="T_36472_row5_col7" class="data row5 col7" >0.67</td>
    </tr>
    <tr>
      <td id="T_36472_row6_col0" class="data row6 col0" >11 101</td>
      <td id="T_36472_row6_col1" class="data row6 col1" >0.00</td>
      <td id="T_36472_row6_col2" class="data row6 col2" >2.00</td>
      <td id="T_36472_row6_col3" class="data row6 col3" >2.00</td>
      <td id="T_36472_row6_col4" class="data row6 col4" >1.00</td>
      <td id="T_36472_row6_col5" class="data row6 col5" >0.50</td>
      <td id="T_36472_row6_col6" class="data row6 col6" >1.00</td>
      <td id="T_36472_row6_col7" class="data row6 col7" >0.67</td>
    </tr>
    <tr>
      <td id="T_36472_row7_col0" class="data row7 col0" >11 011</td>
      <td id="T_36472_row7_col1" class="data row7 col1" >0.00</td>
      <td id="T_36472_row7_col2" class="data row7 col2" >2.00</td>
      <td id="T_36472_row7_col3" class="data row7 col3" >2.00</td>
      <td id="T_36472_row7_col4" class="data row7 col4" >1.00</td>
      <td id="T_36472_row7_col5" class="data row7 col5" >0.50</td>
      <td id="T_36472_row7_col6" class="data row7 col6" >1.00</td>
      <td id="T_36472_row7_col7" class="data row7 col7" >0.67</td>
    </tr>
    <tr>
      <td id="T_36472_row8_col0" class="data row8 col0" >10 000</td>
      <td id="T_36472_row8_col1" class="data row8 col1" >1.00</td>
      <td id="T_36472_row8_col2" class="data row8 col2" >0.00</td>
      <td id="T_36472_row8_col3" class="data row8 col3" >1.00</td>
      <td id="T_36472_row8_col4" class="data row8 col4" >3.00</td>
      <td id="T_36472_row8_col5" class="data row8 col5" >1.00</td>
      <td id="T_36472_row8_col6" class="data row8 col6" >0.50</td>
      <td id="T_36472_row8_col7" class="data row8 col7" >0.67</td>
    </tr>
    <tr>
      <td id="T_36472_row9_col0" class="data row9 col0" >11 111</td>
      <td id="T_36472_row9_col1" class="data row9 col1" >0.00</td>
      <td id="T_36472_row9_col2" class="data row9 col2" >3.00</td>
      <td id="T_36472_row9_col3" class="data row9 col3" >2.00</td>
      <td id="T_36472_row9_col4" class="data row9 col4" >0.00</td>
      <td id="T_36472_row9_col5" class="data row9 col5" >0.40</td>
      <td id="T_36472_row9_col6" class="data row9 col6" >1.00</td>
      <td id="T_36472_row9_col7" class="data row9 col7" >0.57</td>
    </tr>
    <tr>
      <td id="T_36472_row10_col0" class="data row10 col0" >10 100</td>
      <td id="T_36472_row10_col1" class="data row10 col1" >1.00</td>
      <td id="T_36472_row10_col2" class="data row10 col2" >1.00</td>
      <td id="T_36472_row10_col3" class="data row10 col3" >1.00</td>
      <td id="T_36472_row10_col4" class="data row10 col4" >2.00</td>
      <td id="T_36472_row10_col5" class="data row10 col5" >0.50</td>
      <td id="T_36472_row10_col6" class="data row10 col6" >0.50</td>
      <td id="T_36472_row10_col7" class="data row10 col7" >0.50</td>
    </tr>
    <tr>
      <td id="T_36472_row11_col0" class="data row11 col0" >10 010</td>
      <td id="T_36472_row11_col1" class="data row11 col1" >1.00</td>
      <td id="T_36472_row11_col2" class="data row11 col2" >1.00</td>
      <td id="T_36472_row11_col3" class="data row11 col3" >1.00</td>
      <td id="T_36472_row11_col4" class="data row11 col4" >2.00</td>
      <td id="T_36472_row11_col5" class="data row11 col5" >0.50</td>
      <td id="T_36472_row11_col6" class="data row11 col6" >0.50</td>
      <td id="T_36472_row11_col7" class="data row11 col7" >0.50</td>
    </tr>
    <tr>
      <td id="T_36472_row12_col0" class="data row12 col0" >10 001</td>
      <td id="T_36472_row12_col1" class="data row12 col1" >1.00</td>
      <td id="T_36472_row12_col2" class="data row12 col2" >1.00</td>
      <td id="T_36472_row12_col3" class="data row12 col3" >1.00</td>
      <td id="T_36472_row12_col4" class="data row12 col4" >2.00</td>
      <td id="T_36472_row12_col5" class="data row12 col5" >0.50</td>
      <td id="T_36472_row12_col6" class="data row12 col6" >0.50</td>
      <td id="T_36472_row12_col7" class="data row12 col7" >0.50</td>
    </tr>
    <tr>
      <td id="T_36472_row13_col0" class="data row13 col0" >01 100</td>
      <td id="T_36472_row13_col1" class="data row13 col1" >1.00</td>
      <td id="T_36472_row13_col2" class="data row13 col2" >1.00</td>
      <td id="T_36472_row13_col3" class="data row13 col3" >1.00</td>
      <td id="T_36472_row13_col4" class="data row13 col4" >2.00</td>
      <td id="T_36472_row13_col5" class="data row13 col5" >0.50</td>
      <td id="T_36472_row13_col6" class="data row13 col6" >0.50</td>
      <td id="T_36472_row13_col7" class="data row13 col7" >0.50</td>
    </tr>
    <tr>
      <td id="T_36472_row14_col0" class="data row14 col0" >01 010</td>
      <td id="T_36472_row14_col1" class="data row14 col1" >1.00</td>
      <td id="T_36472_row14_col2" class="data row14 col2" >1.00</td>
      <td id="T_36472_row14_col3" class="data row14 col3" >1.00</td>
      <td id="T_36472_row14_col4" class="data row14 col4" >2.00</td>
      <td id="T_36472_row14_col5" class="data row14 col5" >0.50</td>
      <td id="T_36472_row14_col6" class="data row14 col6" >0.50</td>
      <td id="T_36472_row14_col7" class="data row14 col7" >0.50</td>
    </tr>
    <tr>
      <td id="T_36472_row15_col0" class="data row15 col0" >01 001</td>
      <td id="T_36472_row15_col1" class="data row15 col1" >1.00</td>
      <td id="T_36472_row15_col2" class="data row15 col2" >1.00</td>
      <td id="T_36472_row15_col3" class="data row15 col3" >1.00</td>
      <td id="T_36472_row15_col4" class="data row15 col4" >2.00</td>
      <td id="T_36472_row15_col5" class="data row15 col5" >0.50</td>
      <td id="T_36472_row15_col6" class="data row15 col6" >0.50</td>
      <td id="T_36472_row15_col7" class="data row15 col7" >0.50</td>
    </tr>
    <tr>
      <td id="T_36472_row16_col0" class="data row16 col0" >10 101</td>
      <td id="T_36472_row16_col1" class="data row16 col1" >1.00</td>
      <td id="T_36472_row16_col2" class="data row16 col2" >2.00</td>
      <td id="T_36472_row16_col3" class="data row16 col3" >1.00</td>
      <td id="T_36472_row16_col4" class="data row16 col4" >1.00</td>
      <td id="T_36472_row16_col5" class="data row16 col5" >0.33</td>
      <td id="T_36472_row16_col6" class="data row16 col6" >0.50</td>
      <td id="T_36472_row16_col7" class="data row16 col7" >0.40</td>
    </tr>
    <tr>
      <td id="T_36472_row17_col0" class="data row17 col0" >10 110</td>
      <td id="T_36472_row17_col1" class="data row17 col1" >1.00</td>
      <td id="T_36472_row17_col2" class="data row17 col2" >2.00</td>
      <td id="T_36472_row17_col3" class="data row17 col3" >1.00</td>
      <td id="T_36472_row17_col4" class="data row17 col4" >1.00</td>
      <td id="T_36472_row17_col5" class="data row17 col5" >0.33</td>
      <td id="T_36472_row17_col6" class="data row17 col6" >0.50</td>
      <td id="T_36472_row17_col7" class="data row17 col7" >0.40</td>
    </tr>
    <tr>
      <td id="T_36472_row18_col0" class="data row18 col0" >01 110</td>
      <td id="T_36472_row18_col1" class="data row18 col1" >1.00</td>
      <td id="T_36472_row18_col2" class="data row18 col2" >2.00</td>
      <td id="T_36472_row18_col3" class="data row18 col3" >1.00</td>
      <td id="T_36472_row18_col4" class="data row18 col4" >1.00</td>
      <td id="T_36472_row18_col5" class="data row18 col5" >0.33</td>
      <td id="T_36472_row18_col6" class="data row18 col6" >0.50</td>
      <td id="T_36472_row18_col7" class="data row18 col7" >0.40</td>
    </tr>
    <tr>
      <td id="T_36472_row19_col0" class="data row19 col0" >01 101</td>
      <td id="T_36472_row19_col1" class="data row19 col1" >1.00</td>
      <td id="T_36472_row19_col2" class="data row19 col2" >2.00</td>
      <td id="T_36472_row19_col3" class="data row19 col3" >1.00</td>
      <td id="T_36472_row19_col4" class="data row19 col4" >1.00</td>
      <td id="T_36472_row19_col5" class="data row19 col5" >0.33</td>
      <td id="T_36472_row19_col6" class="data row19 col6" >0.50</td>
      <td id="T_36472_row19_col7" class="data row19 col7" >0.40</td>
    </tr>
    <tr>
      <td id="T_36472_row20_col0" class="data row20 col0" >01 011</td>
      <td id="T_36472_row20_col1" class="data row20 col1" >1.00</td>
      <td id="T_36472_row20_col2" class="data row20 col2" >2.00</td>
      <td id="T_36472_row20_col3" class="data row20 col3" >1.00</td>
      <td id="T_36472_row20_col4" class="data row20 col4" >1.00</td>
      <td id="T_36472_row20_col5" class="data row20 col5" >0.33</td>
      <td id="T_36472_row20_col6" class="data row20 col6" >0.50</td>
      <td id="T_36472_row20_col7" class="data row20 col7" >0.40</td>
    </tr>
    <tr>
      <td id="T_36472_row21_col0" class="data row21 col0" >10 011</td>
      <td id="T_36472_row21_col1" class="data row21 col1" >1.00</td>
      <td id="T_36472_row21_col2" class="data row21 col2" >2.00</td>
      <td id="T_36472_row21_col3" class="data row21 col3" >1.00</td>
      <td id="T_36472_row21_col4" class="data row21 col4" >1.00</td>
      <td id="T_36472_row21_col5" class="data row21 col5" >0.33</td>
      <td id="T_36472_row21_col6" class="data row21 col6" >0.50</td>
      <td id="T_36472_row21_col7" class="data row21 col7" >0.40</td>
    </tr>
    <tr>
      <td id="T_36472_row22_col0" class="data row22 col0" >10 111</td>
      <td id="T_36472_row22_col1" class="data row22 col1" >1.00</td>
      <td id="T_36472_row22_col2" class="data row22 col2" >3.00</td>
      <td id="T_36472_row22_col3" class="data row22 col3" >1.00</td>
      <td id="T_36472_row22_col4" class="data row22 col4" >0.00</td>
      <td id="T_36472_row22_col5" class="data row22 col5" >0.25</td>
      <td id="T_36472_row22_col6" class="data row22 col6" >0.50</td>
      <td id="T_36472_row22_col7" class="data row22 col7" >0.33</td>
    </tr>
    <tr>
      <td id="T_36472_row23_col0" class="data row23 col0" >01 111</td>
      <td id="T_36472_row23_col1" class="data row23 col1" >1.00</td>
      <td id="T_36472_row23_col2" class="data row23 col2" >3.00</td>
      <td id="T_36472_row23_col3" class="data row23 col3" >1.00</td>
      <td id="T_36472_row23_col4" class="data row23 col4" >0.00</td>
      <td id="T_36472_row23_col5" class="data row23 col5" >0.25</td>
      <td id="T_36472_row23_col6" class="data row23 col6" >0.50</td>
      <td id="T_36472_row23_col7" class="data row23 col7" >0.33</td>
    </tr>
    <tr>
      <td id="T_36472_row24_col0" class="data row24 col0" >00 111</td>
      <td id="T_36472_row24_col1" class="data row24 col1" >2.00</td>
      <td id="T_36472_row24_col2" class="data row24 col2" >3.00</td>
      <td id="T_36472_row24_col3" class="data row24 col3" >0.00</td>
      <td id="T_36472_row24_col4" class="data row24 col4" >0.00</td>
      <td id="T_36472_row24_col5" class="data row24 col5" >0.00</td>
      <td id="T_36472_row24_col6" class="data row24 col6" >0.00</td>
      <td id="T_36472_row24_col7" class="data row24 col7" >nan</td>
    </tr>
    <tr>
      <td id="T_36472_row25_col0" class="data row25 col0" >00 110</td>
      <td id="T_36472_row25_col1" class="data row25 col1" >2.00</td>
      <td id="T_36472_row25_col2" class="data row25 col2" >2.00</td>
      <td id="T_36472_row25_col3" class="data row25 col3" >0.00</td>
      <td id="T_36472_row25_col4" class="data row25 col4" >1.00</td>
      <td id="T_36472_row25_col5" class="data row25 col5" >0.00</td>
      <td id="T_36472_row25_col6" class="data row25 col6" >0.00</td>
      <td id="T_36472_row25_col7" class="data row25 col7" >nan</td>
    </tr>
    <tr>
      <td id="T_36472_row26_col0" class="data row26 col0" >00 101</td>
      <td id="T_36472_row26_col1" class="data row26 col1" >2.00</td>
      <td id="T_36472_row26_col2" class="data row26 col2" >2.00</td>
      <td id="T_36472_row26_col3" class="data row26 col3" >0.00</td>
      <td id="T_36472_row26_col4" class="data row26 col4" >1.00</td>
      <td id="T_36472_row26_col5" class="data row26 col5" >0.00</td>
      <td id="T_36472_row26_col6" class="data row26 col6" >0.00</td>
      <td id="T_36472_row26_col7" class="data row26 col7" >nan</td>
    </tr>
    <tr>
      <td id="T_36472_row27_col0" class="data row27 col0" >00 100</td>
      <td id="T_36472_row27_col1" class="data row27 col1" >2.00</td>
      <td id="T_36472_row27_col2" class="data row27 col2" >1.00</td>
      <td id="T_36472_row27_col3" class="data row27 col3" >0.00</td>
      <td id="T_36472_row27_col4" class="data row27 col4" >2.00</td>
      <td id="T_36472_row27_col5" class="data row27 col5" >0.00</td>
      <td id="T_36472_row27_col6" class="data row27 col6" >0.00</td>
      <td id="T_36472_row27_col7" class="data row27 col7" >nan</td>
    </tr>
    <tr>
      <td id="T_36472_row28_col0" class="data row28 col0" >00 011</td>
      <td id="T_36472_row28_col1" class="data row28 col1" >2.00</td>
      <td id="T_36472_row28_col2" class="data row28 col2" >2.00</td>
      <td id="T_36472_row28_col3" class="data row28 col3" >0.00</td>
      <td id="T_36472_row28_col4" class="data row28 col4" >1.00</td>
      <td id="T_36472_row28_col5" class="data row28 col5" >0.00</td>
      <td id="T_36472_row28_col6" class="data row28 col6" >0.00</td>
      <td id="T_36472_row28_col7" class="data row28 col7" >nan</td>
    </tr>
    <tr>
      <td id="T_36472_row29_col0" class="data row29 col0" >00 010</td>
      <td id="T_36472_row29_col1" class="data row29 col1" >2.00</td>
      <td id="T_36472_row29_col2" class="data row29 col2" >1.00</td>
      <td id="T_36472_row29_col3" class="data row29 col3" >0.00</td>
      <td id="T_36472_row29_col4" class="data row29 col4" >2.00</td>
      <td id="T_36472_row29_col5" class="data row29 col5" >0.00</td>
      <td id="T_36472_row29_col6" class="data row29 col6" >0.00</td>
      <td id="T_36472_row29_col7" class="data row29 col7" >nan</td>
    </tr>
    <tr>
      <td id="T_36472_row30_col0" class="data row30 col0" >00 001</td>
      <td id="T_36472_row30_col1" class="data row30 col1" >2.00</td>
      <td id="T_36472_row30_col2" class="data row30 col2" >1.00</td>
      <td id="T_36472_row30_col3" class="data row30 col3" >0.00</td>
      <td id="T_36472_row30_col4" class="data row30 col4" >2.00</td>
      <td id="T_36472_row30_col5" class="data row30 col5" >0.00</td>
      <td id="T_36472_row30_col6" class="data row30 col6" >0.00</td>
      <td id="T_36472_row30_col7" class="data row30 col7" >nan</td>
    </tr>
    <tr>
      <td id="T_36472_row31_col0" class="data row31 col0" >00 000</td>
      <td id="T_36472_row31_col1" class="data row31 col1" >2.00</td>
      <td id="T_36472_row31_col2" class="data row31 col2" >0.00</td>
      <td id="T_36472_row31_col3" class="data row31 col3" >0.00</td>
      <td id="T_36472_row31_col4" class="data row31 col4" >3.00</td>
      <td id="T_36472_row31_col5" class="data row31 col5" >nan</td>
      <td id="T_36472_row31_col6" class="data row31 col6" >0.00</td>
      <td id="T_36472_row31_col7" class="data row31 col7" >nan</td>
    </tr>
  </tbody>
</table>
</div>

F1 is undefined when no positives are predicted (division by zero); that’s why some rows show ‘nan’.

Notice that:
- The first entry in the [classification table](#all-classifiers-table) is our perfect Model A (predicting defendants 1 and 2 as high risk)
- The sixth entry in the [classification table](#all-classifiers-table) matches our Model B (predicting defendants 1, 2, 3, and 4 as high risk) and this is plotted as the left-most orange dot in {numref}`f1-isoquants`
- The ninth entry in the [classification table](#all-classifiers-table) matches our Model C (predicting only defendant 1 as high risk) and this is plotted as the right-most orange dot in {numref}`f1-isoquants`
- Five different prediction patterns yield the same F1 score of 66.7%
- The worst possible classifier (the last entry in the [classification table](#all-classifiers-table)) predicts everyone as low risk when there are actually high-risk defendants


```{figure} ../assets/scatter_precision_recall_f1.svg
:name: f1-isoquants
:align: center
:width: 80%

F1 score isoquants showing how different combinations of precision and recall can yield the same F1 score. Each curve represents a constant F1 value.
```

In the figure above, it is clear that the green dot is better than the purple dot because it has both a better precision and recall score. Again, it's not obvious which orange dot is better. The F1 score takes a stance by saying they are tied. Notice the curvature of the F1 isoquants as well. The convexity of the region above the curve means that averages are preferred to extremes, roughly. Any dot on the line segment connecting the two orange dots would have a higher F1 score and thus correspond to a better model.


## Experience and Training

Our computer programs learn from *experience* by being *trained* on data. The training process involves adjusting model parameters to improve performance on the chosen metric.

In reality, the bail prediction problem faces a prohibitive missing data problem. In practice, we only observe outcomes for defendants who were actually released. We never learn what would have happened to those who were detained, so we can never obtain counts for true or false positives. Clever researchers work around this limitation. {cite}`kleinberg2018human` exploits a natural experiment: comparing decisions by judges with different release tendencies to infer what might have happened in the counterfactual cases.

Moreover, our labels themselves are often proxies for what we really care about. When predicting "crime," we typically observe arrests or convictions—not actual criminal behavior. Many crimes go unreported or unsolved. This means our training data, while probably useful, is not perfect.

As the statistician George Box famously said, "All models are wrong, but some are useful." The question isn't whether our data perfectly captures reality (it doesn't), but whether our models can still provide valuable insights despite these limitations. In many cases, even imperfect predictions can improve upon human decision-making—as long as we remain clear-eyed about what our models can and cannot tell us. Finally, remember that your models cannot necessarily tell you anything without appropriate problem-specific knowledge. {cite}`kuhn2013applied` discusses this in the introduction (Section 1.2).


## Summary

We introduced the concept of a learning problem. This requires a performance measure and a way to use data (experience) to improve model performance. Prediction tasks are focal and when we have labeled data, we can use supervised learning techniques. We showed that the choice of a performance measure is (1) ambiguous because there are usually multiple plausible candidates and (2) substantive as it can lead you to prefer one model over the other. This highlights the **role of the researcher** in possessing problem-specific knowledge and technical knowledge. There is not generally a one-to-one mapping from learning tasks to particular models. The no free lunch theorems of {cite}`wolpert1996lack` show that for any two learning algorithms, one won't outperform the other in all contexts, aligning with the agnostic approach to machine learning methods advocated by {cite}`grimmer2021machine`. This places the burden on the researcher to understand and explore different models. 