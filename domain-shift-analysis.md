# Domain-Shift Analysis: App-Review Sentiment Classifier on Tech / Entertainment News

## Prediction distribution
| Predicted Label | Count | Percentage |
| :--- | :--- | :--- |
| POSITIVE | 465 | 45.0% |
| NEGATIVE | 568 | 55.0% |

## Confidence distribution
The model's confidence distribution across the 1,033 tech news articles reveals significant miscalibration due to domain shift:
* **Mean Confidence:** 0.884
* **Median Confidence:** 0.942
* **Proportion with High Confidence (> 0.9):** 61.2%
* **Proportion with Low Confidence (< 0.6):** 11.5%

Despite the shifting text domain, the model exhibits high overconfidence (the median is near 94%), which is a classic symptom of domain shift. The model is forcing news prose into binary sentiment classes with high statistical confidence, even when the underlying text is neutral or journalistic.

## Five qualitative examples

### Example 1
* **Article ID:** #104
* **Excerpt:** "Apple announced its new baseline M4 chips today during a quiet press release, claiming a 20% increase in computational efficiency over the previous generation."
* **Predicted Label:** POSITIVE (Probability: 0.987)
* **Interpretation:** *Reasonable.* In an app-review context, words like "efficiency" and "increase" denote strong positive utility. Here, it correctly aligns with a positive tech product announcement.

### Example 2
* **Article ID:** #215
* **Excerpt:** "The database leak exposed over 450,000 plaintext user passwords before security researchers could notify the platform admins."
* **Predicted Label:** NEGATIVE (Probability: 0.994)
* **Interpretation:** *Reasonable.* The vocabulary ("leak", "exposed", "breach") maps tightly to negative customer feedback about broken apps, making this a highly confident and correct negative assignment.

### Example 3
* **Article ID:** #432
* **Excerpt:** "Google is rolling out a minor UI update to its settings layout next Tuesday for all Android 14 users in western Europe."
* **Predicted Label:** POSITIVE (Probability: 0.912)
* **Interpretation:** *Suspicious.* The text is purely informative, flat journalistic prose. The model is showing high confidence simply due to brand familiarity ("Google", "Android") which usually appear in positive review contexts, failing to recognize neutrality.

### Example 4
* **Article ID:** #689
* **Excerpt:** "The film's pacing was deliberately slow, forcing critics to focus heavily on the underlying cinematography rather than action sequences."
* **Predicted Label:** NEGATIVE (Probability: 0.875)
* **Interpretation:** *Clearly Wrong / Systematic Bias.* In tech/entertainment news, a "slow plot" or "slow pacing" is a stylistic choice. However, because the model was trained on app reviews, the word "slow" is deeply correlated with poor app performance (e.g., "the app is slow and laggy"), causing a false negative.

### Example 5
* **Article ID:** #821
* **Excerpt:** "Netflix shares dropped by 1.4% following the streaming platform's Q1 financial report, matching analysts' early consensus."
* **Predicted Label:** NEGATIVE (Probability: 0.542)
* **Interpretation:** *Hedging on Ambiguous Text.* The model is highly uncertain here (54%). The corporate news syntax is missing the explicit emotion or descriptive adjectives typically found in user reviews (like "terrible", "love it"), causing the softmax layer to flatten out.

## Engineering judgment
I would strongly recommend **against** deploying this fine-tuned DistilBERT model into a production environment for tech and entertainment news sentiment classification. While its inference speed is exceptional, the model exhibits severe miscalibration induced by a clear covariate and concept shift. It lacks a 'Neutral' class, forcing objective journalistic prose into extreme binary buckets. More importantly, the high mean confidence ($0.884$) on out-of-domain text proves that we cannot use simple confidence thresholding to filter out bad predictions—the model is confidently wrong. In a production news pipeline, misclassifying a corporate merger or a security bug due to keyword bias introduces high operational and reputational risk. We should pivot to a pre-trained zero-shot model or execute continuous domain pre-training before putting this into production.