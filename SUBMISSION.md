# Week 2 Lab Submission — Supervised Learning in Practice

AI Principles and Practice · COMP-0987 · Topic 02, Unit 02

One section below for each section of the lab. Answer as you go, rather than
at the end — most of these are the checkpoint questions you have already talked
through at the bench.

Short answers. One or two sentences unless it says otherwise. Delete each
`TODO` as you finish that section, then run `python make_submission.py`.

| | |
|---|---|
| Student number | 20068336 |
| Name | Joe Wemyss |
| Date | 2026-09-17 |

---

## Section 1 — The supervised setup

**1.1 Why did we fold the undocumented `EDUCATION` and `MARRIAGE` codes into
"other" rather than leave them or drop those rows?**

Last week we discovered that there is a strong correlation between one of the undocumented education
codes, and the default rate

**1.2 What makes `default` a classification target and `LIMIT_BAL` a
regression target?**

The default is binary. The user either defaulted or didn't. The limit is a continious value which
means its more suited to regression

**1.3 Why does the scaler have to live inside the pipeline? What would go
wrong if you called `StandardScaler().fit_transform()` on the full dataframe
before splitting?**

This is the same issue we looked at last week. We don't want the test data to leak into the training
data. The test data should represent unseen data

---

## Section 2 — Regression

**2.1 Give the R² and RMSE you got. Why are the linear, ridge and lasso results
almost identical on this data?**

TODO

**2.2 What does an R² of about 0.38 mean, in words, to someone who has never
heard of R²? Is it a bug, or an honest finding?**

TODO

**2.3 One sentence on the shape of your residual plot. Is it an even cloud or
a funnel, and what would a funnel tell you?**

TODO

---

## Section 3 — Classification

**3.1 Paste your results table. Which model has the best AUC, and does the
best-AUC model also have the best recall?**

TODO

**3.2 Logistic regression is about 81% accurate and catches about 24% of
defaulters. Is that model any use to the bank? Why might the bank care more
about recall than accuracy?**

TODO

**3.3 What does `stratify=yc` do, and why does it matter more when a class is
rare?**

TODO

---

## Section 4 — Loss and the bias–variance trade-off

**4.1 Name the failure mode your unlimited-depth learning curve shows, and the
three things you could do about it.**

TODO

**4.2 Where did validation AUC peak in your validation curve, and what happens
to the right of that peak, and why?**

TODO

**4.3 If both curves had met at a low score instead, what would that mean, and
would more data help? Give the one sentence linking `max_depth`, `k` in k-NN,
and the regularisation `alpha`.**

TODO

---

## Section 5 — Evaluation done properly

**5.1 Your cross-validated AUC: the fold scores, the mean and the standard
deviation. What does the spread tell you?**

TODO

**5.2 Why is 81% accuracy almost meaningless on this dataset? Use the
"always predict no-default" number in your answer.**

TODO

**5.3 Your confusion matrix, and which of the two errors you think costs the
bank more.**

TODO

**5.4 In one sentence each: what do precision and recall mean here, in terms of
defaulters? Which curve, ROC or precision–recall, would you trust more with a
22% default rate, and why?**

TODO

**5.5 Where would you set the threshold for the bank, and what would you need
to know about the cost of each error to decide? Two or three sentences.**

TODO

---

## Section 6 — Framing real problems

**6.1 Two sentences arguing why the framing chain (decision → target → metric
→ threshold), not the choice of model, is the hard and valuable part.**

TODO

**6.2 `SEX` is one of your model's inputs. Would removing the column make the
model fair? Give one reason it might not. Commit to an answer; Week 3 tests
it.**

TODO

---

## Section 7 — Exercises

Attempt 7.1 and 7.2, then do *either* 7.3 *or* 7.4, and finish with 7.5.

**7.1 What did you try to push recall above 0.5 with usable precision? Report
the recall and precision you landed on, and whether the trade is worth it.**

TODO

**7.2 Your paragraph for the manager who only wants the accuracy figure.**

TODO

**7.3 or 7.4 — say which one you did, then answer it.**

TODO

**7.5 Which problem did you frame? Walk the question → target → metric →
threshold chain in half a page.**

TODO

---

## Before you build the zip

**Use of AI assistants.** Say what you used one for, if anything, and what you
checked yourself. Using one is allowed; not being able to explain your own
submission is the problem. "None" is a complete answer.

TODO

**Anything you want to flag.** Optional — if something went wrong, if you ran
out of time, or if you want a second opinion on a judgement call. Delete the
TODO either way.

TODO
