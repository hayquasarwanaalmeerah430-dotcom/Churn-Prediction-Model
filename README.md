# Churn Prediction Model

**Internee.pk Internship Task** - predict which users are likely to leave the platform so support teams can reach out early.

## Objective
Analyze user behavior (login frequency, course engagement) and predict churn using **Random Forest** and **Gradient Boosting**. The outcome is a ranked list of at-risk users so personalized support can reduce churn.

## Data
`lms_data.csv` - LMS engagement logs and user activity: 3,000 users, no missing values, 35.6% churned.

| Column | Meaning |
|---|---|
| `logins_per_week` | login frequency |
| `days_since_last_login` | recency of activity |
| `courses_enrolled` / `courses_completed` | course engagement |
| `avg_session_minutes` | time spent per session |
| `avg_quiz_score` | performance |
| `forum_posts` | community participation |
| `churn` | target (1 = left, 0 = stayed) |

## Approach
1. Drop `user_id` (an identifier, not behavior).
2. Feature engineering: `completion_rate` (completed / enrolled), `weekly_minutes` (logins x session length), `inactive_14d` flag.
3. 80/20 stratified train/test split.
4. Stratified 5-fold cross-validation comparing Random Forest and Gradient Boosting.
5. GridSearchCV tuning of the best model (by ROC-AUC).
6. Evaluation on the held-out test set, feature importance, and risk scoring of users.

## Results
| Model (5-fold CV) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.734 | 0.659 | 0.536 | 0.590 | 0.781 |
| Gradient Boosting | 0.745 | 0.668 | 0.567 | 0.613 | 0.789 |

**Tuned Gradient Boosting on the test set:** accuracy 0.77, ROC-AUC 0.80, churn precision 0.70, churn recall 0.59, F1 0.64.

### What drives churn
1. `days_since_last_login` (about 50% of importance). Churned users averaged 38.8 days since last login versus 24.7 for retained users.
2. `weekly_minutes` and `logins_per_week`. Churned users logged in 3.2 times per week versus 4.4.
3. Quiz score and course completion rate matter less.

## Reducing churn with personalized support
`outputs/at_risk_users.csv` lists each test user with a churn probability and risk level (Low / Medium / High):
- **High risk (probability > 0.7):** personal outreach from a mentor or support agent.
- **Medium risk (0.4 - 0.7):** reminder email with course recommendations.
- **Inactive 14+ days:** automated re-engagement nudge.
- **Low completion rate:** offer help or a study plan for stalled courses.

## Limitations
Recall for churners is 0.59, so some leaving users are missed. Lowering the decision threshold below 0.5 would catch more at the cost of precision. Adding richer features (time-series activity, content type) could improve results.

## Run it
```bash
pip install -r requirements.txt
python churn_model.py
```
Charts and CSVs are saved in `outputs/`.

## Tools
Python, pandas, scikit-learn, matplotlib
