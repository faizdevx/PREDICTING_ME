# SIMULATION

## Idea

You already built a model that predicts:

**Given a student's behavior → will they pass or fail?**

In this section, we go one step further and use the trained model as a **simulator**.

Instead of merely predicting existing students, we ask counterfactual questions:

- Create many **synthetic students** that statistically resemble the real dataset.
- Ask the model: **“How many of them pass?”**
- Change their behavior (increase study time, reduce absences, limit alcohol).
- Ask again: **“Now how many pass?”**
- Compare worlds **before vs after interventions**.

Important:

> We are not changing the real world.  
> We are exploring what the *model believes* about the world.

---

---

## 1. Policy Simulation Results

Model output under different simulated interventions:

| Policy                  | Pass Rate |
|--------------------------|-----------|
| Baseline                | 0.555 |
| Cut absences            | 0.559 |
| Boost study time        | 0.622 |
| Reduce alcohol          | 0.567 |
| Combo (absences+study) | 0.625 |

---

### Interpreting like a human

#### Baseline World  
**55.5% pass rate**

This is the model’s view of the status quo.

Outcomes are nearly a coin flip.  
That already signals how noisy and unstable educational prediction is.

---

#### Cut Absences  
**55.9% (+0.4%)**

Reducing absences helps only marginally.

**Interpretation:**

Attendance correlates with failure, but changing attendance alone does not strongly improve outcomes.

---

#### Boost Study Time  
**62.2% (+6.7%)**

This is the largest effect in the entire simulation.

**Interpretation:**

> Active studying matters far more than simply being present.

According to the model, study behavior has the strongest influence on predicted success.

---

#### Reduce Alcohol  
**56.7% (+1.2%)**

Small improvement.

**Interpretation:**

Alcohol use correlates with outcomes, but once study habits and prior failures exist, it is not a dominant factor.

---

#### Combo Policy (Absence + Study)  
**62.5% (+7.0%)**

Only slightly better than boosting study alone.

**Interpretation:**

Study time dominates.  
Absence reduction adds only minimal extra benefit.

---

---

## 2. Conceptual Insight

This entire exercise provides **feature importance through intervention**, without using:

- SHAP
- Model coefficients
- Permutation tests

Pure simulation uncovered the leverage hierarchy:


> Showing up is less important than actually studying.

This is not directly obvious from raw correlations alone.  
Simulation exposes what actually moves the system.

---

---

## 3. Scientific Value

This project performs **counterfactual reasoning using an ML surrogate world**:

> “If the same students lived in a universe where behavior X changed, what outcomes would appear?”

This connects:

- **Prediction** → what will happen  
- **Policy reasoning** → what should we change

This is the same conceptual mechanism used by:

- Education policy simulations  
- Labor market modeling in economics  
- Public health intervention forecasting

Your project uses these tools on a small scale.

---

---

## 4. Execution Pipeline

The entire workflow is intentionally simple:

### ① Load Real Students  
`Real data → behavior features → pass/fail labels`

---

### ② Train Prediction Model  

`Student habits → ML classifier → pass probability`

This is the logistic regression pipeline.

---

### ③ Copy the World (Simulation)

`Resample dataset → build 10,000 synthetic students`

Creates a population statistically similar to the original one.

---

### ④ Evaluate Baseline

`World model(sim_students) → baseline pass rate`

---

### ⑤ Apply Interventions

Modify the synthetic population:

- Increase study time
- Reduce absences
- Reduce alcohol consumption

---

### ⑥ Re-evaluate

`Modified world → new pass rate`

Compare to baseline in a decision table.

---

### ⑦ Interpret Limits

Key realization:

> We are not simulating real causality.  
> We are simulating what the **model believes** would happen.

Therefore:

- **Garbage model → garbage policy simulation**
- **Correlation-based learning → pseudo-causal conclusions**

---

---

## 5. Why This Is Beyond Typical Coursework

Most student projects stop here:

> “Our classifier achieved 82% accuracy.”

This project continues with:

> “Given this classifier, what actions actually change outcomes?”

This reframes ML as:

- A **policy exploration tool**, not just a prediction engine
- A system that can distort recommendations when causality is unknown

You moved from:

**Prediction → Consequence modeling**

That shift represents genuine analytical maturity.

---

---

---

# SUBGROUP SIMULATION (FAIRNESS)

Predicted pass probability by social groups:

| Group Type | Group | Samples | Avg Pass Prob |
|------------|-------|----------|----------------|
| Internet | No | 151 | 0.447 |
| Internet | Yes | 498 | 0.595 |
| Famsup | No | 251 | 0.532 |
| Famsup | Yes | 398 | 0.578 |
| School | GP | 423 | 0.594 |
| School | MS | 226 | 0.497 |
| Sex | F | 383 | 0.595 |
| Sex | M | 266 | 0.510 |

---

### Internet

- No internet → **0.45**
- Yes internet → **0.59**

Model belief:

> Internet access strongly boosts predicted success.

This may reflect real advantages (resources, homework access), but also exposes potential **structural inequality amplification** through ML.

---

### Family Support

- No → **0.53**
- Yes → **0.58**

Moderate impact.  
Support matters, but not as strongly as digital access.

---

### School

- GP → **0.59**
- MS → **0.50**

Model views GP students as more likely to succeed even under similar feature profiles.

This may proxy:

- Teaching quality
- Peer culture
- Socioeconomic environment

Or it reflects **overfitting to dominant-school patterns**, consistent with the cross-school generalization drop observed earlier.

---

### Sex

- Female → **0.595**
- Male → **0.510**

Patterns align with many real datasets.

But in high-stakes use, this becomes **algorithmic bias** if predictions influence decisions.

---

### Fairness Summary

> The model systematically assigns higher success probabilities to students with internet access, family support, from school GP, and to female students. These patterns may reflect social realities but also demonstrate how ML internalizes existing inequalities and propagates them into automated decisions.

---

---

---

# THRESHOLD SIMULATION

Metrics as the decision threshold is swept:

| Threshold | Accuracy | Precision | Recall | F1 |
|----------|-----------|------------|--------|----|
| 0.1 | 0.892 | 0.904 | 0.983 | 0.942 |
| 0.2 | 0.892 | 0.917 | 0.965 | 0.941 |
| 0.3 | 0.869 | 0.930 | 0.922 | 0.926 |
| 0.4 | 0.846 | 0.936 | 0.887 | 0.911 |
| 0.5 | 0.777 | 0.939 | 0.800 | 0.864 |
| 0.6 | 0.600 | 0.933 | 0.591 | 0.723 |
| 0.7 | 0.462 | 0.959 | 0.409 | 0.573 |
| 0.8 | 0.223 | 1.000 | 0.122 | 0.217 |
| 0.9 | 0.115 | 0.000 | 0.000 | 0.000 |

---

### Low Threshold (0.1–0.3)

- **High recall, high precision, high accuracy**
- Model predicts *pass* for almost everyone.

Best at **not missing successful students**, terrible at identifying failures.

---

### Moderate Threshold (0.4–0.5)

- Balanced tradeoff.
- Stricter classification.
- Fewer false positives, more false negatives.

Most reasonable domain for real policy systems.

---

### High Threshold (0.6–0.8)

- Extremely selective.
- Precision stays high.
- Recall collapses.

Creates an **elitist filter** that only selects a small, high-confidence set of students.

---

### Value Judgment

> Changing the threshold is not a technical decision.  
> It is a **human policy choice** embedded inside an ML system.

---

**Summary Sentence:**

> By sweeping the decision threshold, the same model can behave like a generous school that passes most students or like a hyper-selective gatekeeper recognizing only a small elite. ML does not solve these trade-offs, it encodes human values.

---

---

---

# NOISE ROBUSTNESS

### Results

Baseline avg pass prob: 0.560
Noisy avg pass prob: 0.541
Prediction flip rate: 0.324


---

### Interpretation

- Average output shifts only slightly under noise.  
- But **32.4% of individual predictions flip labels**.

This means:

> The model looks stable at the population level but is fragile at the individual level.

Small behavior changes can dramatically alter an individual's predicted outcome.

---

### Final Insight

> Population statistics can look safe while individual decisions remain highly unstable.

This is dangerous for high-stakes individual decision systems.

---

---

---

## Final Synthesis

This project demonstrates that:

- ML predicts crowd-level patterns well.
- ML fails to understand individuals or causes.
- Policy suggestions derived from ML encode social bias and threshold ethics.
- Individual predictions remain fragile under small perturbations.

---

### Simulation Ladder

1. **Policy Interventions**  
   → What behaviors shift outcomes?

2. **Fairness Subgroups**  
   → Who does the model favor?

3. **Threshold Sweeps**  
   → How do policy attitudes change results?

4. **Noise Stability**  
   → Are individuals treated consistently?

---

> Machine Learning learns statistical habits of groups, not causal truths about people.

---
