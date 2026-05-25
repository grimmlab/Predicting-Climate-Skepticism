<div align="left"><img src="https://github.com/user-attachments/assets/7b305774-61aa-434e-8bf7-001a065703ec" height="160"/></div>

Predicting Climate Skepticism: A Machine Learning Analysis of Climate Change Belief, Awareness, and Human Responsibility using Explainable Artificial Intelligence
===============================================================================================================================
Climate skepticism shapes public opinion and increasingly influences political decision-making, yet its underlying drivers remain incompletely understood. Existing research typically examines individual correlates in isolation, limiting insights into their relative importance across different dimensions of skepticism. Here, we apply a machine learning framework based on gradient boosting (`XGBoost`) to predict three core dimensions of climate skepticism—BELIEF, AWARENESS, and perceived HUMAN RESPONSIBILITY—using a large-scale survey dataset (N = 14,825) comprising 127 features spanning demographics, preferences, and attitudes. Model performance indicates substantial predictive power across all dimensions (MCC up to 0.62; R<sup>2</sup> up to 0.73). 

We find that a small set of attitudinal feature groups, in particular Climate Opinion, Policy Actions, and Personal Conviction, accounts for most of the predictive performance, with climate-related opinions alone achieving nearly the same accuracy as the full feature set. In contrast, demographic and preference-based variables contribute little to prediction. Feature-level analysis using SHapley Additive exPlanations (`SHAP`) reveals a consistent core of beliefs—such as perceived climate risks, trust in scientific evidence, and views on media representation—that jointly shape all three dimensions of skepticism.

These findings suggest that climate skepticism is primarily driven by coherent belief systems rather than by socio-demographic characteristics. By identifying the most influential drivers across multiple dimensions, our results provide a basis for more targeted and effective climate communication and policy design.

Contributors
----------------------------------------

This repository is developed and maintained by members of the [Bioinformatics lab](https://bit.cs.tum.de) led by [Prof. Dr. Dominik Grimm](https://bit.cs.tum.de/team/dominik-grimm/):
- [Josef Eiglsperger, M.Sc.](https://bit.cs.tum.de/team/josef-eiglsperger/)

Citation
---------------------
**Machine learning predicts climate skepticism and identifies individual driving features with explainable artificial intelligence.** <br />
J Eiglsperger, M Speckner, A Pondorfer,  DG Grimm, SJ Goerg. <br />
*Currently under review, 2026.* <br />
