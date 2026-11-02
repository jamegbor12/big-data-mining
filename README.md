# Big Data Mining: UK Traffic Accidents and Social Network Analysis

Two analyses from the Big Data Mining and SQL module, MSc Artificial Intelligence and Data
Science, University of Hull, 2026. Author: **Jonathan Amegbor**

---

## Exercise 1 — UK Traffic Accident Analysis

**122,635 accident records** from the Department for Transport road safety database (2018),
queried in **SQLite** across four joined tables: `accident`, `vehicle`, `casualty` and `lsoa`.

### What the analysis found

**Accidents are not evenly distributed in time.** Chi-squared tests rejected uniformity by hour
(χ²(23) = 44,845, p < 0.001) and by day (χ²(6) = 1,397.4, p < 0.001). The peak is **17:00**
(10,615 accidents), with a secondary 08:00 commuting peak. **Friday runs 14.3% above** uniform
expectation; Sunday 21.2% below.

**Pedestrians peak two hours earlier than everyone else.** Of 22,432 pedestrian casualties, the
largest hourly count was at **15:00** (2,404), not 17:00 — consistent with school dismissal and
children walking home unsupervised. This is the single most actionable finding in the study,
because it points at a specific intervention window of 14:30–16:00 on school days.

**The serious-injury environment is dark, fast and rural.** Apriori rule mining on weather, surface,
lighting, junction detail, speed limit and area produced 1,223 frequent itemsets and 726
severity rules. The strongest serious-severity rule was:

```
Light = DarkNoLighting AND Speed = 60  →  Severity = Serious     (lift 4.33)
```

A deliberately low support threshold (0.003) was used to stop rarer fatal and serious patterns
being drowned out by the dominance of slight accidents.

**Clustering answers two different questions.** On 4,132 West Yorkshire records, K-Means (k = 5,
chosen by elbow method and matching the five metropolitan districts) gave broad planning zones;
DBSCAN (eps = 0.1, min_samples = 15) found 22 local hotspots and separated 1,111 noise points
(26.9%). Cluster 8 held only 16 accidents but 2 fatalities — the highest fatal proportion of any
cluster, and a candidate for investigation despite the small sample.

**Classical time series beat gradient boosting on the full-year forecast.**

| Police force | SARIMA RMSE | XGBoost RMSE |
|---|---:|---:|
| Metropolitan | **67.37** | 78.16 |
| West Yorkshire | **23.24** | 29.77 |
| Thames Valley | 20.05 | **19.79** |

SARIMA(0,1,0)(0,1,1,52) was selected by AIC grid search. The honest caveat is stated in the
report: XGBoost's test lags used actual 2019 values, so it represents rolling one-week-ahead
prediction rather than a true 52-week forecast. SARIMA is the fairer full-year comparison;
XGBoost is the better fit for week-to-week monitoring.

**Sparse data limits what forecasting can do.** Daily forecasting for the 30 highest-incident West
Yorkshire LSOAs (276 accidents over 181 days, mean 1.52/day) gave RMSE 1.14 and MAPE 40.73% —
and MAPE is unstable when actual counts are often zero or one. The model is useful for tracking
whether levels are rising or falling, not for predicting individual days.

### Recommendations drawn from it

School-area speed management during dismissal; severity reduction on unlit 60 mph rural roads;
DBSCAN hotspot targeting for West Yorkshire engineering decisions; Friday afternoon enforcement;
and motorcycle interventions tailored by engine category, since 125cc-and-under riders show an
urban commuting profile while over-500cc riders show a weekend and rural one.

### The limitation that matters

No exposure data. Without traffic volumes, pedestrian flows, motorcycle mileage or road length,
these are accident **counts**, not rates. Urban clusters may reflect high traffic exposure rather
than higher risk per journey. Every claim above should be read with that in mind.

---

## Exercise 2 — Social Network Analysis

The Facebook combined ego-network from Stanford SNAP: **4,039 nodes, 88,234 edges**, anonymised
and undirected, analysed in NetworkX.

| Metric | Value |
|---|---:|
| Density | 0.0108 |
| Average degree | 43.69 |
| Average clustering coefficient | 0.6055 |
| Diameter | 8 |

Sparse overall, but highly clustered — friends of friends are friends — and a diameter of 8
confirms the small-world property.

**Bridge detection.** Edge betweenness centrality was computed for all 88,234 edges. The
distribution is strongly right-skewed: most ties are ordinary local connections, while a handful
carry a disproportionate share of shortest paths. The top edge, (107, 1684), scored 0.171 —
these are the ties linking otherwise separate social circles such as school, work or family.

**Community detection, two ways.**

| Algorithm | Communities | Average size | Largest | Modularity |
|---|---:|---:|---:|---:|
| Label Propagation | 44 | 91.8 | 1,030 | 0.7368 |
| Louvain | 16 | 252.4 | 548 | **0.8349** |

Both found strong community structure, at different resolutions. Label Propagation gives a finer
partition into friendship circles; Louvain gives fewer, better-separated communities and the
higher modularity.

---

## Repository structure

```
notebooks/
  01_uk_traffic_accident_analysis.ipynb     SQL, chi-squared, Apriori, clustering, forecasting
  02_social_network_analysis.ipynb          NetworkX, centrality, community detection
```

Both notebooks are committed **with outputs**, so every table and figure can be read without
running anything.

---

## Reproducing

```bash
pip install -r requirements.txt
jupyter lab
```

**The data is not in this repository.** Both sources are public:

- **UK road safety data** — [Department for Transport](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data),
  2017–2020 release. Notebook 1 reads the `accident`, `vehicle`, `casualty` and `lsoa` tables
  from a SQLite database built from those CSVs.
- **Facebook ego-network** — [Stanford SNAP](https://snap.stanford.edu/data/ego-Facebook.html),
  the `facebook_combined.txt` edge list.

---

## Licence

MIT — see [LICENSE](LICENSE).
