# mobility_generation

## Design choices

Preprocessing:
- Staypoints with all activity except waiting is considering activity staypoints.
-  User filtering (after filtering 2094 users):
    - tracked for more than 50 days
    - quality during tracking: calculate quality for every 5 week. Considered user shall have min quality > 0.5 and mean quality > 0.6 
- Only consider staypoints within Switzerland (after filtering 1440822 staypoints).
- Merging staypoints withh time gap shorter than 1 minutes (After merging 1197153 staypoints).

Location generation:
- DBSCAN algorithm parameter: epsilon=20, num_samples=2, agg_level="dataset". After generation 62673 locations
- Location spatial binning: Use s2geometry to bin locations. Level 13. After binning 44106 locations covering Switzerland. The original 62673 locations are projected into 8964 binned locations. TODO:
    - hierarchical s2 location generation.  

Next location prediction. Trained model iteratively generate 50 locations for each test sequence. 
- MHSA: Use previous 7 days, predict the next location. User split 6:2:2 according to time. TODO:
    - compare mask performances 
    - implement the input POI features 
    - hyperparameeter search
- Markov: Train user model with the (6+2) sequence. 