# mobility_generation

## Design choices

Preprocessing sp (4,811,539):
- Filter duplicates: remove sps or tpls that have overlapping timeline (4,804,194). 
- Activity definition (activity: 3,589,215, non-activity: 1,214,979):
    - waiting
    - "unknown" and duration < 25min
-  User filtering (users: 2,113. sp: 1,578,245):
    - tracked for more than 50 days
    - quality during tracking: quality for every 5 week shall have min quality > 0.5 and mean quality > 0.6 
- Spatial filter: Only sp within Switzerland (sp: 1,460,189).
- Filter for activity sp (sp: 1,094,017). 
- Filter non-location sp (sp: 1,094,017). 
- Merge sp with time gap shorter than 1 minutes (sp: 1,079,922).
- Final user: 2112. Final sp: 1,079,922.

Generating locations:
- DBSCAN algorithm (unique locs: 162,303)
    - epsilon=20
    - num_samples=1
    - agg_level="dataset"
- Location spatial binning using s2geometry: 
    - Single level:
        - Level 14 (mean 0.32 km2 globally): 174,434 locations covering Switzerland (41,285 km2). Original locations are projected into 28,742 binned locations.
        - Level 13 (mean 1.27 km2 globally): 44,106 locations covering Switzerland. Original locations are projected into 14,881 binned locations.
    - Hierarchical: 
        - [10, 14]: 142,575 locations covering Switzerland. Original locations are projected into 28,742 binned locations.
        - [10, 13]: 39,177 locations covering Switzerland. Original locations are projected into 14,881 binned locations.
        

- Definition of activity behavior:
    - Preserve the self-transitions
    - Activity duration: activity duration + the transit duration to reach the location (finished_at - previous finished_at)

- TODO: 
    - POI generation check
## Generation

### Next location prediction. 

User split 6:2:2 according to time.

- MHSA: 
    - Use previous 7 days, predict the next location. 
    - hyperparameter: 
        - parameter 2065394: num_encoder_layers: 2; nhead: 4; dim_feedforward: 128; fc_dropout: 0.1: validation loss 2.86, accuracy 42.74% (TODO: tune)
    - Test runs (small) for level 10-14 with features:
        - None (1,384,999):
            - validation acc@1: 42.43; Test acc@1 = 39.34 f1 = 31.20 mrr = 51.81
            - validation acc@1: 42.58; Test acc@1 = 39.59 f1 = 31.94 mrr = 51.83
        - User (1,417,063): 
            - validation acc@1: 43.25; Test acc@1 = 40.05 f1 = 31.90 mrr = 52.56
            - validation acc@1: 43.01; Test acc@1 = 39.73 f1 = 32.22 mrr = 52.40
        - User + poi (1,420,311):
            - validation acc@1: 43.27; Test acc@1 = 40.09 f1 = 32.18 mrr = 52.70
            - validation acc@1: 43.35; Test acc@1 = 40.29 f1 = 31.75 mrr = 52.71
        - User + Time (1,419,495):
            - validation acc@1: 45.55; Test acc@1 = 42.53 f1 = 34.91 mrr = 54.16
            - validation acc@1: 45.30; Test acc@1 = 42.08 f1 = 33.87 mrr = 53.84
        - User + Duration (1,423,271):
            - validation acc@1: 45.16; Test acc@1 = 41.82 f1 = 33.90 mrr = 53.71
            - validation acc@1: 45.30; Test acc@1 = 42.04 f1 = 34.01 mrr = 53.76
        - User + Duration + Time (1,425,703): 
            - validation acc@1: 48.02; Test acc@1 = 44.62 f1 = 36.35 mrr = 55.31
            - validation acc@1: 48.74; Test acc@1 = 45.29 f1 = 37.55 mrr = 55.78
            - validation acc@1: 48.30; Test acc@1 = 45.23 f1 = 37.68 mrr = 55.73
            - validation acc@1: 48.08; Test acc@1 = 44.72 f1 = 36.43 mrr = 55.44
        - All (1,428,951)
            - validation acc@1: 48.78; Test acc@1 = 45.69 f1 = 38.13 mrr = 56.03
            - validation acc@1: 48.66; Test acc@1 = 45.36 f1 = 37.89 mrr = 55.82
            - validation acc@1: 48.72; Test acc@1 = 45.70 f1 = 37.73 mrr = 56.03
            - validation acc@1: 48.48; Test acc@1 = 45.28 f1 = 37.78 mrr = 55.74
- Markov: 
    - Train user model with train and validation (6+2) sequences. 
    - Each next location is sampled from the top3 most likely locations according to the markov transition matrix. If no prior knowledge, next location sampled from the top3 overall most visited locations.


Trained model autoregressively generate 50 locations for each test sequence. 

TODO:
- hyper-parameter search
- implement beam search

### With mechanistic individual models. 

- DTEPR (TODO:)
- TimeGeo (TODO:)
- Container (TODO:)

### With neural generative models.

Use 4 weeks as input (TODO: Tune weeks)

- SeqGAN (TODO:) 
    - implement
- moveSim (TODO:) 
    - sample from population distribution
    - functional similarity of POIs
    - period loss to account for variable activity time prediction 
- VOLUNTEER (TODO:)
    - implement
- DiffSeq (TODO:)
    - implement

## Metrics (TODO: verify)

- Travel distance 
- Radius of Gyration
- Number of repetitive locations
- Overall visitation frequency
- Individual visitation frequency
- TODO: Activity duration
- TODO: Daily visited locations
- TODO: motifs distribution

## TODO:
- Generation with predictive models, implement beam search
- Implement classical generation model 
- ensure evaluation is on the same test dataset/sequence