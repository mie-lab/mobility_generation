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
        - parameter 2065394: num_encoder_layers: 2; nhead: 4; dim_feedforward: 128; fc_dropout: 0.1 (TODO: tune)
    - Test runs (small) for level 10-14 with features:
        - None (1,418,280):
            - validation acc@1: 42.89; Test acc@1 = 39.80 f1 = 31.51 mrr = 52.15
            - validation acc@1: 42.57; Test acc@1 = 39.53 f1 = 31.75 mrr = 51.74
        - User (1,450,344):  
            - validation acc@1: 42.83; Test acc@1 = 39.71 f1 = 32.08 mrr = 52.19
            - validation acc@1: 42.69; Test acc@1 = 39.72 f1 = 31.80 mrr = 52.17
        - User + poi (1,453,592):
            - validation acc@1: 43.34; Test acc@1 = 40.07 f1 = 31.59 mrr = 52.67
            - validation acc@1: 43.37; Test acc@1 = 40.54 f1 = 32.56 mrr = 52.98
        - User + Time (1,452,776):
            - validation acc@1: 46.19; Test acc@1 = 43.08 f1 = 35.30 mrr = 54.47
            - validation acc@1: 45.38; Test acc@1 = 42.29 f1 = 34.42 mrr = 53.95
        - User + Duration (1,456,552):
            - validation acc@1: 45.31; Test acc@1 = 42.18 f1 = 34.04 mrr = 53.81
            - validation acc@1: 44.99; Test acc@1 = 41.92 f1 = 34.20 mrr = 53.73
        - User + Duration + Time (1,458,984): 
            - validation acc@1: 48.38; Test acc@1 = 45.03 f1 = 37.35 mrr = 55.60
            - validation acc@1: 48.86; Test acc@1 = 45.20 f1 = 37.69 mrr = 55.59
        - All (1,462,232)
            - validation acc@1: 48.79; Test acc@1 = 45.50 f1 = 37.57 mrr = 55.81
            - validation acc@1: 48.99; Test acc@1 = 45.38 f1 = 37.97 mrr = 55.79

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
- moveSim
    - Generator: 
        - Samples from emperical visit frequency
        - Inputs Physical distance, function similarity, and historical transition matrices
        - Accepts duration and location as input. Output duration and location of the next step. 
    - Discriminator:
        - Accepts duration and location as input. Output p of real
    - Loss:
        - implemented reward based on rollout
        - Based on CrossEntropyLoss and MSELoss
        - Included distance loss; decide to not include period loss
    - Training:
        Learning signal weak -> Generator capacity low

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