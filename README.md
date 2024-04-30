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

## Generation

### Next location prediction. 

User split 6:2:2 according to time.

- MHSA: 
    - Use GPT2 implementation
    - Use previous 7 days, predict the next location. 
    - hyperparameter: 
    - Test runs (small) for level 10-14 with features (checked that all features are useful):

- Markov: 
    - Train user model with train and validation (6+2) sequences. 
    - Top-k sampling: Each next location is sampled from the top3 most likely locations according to the markov transition matrix. 
        - If no prior knowledge, next location sampled from the top3 overall most visited locations.


Trained model autoregressively generate 50 locations for each test sequence. 

TODO:
- hyper-parameter search

### With mechanistic individual models. 

- EPR
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
- DiffSeq-v2
    - loss shall be < 0.01
    - V2 shall be initialized from random
    - We only consider V2 
    - pretrain slightly better

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
- Implement classical generation model 