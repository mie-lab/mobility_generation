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
    - Use previous 7 days, predict the next location. 
    - num_encoder_layers: 2; nhead: 4; dim_feedforward: 128; fc_dropout: 0.1 (parameter 2065394): validation loss 2.86, accuracy 42.74% (TODO: tune)
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