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
- Location spatial binning: Use s2geometry to bin locations. Level 14 (13). After binning 174434 (44106) locations covering Switzerland. The original 62673 locations are projected into 15841 (8964) binned locations. TODO:
    - hierarchical s2 location generation.  

## Generation

### With next location prediction neural networks. 

Trained model iteratively generate 50 locations for each test sequence. User split 6:2:2 according to time.
- MHSA: Use previous 7 days, predict the next location. 
    - num_encoder_layers: 2; nhead: 4; dim_feedforward: 128; fc_dropout: 0.1 (parameter 2065394): validation loss 2.86, accuracy 42.74% (TOTEST)
    - (TODO:) hyper-parameter search
    - (TODO:) implement beam search
- Markov: Train user model with train and validation (6+2) sequence. Each next location is sampled from the top3 most likely locations according to the markov transition matrix. If no prior knowledge, next location sampled from the top3 overall most visited locations.

### With mechanistic individual models. 

- DTEPR (TODO:)
- TimeGeo (TODO:)
- Container (TODO:)

### With neural generative models.
Use 4 weeks as input (TODO: Tune weeks)

- SeqGAN (TODO:)
    - 
- moveSim (TODO:)
    - implement
- VOLUNTEER (TODO:)
    - implement
- DiffSeq (TODO:)

## Metrics

- (TODO:) Travel distance 
- Radius of Gyration
- Activity duration 
- Daily visited locations
- motifs distribution

## TODO:
- [ ] Generation with predictive models, implement beam search
- [ ] Implement classical generation model 
- definition of locations: activity duration or transit duration or their sum?