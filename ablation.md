
Quali componenti abbiamo? Spaziale e Temporale
Ci concentriamo solamente sulla configurazione con pochi pesi e dimostriamo che fa meglio rispetto alle alternative, almeno su questa scala ridotta

- Spaziale: gMLP, GAT
- Temporale: 
    - Core: LRU, RNN, GRU, LSTM
    - Convoluzione: Si, No 

| Ex | Conv+LRU+gMLP | LRU+gMLP | Conv+RNN+gMLP | Conv+RNN      | Conv+GRU+gMLP |Conv+GRU         |Conv+LSTM+gMLP    | LRU+GAT |
|----|---------------|----------|---------------|---------------|---------------|-----------------|------------------|---------|
|  1 |**3.050**      |3.540     |4.900          |4.360          |2.360          |3.550            |2.640             |1.141    |
|  2 |**3.100**      |0.753     |3.390          |4.900          |3.060          |5.030            |2.370             |1.528    |
|  3 |**3.670**      |0.210     |3.740          |4.900          |2.310          |2.440            |2.430             |0.845    |
|  4 |**3.110**      |3.540     |4.800          |4.330          |1.680          |2.300            |3.560             |0.468    |
|  m |3.2325         |          |               |               |2.233          |3.330            |2.750             |         |