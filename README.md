# A heuristic approach to identify cancdidate drugs to use in combination for CRC treatment

## Data Processing
   - The notebook EDA contains the code to process the raw drugs, cell lines gene expression, and mutation data for downstream procedures

## Train FastJTNNpy3 : Junction Tree Variational Autoencoder for Molecular Graph Generation
   - The TrainJTNN notebook in FastJTNNpy3 folder contains the code to train the Junction Tree Variational Autoencoder and obtain the latent representation for the drugs

## Train GeneVAE : an Overparameterized Autoencoder for cell line gene expression profile encoding

   - The Train_GeneAE notebook contains the code to train the autoencoder and obtain the latent representation of the cell lines
   
## Train MLP to predict the log2-fold change vialbility in a cell line given a drug
   - The MLP notebook contains the code to train the model
   - The Inference notebook contains the code to evaluate the model and run the heursitic algorith to identify candidate drugs
   


 