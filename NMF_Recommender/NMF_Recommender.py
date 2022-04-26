import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error


class NMFRecommender:

    def __init__(self, random_state=15, rank=3, maxiter=200, tol=1e-3):
        """The parameter values for the algorithm"""
        # set attributes
        self.random_state = random_state
        self.rank = rank
        self.maxiter = maxiter
        self.tol = tol
  
    def initialize_matrices(self, m, n):
        """Initialize the W and H matrices"""
        # set random seed and intialize matrices 
        np.random.seed(self.random_state)
        self.W = np.random.random((m, self.rank))
        self.H = np.random.random((self.rank, n))

        return self.W, self.H
        
    def compute_loss(self, V, W, H):
        """Computes the loss of the algorithm according to the frobenius norm"""

        # compute loss using frobeneous norm
        return np.linalg.norm(V - W@H, ord='fro') 
    
    def update_matrices(self, V, W, H):
        """The multiplicative update step to update W and H"""
        # update W
        H *= ((W.T@V)/(W.T@W@H))
        W *= ((V@H.T)/(W@H@H.T))

        self.W = W.copy()
        self.H = H.copy()

        return self.W, self.H
      
    def fit(self, V):
        """Fits W and H weight matrices according to the multiplicative update 
        algorithm. Return W and H"""


        self.initialize_matrices(*V.shape)

        for _  in range(self.maxiter):
            # update W and H
            self.update_matrices(V, self.W, self.H)

            # check losss
            if self.compute_loss(V, self.W, self.H) < self.tol:
                break

        return self.W, self.H
        

    def reconstruct(self, W, H):
        """Reconstructs the V matrix for comparison against the original V 
        matrix"""

        # return matrix multiplication
        return W@H 

def prob4():
    """Run NMF recommender on the grocery store example"""
    V = np.array([[0,1,0,1,2,2],
                  [2,3,1,1,2,2],
                  [1,1,1,0,1,1],
                  [0,2,3,4,1,1],
                  [0,0,0,0,1,0]])

    # instantiate nmf
    nmf = NMFRecommender(rank=2)
    # fit
    W, H = nmf.fit(V)
    # get number of peole and return
    num_people = np.sum(np.argmax(H, axis=0) == np.ones(H.shape[1]))
    return W, H, num_people

def prob5():
    """Calculate the rank and run NMF
    """
    # read in the data frame with the user id as the index.
    df = pd.read_csv("artist_user.csv", index_col=0)
    # calculate fro norm and benchmark
    fro_norm = np.linalg.norm(df, ord='fro')
    benchmark = fro_norm * 0.0001
    rank = 3
    while rank <= df.shape[1]:
        # calculate NMF
        model=NMF(n_components=rank, init='random', random_state=0, max_iter=1000)
        W = model.fit_transform(df)
        H = model.components_
        # reconstruct
        V = W@H
        # calculate root mean squared error
        RMSE = np.sqrt(mean_squared_error(df, V))
        # check against benchmark
        if RMSE < benchmark:
            break 
        rank += 1

    return rank, V

def discover_weekly(user_id, V):
    """
    Create the recommended weekly 30 list for a given user
    """
    # get CSVs
    artists = pd.read_csv("artists.csv", index_col=0)
    artists_users = pd.read_csv("artist_user.csv", index_col=0)

    # get artist ids (index of artist datafram)
    artist_ids = artists.index.values
    # get user ids (we need the index of the user id)
    user_ids = list(artists_users.index)
    # get user index
    user_index = user_ids.index(user_id)

    # get the row of V corresponding to the user
    user = V[user_index]

    # get the indicies of the best artists
    best_artists_idx = np.argsort(user)[::-1]
    # instantiate the list
    best_artists = []
    for idx in best_artists_idx:
        # get artist id from index
        artist_id = artist_ids[idx]
        # check that in the original dataframe the user has not listend to that artist
        if artists_users.loc[user_id, str(artist_id)] == 0:
            # append the artist name to our list
            best_artists.append(list(artists.loc[artist_id]))

        # check for 30
        if len(best_artists) == 30:
            break

    return np.array(best_artists)

def main(key):
    
    if key == "4":
        W, H, num = prob4()
        print('W:')
        print(W)
        print('H:')
        print(H)
        print('num people:')
        print(num)

    elif key == "5":
        rank, V = prob5()
        np.save("V_test.npy", V)
        print(rank)
        print(V)

        return rank, V

    elif key == "6":
        V = np.load("V_test.npy")
        best_artists = discover_weekly(2, V)
        print(best_artists)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        pass
    elif len(sys.argv) == 2:
        main(sys.argv[-1])