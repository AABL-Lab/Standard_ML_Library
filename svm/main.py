import pandas as pd
from sklearn import svm
from IPython import embed

def main():
    path = "~/workspace/hri_gym/hri_gym_server/data_analysis"
    focused_df = pd.read_pickle(path+"/test_focused_df")
    unfocused_df = pd.read_pickle(path+"/test_unfocused_df")
    embed()
    pass

if __name__ == "__main__":
    main()