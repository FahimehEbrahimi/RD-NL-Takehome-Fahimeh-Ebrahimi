from train import load_model, predict
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pt", type=str, default="/model.pt")
    parser.add_argument("--test_data_path", type=str, default="/data/rd-gen-text-toy-dataset - test.csv")
    args = parser.parse_args()

    model = load_model(args.model_pt)
    test_df = pd.read_csv(args.test_data_path)

    predicted_df = predict(model, test_df)
    predicted_df.to_csv('predicted_df.csv')