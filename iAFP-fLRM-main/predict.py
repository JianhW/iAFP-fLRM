from model import Net
from data_loader import load_Ind_data
from train import evaluate
import torch
import pandas as pd
device = torch.device("cuda:0")


def predict(test_file, model_path):

    model = Net()
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    test_iter = load_Ind_data(test_file)
    all_predictions = []
    all_probabilities = []
    all_positive_probs = []
    all_labels = []
    all_sequences = []

    tmp = pd.read_csv(test_file, header=None)
    seqs = tmp[0].values.tolist()
    true_labels = tmp[1].values.tolist() if tmp.shape[1] > 1 else None
    with torch.no_grad():
        batch_idx = 0
        for x, pos, hf, y in test_iter:
            outputs, _ = model(x, pos, hf)


            probabilities = torch.softmax(outputs, dim=1)


            predictions = torch.argmax(outputs, dim=1)


            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_positive_probs.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(y.cpu().numpy())


            batch_size = x.size(0)
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(seqs))
            batch_sequences = seqs[start_idx:end_idx]
            all_sequences.extend(batch_sequences)

            batch_idx += 1

    results_df = pd.DataFrame({
        'Sequence': all_sequences,
        'True_Label': all_labels,
        'Predicted_Label': all_predictions,
        'Positive_Probability': all_positive_probs,
        'Negative_Probability': [probs[0] for probs in all_probabilities]
    })


    results_df['Predicted_Class'] = results_df['Predicted_Label'].map({0: 'Negative', 1: 'Positive'})
    results_df['True_Class'] = results_df['True_Label'].map({0: 'Negative', 1: 'Positive'})


    results_df['Confidence'] = results_df[['Positive_Probability', 'Negative_Probability']].max(axis=1)


    results_df['Correct'] = (results_df['True_Label'] == results_df['Predicted_Label']).astype(int)

    return results_df

if __name__ == "__main__":

        results = predict("Ind.csv", "iAFP-fLRM.pt")
        print(results)
