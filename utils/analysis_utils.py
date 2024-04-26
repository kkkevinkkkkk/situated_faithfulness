import numpy as np
def get_single_status(df, interal_correctness, document_correctness, current_correctness):
    return df[(df["internal_correctness"] == interal_correctness) & (df["document_correctness"] == document_correctness) & (df["current_correctness"] == current_correctness)]

def get_all_status(df):
    faithfulness_map = {
        (0, 0, 0): "Noise: Both wrong and result is wrong",
        (0, 0, 1): "Noise: doc and model are wrong but result is correct",
        (0, 1, 0): "Not Faithful: stay wrong",
        (0, 1, 1): "Situated Faithful: should trust doc and do trust doc",
        (1, 0, 0): "Bad Faithful: should trust model but trust doc",
        (1, 0, 1): "Situated Faithful: should trust model and trust model",
        (1, 1, 0): "Noise: doc and model are correct but result is wrong",
        (1, 1, 1): "Situated Faithful:: Both correct and result is correct",
    }
    for internal_correctness in [0, 1]:
        for document_correctness in [0, 1]:
            for current_correctness in [0, 1]:
                print(faithfulness_map[(internal_correctness, document_correctness, current_correctness)])
                print(internal_correctness, document_correctness, current_correctness, len(get_single_status(df, internal_correctness, document_correctness, current_correctness)))


def calculate_ece_score(confidence_scores, correctness_scores, n_bins=10):
    # calculate ece score
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_boundaries[-1] += 1e-4
    ece_score = 0.0
    total_sample = len(confidence_scores)
    total_props = 0
    for bin_idx in range(n_bins):
        bin_mask = (confidence_scores >= bin_boundaries[bin_idx]) & (confidence_scores < bin_boundaries[bin_idx + 1])
        bin_confidence_scores = confidence_scores[bin_mask]
        bin_correctness_scores = correctness_scores[bin_mask]
        if len(bin_confidence_scores) == 0:
            continue
        bin_prop = len(bin_confidence_scores) / total_sample
        total_props += bin_prop
        bin_err = np.abs(np.mean(bin_correctness_scores) - np.mean(bin_confidence_scores))
        ece_score += bin_err * bin_prop
    return ece_score