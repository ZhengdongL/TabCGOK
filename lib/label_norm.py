def normalize_reg_label(label, mu, std):
    norm_label = ((label - mu) / std)
    # norm_label = norm_label.reshape(-1, 1)
    return norm_label